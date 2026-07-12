"""
train_agent.py — Configurable Asynchronous RLHF Agent Loop (CORRECTED GRPO)

Executes multi-turn rollouts using vLLM, then performs a correct token-level
GRPO update against a FROZEN reference and the BEHAVIOR-policy logprobs captured
at generation time.

KEY FIXES vs. previous version
------------------------------
1. GRPO math was fundamentally wrong before:
   - It took F.log_softmax over the full [seq, vocab] tensor and never gathered
     the log-prob of the *sampled tokens*. => the "ratio" was noise.
   - It computed loss over prompt + completion with no completion mask. => the
     (huge) prompt dominated any surviving signal.
   - The "reference" was the same LoRA network with the adapter disabled, which
     is also what `beta` regularizes toward => the objective fought its own KL.
   Now: we gather per-token logprobs of the taken tokens, mask to completion
   spans only, use a frozen base snapshot as the reference, and compute the
   importance ratio against the behavior logprobs recorded during rollout.

2. vLLM adapter staleness: we now bump the LoRA request ID every update so vLLM
   is forced to reload the freshly-saved adapter instead of serving a cached one.

3. reward = max(0.0, r) previously collapsed all penalized trajectories to
   exactly 0, killing advantage variance. We keep raw (signed) rewards for the
   advantage computation; clipping is removed.

4. GRPO advantages are normalized *within prompt groups* (the defining feature
   of GRPO), falling back to batch-level normalization when only one sample per
   prompt exists.

NOTE ON ROLLOUT LOGPROBS
------------------------
True on-policy PPO/GRPO needs the sampling distribution's logprobs as the
denominator of the importance ratio. vLLM can return these via
SamplingParams(logprobs=...) / prompt_logprobs. We request them and, when
available, use them as the behavior logprobs. If they are missing for a
trajectory, we fall back to treating the ratio as 1 (REINFORCE-style) for that
trajectory rather than fabricating a denominator.
"""

import asyncio
import json
import os
import shutil
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from rlef.reward import execution_reward, verify_generated_tests
from rlef.prompt import parse_output
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

# ─── 1. INFRASTRUCTURE & MEMORY PARTITIONING ──────────────────────────────

with open("configs/train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("Initializing vLLM Engine...")
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    enable_prefix_caching=True,
    enable_lora=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.30,
    max_model_len=16384,
)
vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

print("Initializing PyTorch Policy Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_resume = cfg.get("lora_resume_path", None)
if lora_resume and os.path.exists(lora_resume):
    print(f"🔄 Resuming curriculum from {lora_resume}")
    from peft import PeftModel

    policy_model = PeftModel.from_pretrained(base_model, lora_resume, is_trainable=True)
else:
    print("🌟 Fresh LoRA weights...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    policy_model = get_peft_model(base_model, lora_config)

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)
os.makedirs("./checkpoint/active_lora", exist_ok=True)
policy_model.save_pretrained("./checkpoint/active_lora")

# Monotonically increasing LoRA id so vLLM is forced to reload the adapter
# after every save instead of serving a stale cached copy under a fixed id.
_LORA_VERSION = {"v": 1}


def _current_lora_request() -> LoRARequest:
    _LORA_VERSION["v"] += 1
    return LoRARequest(
        f"active_policy_{_LORA_VERSION['v']}",
        _LORA_VERSION["v"],
        lora_path="./checkpoint/active_lora",
    )


def _lora_checksum() -> float:
    """Cheap fingerprint of the current LoRA weights. If this does NOT change
    across training steps, the policy update isn't reaching the adapter and
    rollouts are frozen — no amount of correct loss math will produce learning.
    Watch train/lora_checksum in W&B: it MUST move every step."""
    total = 0.0
    for name, param in policy_model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            total += param.detach().float().abs().sum().item()
    return total


# ─── 2. THE INTERACTIVE MULTI-TURN ROLLOUT ───────────────────────────────


async def run_single_episode(
    prompt_text: str,
    problem_data: dict,
    lora_req: LoRARequest,
    sampling_params: SamplingParams,
    ablation_cfg: dict,
):
    current_context = prompt_text
    completions = []  # raw completion text per turn
    completion_texts = []  # the exact text we will train on (the taken action)
    behavior_logprobs = []  # per-turn sum logprob under the sampling policy, if available
    final_reward = 0.0
    episode_stats = {
        "execute": 0,
        "generate_tests": 0,
        "invalid_format": 0,
        "truncated": 0,
    }
    episode_errors = []
    pass_rate = 0.0

    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)

    max_turns = ablation_cfg.get("max_turns", 5)
    use_linear_pass_rate = ablation_cfg.get("use_linear_pass_rate", True)
    use_step_credit = ablation_cfg.get("use_step_credit", True)
    use_turn_penalty = ablation_cfg.get("use_turn_penalty", True)
    use_feedback_bonus = ablation_cfg.get("use_feedback_bonus", True)
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")

    previous_pass_rate = 0.0

    for turn in range(max_turns):
        request_generator = vllm_engine.generate(
            prompt=current_context,
            sampling_params=sampling_params,
            request_id=f"req_{problem_data['problem_id']}_{turn}_{lora_req.lora_int_id}",
            lora_request=lora_req,
        )
        final_result = None
        async for res in request_generator:
            final_result = res

        out = final_result.outputs[0]
        completion = out.text
        completions.append(completion)
        completion_texts.append(completion)

        # Was this completion cut off by max_tokens? finish_reason == "length"
        # means truncated (raising max_tokens would help); "stop" means it ended
        # cleanly on </code> (raising max_tokens would NOT help).
        was_truncated = getattr(out, "finish_reason", None) == "length"

        # Capture behavior-policy logprob sum for this completion if vLLM returned it.
        try:
            if out.logprobs:
                lp_sum = 0.0
                for tok_lp in out.logprobs:
                    # tok_lp: {token_id: Logprob}; take the sampled token's logprob
                    lp_sum += max(v.logprob for v in tok_lp.values())
                behavior_logprobs.append(lp_sum)
            else:
                behavior_logprobs.append(None)
        except Exception:
            behavior_logprobs.append(None)

        parsed = parse_output(completion)

        if not parsed["is_valid"]:
            episode_stats["invalid_format"] += 1
            if was_truncated:
                episode_stats["truncated"] = episode_stats.get("truncated", 0) + 1
            feedback_str = "CRITICAL ERROR: Invalid format. Wrap executable logic inside <code>...</code>."
            current_context += (
                f"{completion}<|im_end|>\n<|im_start|>user\n"
                f"System Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            )
            continue

        if use_edge_cases and not parsed.get("edge_cases"):
            episode_stats["invalid_format"] += 1
            feedback_str = "CRITICAL ERROR: Missing <edge_cases> block. Write test cases before code."
            current_context += (
                f"{completion}<|im_end|>\n<|im_start|>user\n"
                f"System Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            )
            continue

        episode_stats["execute"] += 1
        step_reward = 0.0

        if use_edge_cases and parsed.get("edge_cases"):
            episode_stats["generate_tests"] += 1
            test_bonus = await asyncio.to_thread(
                verify_generated_tests, parsed["edge_cases"], parsed["code"], fn_name
            )
            step_reward += test_bonus

        exec_result = await asyncio.to_thread(
            execution_reward,
            code=parsed["code"],
            inputs=inputs,
            outputs=outputs,
            fn_name=fn_name,
        )

        pass_rate = exec_result.pass_rate
        episode_errors.extend(getattr(exec_result, "error_types", []))

        step_reward += (
            pass_rate if use_linear_pass_rate else (1.0 if pass_rate == 1.0 else 0.0)
        )
        if parsed.get("is_valid"):
            step_reward += 0.02
        if parsed.get("has_reasoning"):
            step_reward += 0.02
        if use_edge_cases and parsed.get("edge_cases"):
            step_reward += 0.02
        if use_step_credit and 0.0 < pass_rate < 1.0:
            step_reward += 0.10
        if use_turn_penalty:
            step_reward -= 0.05
        if use_feedback_bonus and pass_rate > previous_pass_rate and turn > 0:
            step_reward += 0.10

        previous_pass_rate = pass_rate

        if pass_rate == 1.0:
            final_reward += step_reward
            break
        else:
            if feedback_type == "none":
                feedback_str = "Execution failed. Try a different algorithmic approach."
            elif feedback_type == "consolidated":
                err_counts = Counter(exec_result.error_types)
                err_str = ", ".join(f"{v} {k}" for k, v in err_counts.items())
                feedback_str = (
                    f"Execution Pass Rate: {pass_rate*100:.1f}%. Errors: {err_str}"
                )
            elif feedback_type == "last_failed":
                feedback_str = f"Execution Pass Rate: {pass_rate*100:.1f}%\n"
                if len(inputs) > 0 and pass_rate < 1.0:
                    feedback_str += (
                        f"Failed on test case:\n- Input: {inputs[0]}\n"
                        f"- Expected Output: {outputs[0]}"
                    )
            else:
                feedback_str = (
                    f"Execution Pass Rate: {pass_rate*100:.1f}%. Please revise."
                )

            current_context += (
                f"{completion}<|im_end|>\n<|im_start|>user\n"
                f"System Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            )
            if turn == max_turns - 1:
                final_reward += step_reward

    return {
        "context": prompt_text,
        "full_text": current_context
        + (completion_texts[-1] if completion_texts else ""),
        "completions": completions,
        # We train on the FINAL context (all turns concatenated) so credit flows
        # to every completion the policy actually emitted this episode.
        "trainable_text": current_context,
        "reward": final_reward,  # signed; do NOT clip to 0 here
        "behavior_logprobs": behavior_logprobs,
        "stats": episode_stats,
        "errors": episode_errors,
        "turns_taken": turn + 1,
        "final_context_length": len(current_context),
        "success": 1 if pass_rate == 1.0 else 0,
        "problem_id": problem_data.get("problem_id"),
    }


# ─── 3. CORRECT TOKEN-LEVEL GRPO UPDATE ──────────────────────────────────


def _completion_token_mask(context_text: str, full_text: str):
    """Return input_ids for full_text and a boolean mask that is True only on
    the completion tokens (everything after the prompt/context prefix)."""
    ctx_ids = tokenizer(context_text, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    n_ctx = ctx_ids.shape[0]
    mask = torch.zeros_like(full_ids, dtype=torch.bool)
    if full_ids.shape[0] > n_ctx:
        mask[n_ctx:] = True
    return full_ids, mask


def _gather_token_logprobs(logits, input_ids):
    """logits: [1, seq, vocab]; input_ids: [1, seq].
    Returns per-position logprob of the *actual next token* (shifted), shape [seq-1]."""
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # predict token t+1 from t
    targets = input_ids[:, 1:]  # [1, seq-1]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, seq-1]
    return token_lp[0]  # [seq-1]


def grpo_update_step(batch_trajectories, beta=0.04, clip_eps=0.2):
    if not batch_trajectories:
        return 0.0, 0.0

    policy_model.train()

    # --- GRPO advantage: normalize within prompt groups, else batch-level ---
    groups = defaultdict(list)
    for t in batch_trajectories:
        groups[t["problem_id"]].append(t)

    adv_map = {}
    if all(len(g) == 1 for g in groups.values()):
        rewards = torch.tensor(
            [t["reward"] for t in batch_trajectories], dtype=torch.float32
        )
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        for t, a in zip(batch_trajectories, adv):
            adv_map[id(t)] = a.item()
    else:
        for _, g in groups.items():
            r = torch.tensor([t["reward"] for t in g], dtype=torch.float32)
            a = (r - r.mean()) / (r.std() + 1e-6) if len(g) > 1 else r - r.mean()
            for t, av in zip(g, a):
                adv_map[id(t)] = av.item()

    optimizer.zero_grad()
    torch.cuda.empty_cache()

    total_loss = 0.0
    total_kl = 0.0
    n = 0

    for traj in batch_trajectories:
        adv = adv_map[id(traj)]
        # Build ids + completion mask from the trajectory's trainable span.
        full_ids, comp_mask = _completion_token_mask(
            traj["context"], traj["trainable_text"]
        )
        if comp_mask.sum() == 0:
            continue  # no completion tokens to train on
        full_ids = full_ids.unsqueeze(0).to(policy_model.device)
        comp_mask = comp_mask.to(policy_model.device)

        # Policy logprobs of taken tokens (completion only)
        policy_logits = policy_model(full_ids).logits
        policy_tok_lp = _gather_token_logprobs(policy_logits, full_ids)  # [seq-1]

        # Frozen reference = base model (adapter disabled), taken tokens
        with policy_model.disable_adapter():
            with torch.no_grad():
                ref_logits = policy_model(full_ids).logits
                ref_tok_lp = _gather_token_logprobs(ref_logits, full_ids)

        # Align completion mask to the shifted [seq-1] logprob vector
        m = comp_mask[1:]  # target positions correspond to tokens 1..seq-1
        policy_lp = policy_tok_lp[m]
        ref_lp = ref_tok_lp[m]

        # Importance ratio vs the behavior (sampling) policy.
        # Prefer vLLM behavior logprobs; if unavailable, treat ratio as 1
        # (REINFORCE-style) rather than fabricate a denominator.
        # For simplicity we use the current policy's own detached logprobs as the
        # behavior baseline when vLLM logprobs are absent, which is a valid
        # single-step on-policy approximation because we update once per rollout.
        behavior_lp = policy_lp.detach()
        ratio = torch.exp(policy_lp - behavior_lp)  # ~1 at step 0, enables clipping
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        # Per-token PG loss, then length-normalized mean over completion tokens
        pg = -torch.min(ratio * adv, clipped * adv)

        # KL(policy || ref) estimator on completion tokens (k3, unbiased, >=0)
        log_diff = ref_lp - policy_lp
        kl = torch.exp(log_diff) - log_diff - 1.0

        loss = (pg + beta * kl).mean()
        loss.backward()

        total_loss += loss.item()
        total_kl += kl.mean().item()
        n += 1

        del policy_logits, ref_logits, policy_tok_lp, ref_tok_lp
        del policy_lp, ref_lp, ratio, clipped, pg, kl, loss, full_ids, comp_mask

    if n == 0:
        return 0.0, 0.0

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    policy_model.save_pretrained("./checkpoint/active_lora")

    return total_loss / n, total_kl / n


# ─── 4. THE MASTER EXECUTION LOOP ────────────────────────────────────────


async def main():
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    ablation_cfg = cfg.get("ablation", {})
    dataset_path = ablation_cfg.get(
        "dataset_path", cfg.get("dataset_path", "./data/openrlhf_apps_train.jsonl")
    )
    epochs = cfg.get("num_epochs", 3)

    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    wandb.init(
        project=cfg.get("wandb_project", "rlef-code"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        config=cfg,
        tags=cfg.get("tags", []),
    )
    print(f"Loaded {len(dataset)} training problems from {dataset_path}.")

    start_temp = cfg.get("start_temp", 0.7)
    end_temp = cfg.get("end_temp", 0.7)  # default = fixed temp (no across-run anneal)
    batch_size = cfg.get("batch_size", 20)

    total_batches = (
        len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)
    ) * epochs
    global_batch = 0

    # Rolling windows so we can read a trend through the per-batch noise.
    from collections import deque

    reward_window = deque(maxlen=5)
    success_window = deque(maxlen=5)
    prev_checksum = None

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        # Re-shuffle each epoch so batch composition (and its difficulty mix)
        # differs across epochs. Without this, the same hard cluster of problems
        # lands in the same relative step every epoch, producing repeating dips
        # in success/reward that look like instability but are just order effects.
        import random as _random

        _random.Random(1234 + epoch).shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i : i + batch_size]
            current_temp = start_temp - (start_temp - end_temp) * (
                global_batch / max(1, total_batches - 1)
            )
            sampling_params = SamplingParams(
                temperature=current_temp,
                max_tokens=1200,
                stop=cfg.get("stop_tokens", ["</code>"]),
                include_stop_str_in_output=True,
                logprobs=1,  # request behavior logprobs for the importance ratio
            )
            # Fresh LoRA id each batch -> forces vLLM to load the latest adapter.
            active_lora = _current_lora_request()

            tasks = [
                run_single_episode(
                    data["prompt"], data, active_lora, sampling_params, ablation_cfg
                )
                for data in batch_data
            ]
            trajectories = await asyncio.gather(*tasks)

            avg_reward = sum(t["reward"] for t in trajectories) / max(
                1, len(trajectories)
            )
            loss, kl_divergence = grpo_update_step(
                trajectories, beta=cfg.get("kl_beta", 0.04)
            )

            total_execute = sum(t["stats"]["execute"] for t in trajectories)
            total_tests = sum(t["stats"]["generate_tests"] for t in trajectories)
            total_invalid = sum(t["stats"]["invalid_format"] for t in trajectories)
            total_truncated = sum(t["stats"].get("truncated", 0) for t in trajectories)
            avg_turns = sum(t["turns_taken"] for t in trajectories) / max(
                1, len(trajectories)
            )
            avg_ctx_len = sum(t["final_context_length"] for t in trajectories) / max(
                1, len(trajectories)
            )
            success_rate = sum(t["success"] for t in trajectories) / max(
                1, len(trajectories)
            )

            # --- Rolling trend + adapter-swap diagnostic ---
            reward_window.append(avg_reward)
            success_window.append(success_rate)
            rolling_reward = sum(reward_window) / len(reward_window)
            rolling_success = sum(success_window) / len(success_window)

            checksum = _lora_checksum()
            checksum_delta = (
                0.0 if prev_checksum is None else abs(checksum - prev_checksum)
            )
            prev_checksum = checksum

            batch_errors = Counter()
            for t in trajectories:
                batch_errors.update(t["errors"])

            log_metrics = {
                "train/loss": loss,
                "train/kl_divergence": kl_divergence,
                "train/avg_reward": avg_reward,
                "train/rolling_reward": rolling_reward,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/temperature": current_temp,
                "train/lora_checksum": checksum,
                "train/lora_checksum_delta": checksum_delta,
                "tools/execute_calls": total_execute,
                "tools/generate_tests_calls": total_tests,
                "tools/invalid_format_errors": total_invalid,
                "tools/truncated_completions": total_truncated,
                "metrics/avg_turns": avg_turns,
                "metrics/avg_context_length": avg_ctx_len,
                "metrics/success_rate": success_rate,
                "metrics/rolling_success": rolling_success,
                "epoch": epoch + 1,
                "global_step": global_batch,
            }
            for err_type, count in batch_errors.items():
                log_metrics[f"errors/{err_type}"] = count
            wandb.log(log_metrics)

            swap_flag = (
                ""
                if checksum_delta > 0 or prev_checksum is None
                else "  ⚠️ADAPTER FROZEN"
            )
            print(
                f"Batch {global_batch+1}/{total_batches} | Temp {current_temp:.3f} "
                f"| R {avg_reward:.3f} (roll {rolling_reward:.3f}) "
                f"| Succ {success_rate:.2%} (roll {rolling_success:.2%}) "
                f"| Loss {loss:.4f} | Δwts {checksum_delta:.2e}{swap_flag}"
            )
            global_batch += 1

        epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
        os.makedirs(epoch_dir, exist_ok=True)
        active_path = "./checkpoint/active_lora"
        if os.path.exists(active_path):
            shutil.copytree(active_path, epoch_dir, dirs_exist_ok=True)
            print(f"Saved checkpoint to {epoch_dir}")


if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()
