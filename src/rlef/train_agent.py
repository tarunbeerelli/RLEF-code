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

# Reduce CUDA fragmentation on the training side. The OOM traceback showed
# reserved-but-unallocated memory, i.e. fragmentation — this lets the allocator
# grow segments instead of failing on a fragmented pool. Must be set before torch
# is imported, so these imports intentionally follow it (E402 suppressed).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import wandb  # noqa: E402
import yaml  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from rlef.reward import execution_reward, verify_generated_tests  # noqa: E402
from rlef.prompt import parse_output  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

# ─── 1. INFRASTRUCTURE & MEMORY PARTITIONING ──────────────────────────────

with open("configs/train.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

print("Initializing vLLM Engine...")
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    enable_prefix_caching=True,
    enable_lora=True,
    max_lora_rank=cfg.get(
        "lora_rank", 16
    ),  # MUST match LoraConfig rank or vLLM rejects the adapter
    gpu_memory_utilization=cfg.get(
        "gpu_memory_utilization", 0.42
    ),  # 0.42 gives KV room for 16384 window
    max_model_len=cfg.get("max_model_len", 16384),
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
    lora_rank = cfg.get("lora_rank", 16)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=cfg.get("lora_alpha", lora_rank * 2),  # scale alpha with rank
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    policy_model = get_peft_model(base_model, lora_config)

optimizer = torch.optim.AdamW(
    policy_model.parameters(), lr=cfg.get("learning_rate", 5e-6)
)
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
    gen_idx: int = 0,
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
            request_id=f"req_{problem_data['problem_id']}_g{gen_idx}_{turn}_{lora_req.lora_int_id}",
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
        return 0.0, 0.0, "empty", 0

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
    traj_oom_count = 0  # trajectories skipped due to OOM within THIS batch

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

        # Pre-bind so the OOM handler can unconditionally drop them (a name left
        # None just frees nothing). Avoids NameError gymnastics on partial failure.
        policy_logits = ref_logits = policy_tok_lp = ref_tok_lp = None
        policy_lp = ref_lp = ratio = clipped = pg = kl = loss = None

        try:
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

            # Free tensors by dropping references (None, not del) so the names stay
            # bound for the exception path's cleanup below.
            policy_logits = ref_logits = policy_tok_lp = ref_tok_lp = None
            policy_lp = ref_lp = ratio = clipped = pg = kl = loss = None
        except torch.cuda.OutOfMemoryError:
            # One pathologically long trajectory shouldn't kill the run OR waste the
            # gradients already accumulated from earlier trajectories in this batch.
            # Drop ONLY this trajectory's tensors, reclaim memory, and CONTINUE — the
            # already-backward'd grads from prior trajectories stay intact and will
            # be applied by optimizer.step() as long as n > 0.
            seq_len = full_ids.shape[1]
            traj_oom_count += 1
            print(
                f"⚠️ OOM on a single trajectory (seq_len={seq_len}); skipping just "
                f"this trajectory. {n} good grads preserved. "
                f"Cumulative traj OOM skips this batch: {traj_oom_count}"
            )
            # Drop references (names pre-bound, always safe) then reclaim.
            policy_logits = ref_logits = policy_tok_lp = ref_tok_lp = None
            policy_lp = ref_lp = ratio = clipped = pg = kl = loss = None
            torch.cuda.empty_cache()
            # NOTE: do NOT zero_grad — that would discard the good accumulated grads.
            # NOTE: do NOT break — continue to the next trajectory.
            continue
        finally:
            full_ids = comp_mask = None

    if n == 0:
        # Every trajectory was skipped. If any OOM'd, this is an OOM-driven skip;
        # otherwise it's a legitimately empty (all-zero-advantage) batch.
        status = "oom_skip" if traj_oom_count > 0 else "empty"
        return 0.0, 0.0, status, traj_oom_count

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    policy_model.save_pretrained("./checkpoint/active_lora")

    # n > 0: we stepped on the good grads. If some trajectories OOM'd, flag it as
    # a partial step so the caller can track degraded batches distinctly from clean.
    status = "partial_oom" if traj_oom_count > 0 else "ok"
    return total_loss / n, total_kl / n, status, traj_oom_count


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
        project=cfg.get("wandb_project", "rlef-code2"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        name=cfg.get(
            "run_name"
        ),  # descriptive name; None -> W&B random name (manual runs)
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
    kl_window = deque(maxlen=5)  # for KL early-stop
    max_kl_stop = cfg.get(
        "max_kl_stop", None
    )  # halt if rolling KL exceeds this (None = disabled)
    stop_training = False
    oom_skip_count = 0  # optimizer steps skipped due to OOM (no update applied)
    empty_batch_count = 0  # legit all-zero-advantage batches (no update needed)
    partial_oom_count = 0  # steps that DID apply but dropped >=1 OOM trajectory
    traj_oom_total = 0  # cumulative individual trajectories dropped to OOM
    best_kl = float("inf")  # lowest rolling_kl seen -> checkpoint selection
    best_kl_step = 0
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
                max_tokens=cfg.get("max_tokens", 1200),
                stop=cfg.get("stop_tokens", ["</code>"]),
                include_stop_str_in_output=True,
                logprobs=1,  # request behavior logprobs for the importance ratio
            )
            # Fresh LoRA id each batch -> forces vLLM to load the latest adapter.
            active_lora = _current_lora_request()

            # MULTI-GENERATION: sample num_generations rollouts PER problem so GRPO
            # normalizes advantage WITHIN each problem group (its intended design),
            # instead of across unrelated problems. This gives a real gradient on
            # hard problems: if 1 of 8 attempts solves it, that attempt gets positive
            # advantage relative to its 7 siblings. Single-generation could never do this.
            num_generations = cfg.get("num_generations", 1)
            tasks = [
                run_single_episode(
                    data["prompt"],
                    data,
                    active_lora,
                    sampling_params,
                    ablation_cfg,
                    gen_idx=g,
                )
                for data in batch_data
                for g in range(num_generations)
            ]
            trajectories = await asyncio.gather(*tasks)

            avg_reward = sum(t["reward"] for t in trajectories) / max(
                1, len(trajectories)
            )
            loss, kl_divergence, step_status, batch_traj_ooms = grpo_update_step(
                trajectories, beta=cfg.get("kl_beta", 0.1)
            )

            # Step statuses:
            #   ok          -> clean full step
            #   partial_oom -> stepped, but some trajectories OOM-skipped (degraded)
            #   oom_skip    -> whole batch OOM'd, no step applied
            #   empty       -> legit all-zero-advantage batch, no step needed
            traj_oom_total += batch_traj_ooms
            if step_status == "oom_skip":
                oom_skip_count += 1
                print(
                    f"⚠️ Step {global_batch}: ENTIRE batch OOM-skipped. "
                    f"Cumulative full-batch skips: {oom_skip_count}"
                )
            elif step_status == "partial_oom":
                partial_oom_count += 1
            elif step_status == "empty":
                empty_batch_count += 1

            # RED LINE: a couple of dropped trajectories per batch is fine, but if a
            # single batch drops many, the memory config is too tight -> lower bs or
            # concurrency. Surface it loudly so it's actionable mid-run.
            oom_redline = cfg.get("oom_redline_per_batch", 3)
            if batch_traj_ooms >= oom_redline:
                print(
                    f"🚨 Step {global_batch}: {batch_traj_ooms} trajectories OOM'd in ONE "
                    f"batch (>= redline {oom_redline}). Memory config too tight — "
                    f"consider lowering batch_size/num_generations or max_model_len."
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
            # Feed only steps that APPLIED an update (ok or partial_oom) into rolling
            # windows. oom_skip/empty applied nothing and return 0.0, which would
            # pollute rolling_kl (falsely suppressing KL early-stop) and success.
            if step_status in ("ok", "partial_oom"):
                reward_window.append(avg_reward)
                success_window.append(success_rate)
                kl_window.append(kl_divergence)
            rolling_reward = sum(reward_window) / max(1, len(reward_window))
            rolling_success = sum(success_window) / max(1, len(success_window))
            rolling_kl = sum(kl_window) / max(1, len(kl_window))

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
                "train/rolling_kl": rolling_kl,
                "train/step_skipped_oom": 1 if step_status == "oom_skip" else 0,
                "train/step_partial_oom": 1 if step_status == "partial_oom" else 0,
                "train/batch_traj_ooms": batch_traj_ooms,
                "train/cumulative_oom_skips": oom_skip_count,
                "train/cumulative_partial_oom_steps": partial_oom_count,
                "train/cumulative_traj_ooms": traj_oom_total,
                "train/cumulative_empty_batches": empty_batch_count,
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

            # --- BEST-KL CHECKPOINT TRACKING ---
            # Save `last_good` only when rolling_kl reaches a NEW MINIMUM, so it
            # holds the LEAST-drifted policy seen so far — not merely one that was
            # under some ceiling. The old ceiling approach kept overwriting with
            # progressively-more-drifted checkpoints (run_6: KL crept 0.005->0.045
            # without early-stopping, so the archived end checkpoint was the WORST
            # one, and eval measured a reward-hacked policy -> 3.6%).
            ckpt_every = cfg.get("checkpoint_every", 40)
            if (
                global_batch % ckpt_every == 0
                and len(kl_window) == kl_window.maxlen
                and rolling_kl < best_kl
            ):
                best_kl = rolling_kl
                best_kl_step = global_batch
                _good_dir = "./checkpoint/last_good_lora"
                if os.path.exists("./checkpoint/active_lora"):
                    shutil.copytree(
                        "./checkpoint/active_lora", _good_dir, dirs_exist_ok=True
                    )
                    print(
                        f"💾 New best-KL checkpoint at step {global_batch} (rolling_kl {rolling_kl:.4f})"
                    )

            # --- KL EARLY-STOP ---
            if (
                max_kl_stop is not None
                and global_batch > 5
                and len(kl_window) == kl_window.maxlen
                and rolling_kl > max_kl_stop
            ):
                print(
                    f"\n⛔ KL EARLY-STOP: rolling KL {rolling_kl:.3f} exceeded "
                    f"max_kl_stop {max_kl_stop}. Policy diverging — halting."
                )
                stop_training = True
                _stop_dir = f"./checkpoints/epoch_{epoch + 1}"
                os.makedirs(_stop_dir, exist_ok=True)
                _src = (
                    "./checkpoint/last_good_lora"
                    if os.path.exists("./checkpoint/last_good_lora")
                    else "./checkpoint/active_lora"
                )
                if os.path.exists(_src):
                    shutil.copytree(_src, _stop_dir, dirs_exist_ok=True)
                    which = (
                        f"best-KL @ step {best_kl_step}"
                        if "last_good" in _src
                        else "current (no good ckpt existed)"
                    )
                    print(f"Saved early-stop checkpoint to {_stop_dir} from {which}")
                break

        if stop_training:
            break

        # --- NORMAL END-OF-EPOCH CHECKPOINT ---
        # Even without an early-stop, the FINAL policy may have drifted well above
        # its best-KL point (run_6's failure mode). If the current rolling_kl is
        # meaningfully worse than the best seen, archive the BEST-KL checkpoint
        # instead of the drifted end state. "Meaningfully worse" = >2x best KL.
        epoch_dir = f"./checkpoints/epoch_{epoch + 1}"
        os.makedirs(epoch_dir, exist_ok=True)
        drifted = (
            best_kl < float("inf")
            and rolling_kl > max(2.0 * best_kl, best_kl + 0.01)
            and os.path.exists("./checkpoint/last_good_lora")
        )
        _end_src = (
            "./checkpoint/last_good_lora" if drifted else "./checkpoint/active_lora"
        )
        if os.path.exists(_end_src):
            shutil.copytree(_end_src, epoch_dir, dirs_exist_ok=True)
            if drifted:
                print(
                    f"⚠️ End policy drifted (rolling_kl {rolling_kl:.4f} >> best {best_kl:.4f} "
                    f"@ step {best_kl_step}). Archived BEST-KL checkpoint, not the drifted end."
                )
            else:
                print(
                    f"Saved end-of-epoch checkpoint to {epoch_dir} (rolling_kl {rolling_kl:.4f})"
                )


if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()
