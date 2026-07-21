"""
evaluate.py — Asynchronous rollout evaluator.

Measures pass@1 and pass@N under a matched multi-turn harness whose feedback and
execution physics mirror training exactly, so a policy is evaluated on the same
distribution it was trained on. Logs macro suites to W&B and exports granular
per-problem JSON locally.

- execution_reward is called with fn_name, so call-based problems are graded via the
  function-call harness and stdin/stdout problems via the script harness.
- pass@1 is measured strictly at turn 0; pass@N spans the full turn window.
- Decoding is greedy (temperature 0) for a deterministic pass@1. For sampled pass@k,
  set eval temperature > 0 with n > 1 and aggregate.
"""

import argparse
import asyncio
import json
import os
import random
import yaml
import wandb
from collections import Counter
from rlef.reward import execution_reward
from rlef.prompt import parse_output, format_prompt
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest


def build_eval_prompt(problem_data, ablation_cfg, tokenizer):
    """
    Rebuild the eval prompt per-arm from the raw question so the system prompt
    matches THIS arm's max_turns AND feedback_type. Mirrors prepare_openrlhf_data
    exactly, including the edge-case ground-truth anchor injection, so train and
    eval prompts are identical in structure.
    Falls back to the stored one-shot `prompt` if raw question is unavailable
    (legacy eval files).
    """
    question = problem_data.get("question")
    if question is None:
        return problem_data["prompt"]  # legacy fallback

    # Reconstruct a minimal APPSProblem-like object for format_prompt.
    class _P:
        pass

    p = _P()
    p.question = question
    p.problem_id = problem_data.get("problem_id")
    p.difficulty = problem_data.get("difficulty")
    p.inputs = problem_data.get("inputs", [])
    p.outputs = problem_data.get("outputs", [])
    p.fn_name = problem_data.get("fn_name")

    # Edge-case anchor injection — MUST match prepare_openrlhf_data.
    if ablation_cfg.get("use_edge_cases") and p.inputs and p.outputs and p.fn_name:
        anchor_text = (
            "\n\n=== GROUND TRUTH ANCHOR ===\n"
            "Use this exact syntax for your <edge_cases> block:\n"
            f"assert {p.fn_name}({repr(p.inputs[0])}) == {repr(p.outputs[0])}\n"
            "===========================\n"
        )
        p.question = p.question.strip() + anchor_text

    messages = format_prompt(p, ablation_cfg)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


async def evaluate_single_episode(
    prompt_text, problem_data, vllm_engine, lora_req, sampling_params, ablation_cfg
):
    current_context = prompt_text
    max_turns = ablation_cfg.get("max_turns", 5)
    difficulty = problem_data.get("difficulty", "unknown")
    feedback_type = ablation_cfg.get("feedback_type", "last_failed")

    inputs = problem_data.get("inputs", [])
    outputs = problem_data.get("outputs", [])
    fn_name = problem_data.get("fn_name", None)
    # B1 fixed-test-conditioning: shown_* are revealed as feedback, disjoint from the
    # graded inputs/outputs the reward uses. Mirrors train_agent — read shown_* ONLY.
    shown_inputs = problem_data.get("shown_inputs", [])
    shown_outputs = problem_data.get("shown_outputs", [])
    # The eval file is ARM-AGNOSTIC (stores raw inputs/outputs), so for the real_tests
    # arm the shown/graded split is applied HERE, per-problem, with the SAME logic and
    # seed as data prep — the model sees shown cases as feedback and is graded ONLY on
    # the held-out remainder, exactly as in training. Other arms are unaffected.
    if ablation_cfg.get("feedback_type") == "real_tests" and not shown_inputs:
        _ns = int(ablation_cfg.get("n_shown_tests", 5))
        _mg = int(ablation_cfg.get("min_graded_tests", 2))
        if inputs and outputs and len(inputs) >= _ns + _mg:
            import random as _r

            _r.seed(1234)  # SAME seed as prepare_openrlhf_data._split_cases
            _n = len(inputs)
            _idx = list(range(_n))
            _shown = {_idx[0], _idx[-1]}
            _pool = [i for i in _idx if i not in _shown]
            _r.shuffle(_pool)
            while len(_shown) < _ns and _pool:
                _shown.add(_pool.pop())
            _si = sorted(_shown)
            _gi = [i for i in _idx if i not in _shown]
            shown_inputs = [inputs[i] for i in _si]
            shown_outputs = [outputs[i] for i in _si]
            inputs = [inputs[i] for i in _gi]  # grade on held-out only
            outputs = [outputs[i] for i in _gi]

    pass_at_1 = 0
    final_pass_rate = 0.0
    episode_errors = []
    turn_history = []
    final_completion = ""
    turn = 0

    for turn in range(max_turns):
        request_generator = vllm_engine.generate(
            prompt=current_context,
            sampling_params=sampling_params,
            request_id=f"eval_{problem_data['problem_id']}_{turn}",
            lora_request=lora_req,
        )
        final_result = None
        async for res in request_generator:
            final_result = res

        completion = final_result.outputs[0].text
        final_completion = completion
        parsed = parse_output(completion)

        if not parsed["is_valid"]:
            feedback_str = "CRITICAL ERROR: Invalid format. Use <code>...</code>."
            current_context += (
                f"{completion}<|im_end|>\n<|im_start|>user\n"
                f"System Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
            )
            turn_history.append(
                {"turn": turn + 1, "pass_rate": 0.0, "errors": ["InvalidFormat"]}
            )
            episode_errors.append("InvalidFormat")
            continue

        exec_result = await asyncio.to_thread(
            execution_reward,
            code=parsed["code"],
            inputs=inputs,
            outputs=outputs,
            fn_name=fn_name,  # routes call-based problems to the function-call harness
        )

        final_pass_rate = exec_result.pass_rate
        turn_errors = getattr(exec_result, "error_types", [])
        episode_errors.extend(turn_errors)
        turn_history.append(
            {"turn": turn + 1, "pass_rate": final_pass_rate, "errors": turn_errors}
        )

        if turn == 0 and final_pass_rate == 1.0:
            pass_at_1 = 1
        if final_pass_rate == 1.0:
            break

        # The feedback string mirrors train_agent exactly, so every arm is evaluated
        # on the same feedback distribution it was trained under (in particular
        # consolidated, which summarizes error types rather than a single case).
        if feedback_type == "none":
            feedback_str = "Execution failed. Try a different algorithmic approach."
        elif feedback_type == "consolidated":
            err_counts = Counter(turn_errors)
            err_str = ", ".join(f"{v} {k}" for k, v in err_counts.items())
            feedback_str = (
                f"Execution Pass Rate: {final_pass_rate*100:.1f}%. Errors: {err_str}"
            )
        elif feedback_type == "last_failed":
            feedback_str = f"Execution Pass Rate: {final_pass_rate*100:.1f}%\n"
            if len(inputs) > 0 and final_pass_rate < 1.0:
                feedback_str += (
                    f"Failed on test case:\n- Input: {inputs[0]}\n"
                    f"- Expected Output: {outputs[0]}"
                )
        elif feedback_type == "real_tests":
            # Mirrors train_agent exactly: execute against the SHOWN cases in a single
            # call and report the executed pass rate plus the shown spec. Reads
            # shown_* ONLY (never the graded inputs/outputs), so eval feedback matches
            # training and never leaks a graded case.
            feedback_str = f"Execution Pass Rate: {final_pass_rate*100:.1f}%\n"
            if shown_inputs and final_pass_rate < 1.0:
                n_show = min(3, len(shown_inputs))
                shown_res = await asyncio.to_thread(
                    execution_reward,
                    code=parsed["code"],
                    inputs=shown_inputs[:n_show],
                    outputs=shown_outputs[:n_show],
                    fn_name=fn_name,
                )
                feedback_str += (
                    f"You pass {shown_res.pass_rate*100:.0f}% of the provided real "
                    f"test cases. Revise your code to satisfy all of them:\n"
                )
                for j in range(n_show):
                    feedback_str += (
                        f"- Input: {shown_inputs[j]}\n"
                        f"  Expected Output: {shown_outputs[j]}\n"
                    )
        else:
            feedback_str = (
                f"Execution Pass Rate: {final_pass_rate*100:.1f}%. Please revise."
            )
        current_context += (
            f"{completion}<|im_end|>\n<|im_start|>user\n"
            f"System Result:\n{feedback_str}<|im_end|>\n<|im_start|>assistant\n"
        )

    return {
        "problem_id": problem_data.get("problem_id", "unknown"),
        "difficulty": difficulty,
        "pass_at_1": pass_at_1,
        "pass_at_N": 1 if final_pass_rate == 1.0 else 0,
        "final_pass_rate": final_pass_rate,
        "turns_taken": turn + 1,
        "errors": episode_errors,
        "completion": final_completion,
        "turn_history": turn_history,
    }


async def main(baseline: bool = False):
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    ablation_cfg = cfg.get("ablation", {})
    max_turns = ablation_cfg.get("max_turns", 5)

    dataset_path = eval_cfg.get("dataset_path", "./data/apps_eval.jsonl")
    lora_path = eval_cfg.get("lora_path", "./checkpoint/active_lora")
    output_filename = eval_cfg.get("output_filename", "apps_eval_output.json")
    if baseline:
        output_filename = output_filename.replace(".json", "") + "_BASELINE.json"

    raw_dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            raw_dataset.append(json.loads(line))

    random.seed(42)
    dataset = []
    for diff in ["introductory", "interview", "competition"]:
        diff_problems = [d for d in raw_dataset if d.get("difficulty") == diff]
        random.shuffle(diff_problems)
        dataset.extend(diff_problems[:250])

    print(f"Loaded {len(dataset)} eval problems from {dataset_path}")
    if baseline:
        print("MODE: BASELINE — evaluating BASE model (no LoRA adapter).")
    else:
        print(f"Targeting LoRA checkpoint: {lora_path}")

    eval_tags = cfg.get("tags", []).copy()
    if "eval" not in eval_tags:
        eval_tags.append("eval")
    if baseline and "baseline" not in eval_tags:
        eval_tags.append("baseline")

    base_name = cfg.get("run_name")
    if base_name:
        eval_run_name = f"{base_name}_eval" + ("_baseline" if baseline else "")
    else:
        eval_run_name = None
    wandb.init(
        project=cfg.get("wandb_project", "rlef-code2"),
        entity=cfg.get("wandb_entity", "tarunbeerelli-northeastern-university"),
        name=eval_run_name,
        config=cfg,
        tags=eval_tags,
        job_type="evaluation",
    )

    print("Initializing vLLM Engine...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=cfg.get("lora_rank", 16),  # MUST match the trained adapter's rank
        gpu_memory_utilization=cfg.get("eval_gpu_memory_utilization", 0.60),
        max_model_len=cfg.get(
            "max_model_len", 16384
        ),  # MUST match train, or long multi-turn prompts truncate
    )
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Tokenizer for per-arm eval-prompt rebuild (system prompt must match this
    # arm's max_turns + feedback_type; see build_eval_prompt).
    eval_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy -> deterministic pass@1
        max_tokens=cfg.get("max_tokens", 1200),
        stop=cfg.get("stop_tokens", ["</code>"]),
        include_stop_str_in_output=True,
    )

    if baseline:
        active_lora = None  # evaluate the raw base model on the identical harness
    else:
        # Fail loud if a checkpoint was requested but is absent: silently falling back
        # to the base model would mislabel a base-model score as this run's result.
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(
                f"Eval checkpoint not found: {lora_path}. Refusing to fall back to the "
                f"base model, which would mislabel a base score as '{cfg.get('run_name')}'. "
                f"Verify the checkpoint path exists."
            )
        try:
            active_lora = LoRARequest("eval_policy", 1, lora_path=lora_path)
        except Exception as e:
            # The directory exists but the adapter failed to load — surface it rather
            # than quietly evaluating the base model.
            raise RuntimeError(
                f"Failed to load LoRA adapter at {lora_path}: {e}"
            ) from e

    semaphore = asyncio.Semaphore(50)

    async def bounded_eval(data):
        async with semaphore:
            eval_prompt = build_eval_prompt(data, ablation_cfg, eval_tokenizer)
            return await evaluate_single_episode(
                eval_prompt,
                data,
                vllm_engine,
                active_lora,
                sampling_params,
                ablation_cfg,
            )

    print("\n=== Commencing Evaluation Sweep ===")
    tasks = [bounded_eval(data) for data in dataset]
    results = await asyncio.gather(*tasks)

    total = len(results)
    metrics = {"eval/total_problems": total}

    sweep_errors = Counter()
    for r in results:
        sweep_errors.update(r["errors"])
    for err_type, count in sweep_errors.items():
        metrics[f"eval_errors/{err_type}"] = count

    metrics["eval/pass_at_1_overall"] = sum(r["pass_at_1"] for r in results) / total
    metrics[f"eval/pass_at_{max_turns}_overall"] = (
        sum(r["pass_at_N"] for r in results) / total
    )

    by_diff_stats = {}
    for diff in ["introductory", "interview", "competition"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            d_total = len(diff_results)
            pass_1 = sum(r["pass_at_1"] for r in diff_results) / d_total
            pass_n = sum(r["pass_at_N"] for r in diff_results) / d_total
            metrics[f"eval/pass_at_1_{diff}"] = pass_1
            metrics[f"eval/pass_at_{max_turns}_{diff}"] = pass_n
            by_diff_stats[diff] = {
                "pass_at_1": pass_1,
                "pass_at_N": pass_n,
                "total": d_total,
            }

    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")

    # In baseline mode, remap eval/* -> eval_baseline/* so base and trained numbers
    # are distinct series in W&B and can be overlaid for the base-vs-trained delta.
    if baseline:
        metrics = {
            (k.replace("eval/", "eval_baseline/") if k.startswith("eval/") else k): v
            for k, v in metrics.items()
        }
    wandb.log(metrics)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", output_filename)
    final_json = {
        "summary": {
            "pass_at_1": metrics.get("eval/pass_at_1_overall", 0.0),
            "by_difficulty": by_diff_stats,
        },
        "results": [
            {
                "problem_id": r["problem_id"],
                "difficulty": r["difficulty"],
                "pass_rate": r["final_pass_rate"],
                "pass_at_1": r["pass_at_1"],
                "pass_at_N": r["pass_at_N"],
                "completion": r["completion"],
                "error_types": r["errors"],
                "turn_history": r["turn_history"],
            }
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\n💾 Granular JSON saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the BASE model with no LoRA on the identical harness. "
        "Gives the denominator for every trained-run comparison.",
    )
    args = parser.parse_args()
    asyncio.run(main(baseline=args.baseline))
    wandb.finish()
