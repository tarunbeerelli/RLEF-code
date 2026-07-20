"""
prepare_openrlhf_data.py

Converts the APPS dataset into the flat JSONL format the agent loop consumes.
Builds prompts and (where configured) test-case anchors from the train.yaml ablation
config, applies stratified / curriculum sampling with a provably disjoint unseen/seen
split, and generates a static arm-agnostic evaluation set when one is absent.
"""

import json
from pathlib import Path

import yaml
import random
from transformers import AutoTokenizer

from rlef.data import load_apps_split
from rlef.prompt import format_prompt


def main():
    # 1. Load the central configuration
    print("Reading ablation config from train.yaml...")
    try:
        with open("configs/train.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("CRITICAL: configs/train.yaml not found. Please ensure the file exists.")
        return

    ablation_cfg = cfg.get("ablation", {})
    evaluation_cfg = cfg.get("evaluation", {})
    use_edge_cases = ablation_cfg.get("use_edge_cases", False)

    # Target output paths defined in yaml
    output_path_str = ablation_cfg.get(
        "dataset_path", cfg.get("dataset_path", "data/openrlhf_apps_train.jsonl")
    )
    output_file = Path(output_path_str)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    eval_path_str = evaluation_cfg.get("dataset_path", "data/apps_eval.jsonl")
    eval_file = Path(eval_path_str)
    eval_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw APPS train split...")
    problems = load_apps_split("data/raw/APPS", split="train")

    # Deterministic shuffle for identical splits every time
    random.seed(42)
    random.shuffle(problems)

    # ─── STRATIFIED CAP ACROSS DIFFICULTY ──────────────────────────────────
    # Cap total training problems while preserving the difficulty distribution.
    # `train_cap` and `stratify_mode` come from train.yaml so run_pipeline controls them.
    train_cap = cfg.get("train_cap", ablation_cfg.get("train_cap", 1000))
    stratify_mode = cfg.get("stratify_mode", "proportional")  # or "balanced"
    curriculum_mode = ablation_cfg.get("curriculum_mode", "full")
    manifest_path = Path(
        ablation_cfg.get("manifest_path", "./data/run5_trained_ids.json")
    )

    from collections import defaultdict

    by_diff = defaultdict(list)
    for p in problems:
        by_diff[p.difficulty].append(p)

    diff_order = ["introductory", "interview", "competition"]
    available = {d: len(by_diff.get(d, [])) for d in diff_order}
    total_available = sum(available.values())
    print(
        f"Available verified train problems by difficulty: {available} (total {total_available})"
    )

    # ── CURRICULUM: hard_specialize ─────────────────────────────────────────
    # Phase-2 dataset = ALL unseen hard (interview+competition not in the base
    # run's manifest) + a stratified replay sampled from the SEEN set (mirrors
    # the base run's difficulty mix, for anti-forgetting). Requires the base run
    # to have written a manifest of the problem_ids it trained on.
    if curriculum_mode == "hard_specialize":
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"curriculum_mode=hard_specialize needs a manifest at {manifest_path}, "
                f"written by the base (phase-1) run. Run phase 1 first."
            )
        seen_ids = set(json.loads(manifest_path.read_text()))
        print(f"Loaded manifest: {len(seen_ids)} seen problem_ids from base run.")

        hard = [p for p in problems if p.difficulty in ("interview", "competition")]
        unseen_hard = [p for p in hard if p.problem_id not in seen_ids]
        seen_all = [p for p in problems if p.problem_id in seen_ids]

        replay_frac = float(ablation_cfg.get("replay_frac", 0.15))
        # replay count is a fraction of the UNSEEN-HARD CORE (so the sprinkle scales
        # with the fresh data, keeping ~87% hard / ~87% fresh as sized).
        replay_n = int(round(len(unseen_hard) * replay_frac))
        # stratified replay: mirror the seen set's difficulty mix
        seen_by_diff = defaultdict(list)
        for p in seen_all:
            seen_by_diff[p.difficulty].append(p)
        seen_total = max(1, len(seen_all))
        replay = []
        for d in diff_order:
            pool = seen_by_diff.get(d, [])
            random.shuffle(pool)
            share = int(round(replay_n * (len(pool) / seen_total)))
            replay.extend(pool[:share])

        selected = unseen_hard + replay
        random.shuffle(selected)
        problems = selected
        n_hard = sum(
            1 for p in problems if p.difficulty in ("interview", "competition")
        )
        print(
            f"Curriculum hard_specialize: {len(unseen_hard)} unseen-hard "
            f"+ {len(replay)} stratified seen-replay = {len(problems)} "
            f"({n_hard/max(1,len(problems))*100:.0f}% hard, "
            f"{len(unseen_hard)/max(1,len(problems))*100:.0f}% fresh)."
        )
        # Skip normal stratified sampling below.
        _skip_stratify = True
    else:
        _skip_stratify = False

    if _skip_stratify:
        pass
    elif train_cap >= total_available:
        # Use everything, but still report the distribution.
        selected = []
        for d in diff_order:
            selected.extend(by_diff.get(d, []))
        print(
            f"train_cap ({train_cap}) >= available ({total_available}); using all problems."
        )
    elif stratify_mode == "balanced":
        # Equal count per difficulty, clamped by availability.
        per = train_cap // len(diff_order)
        selected = []
        for d in diff_order:
            pool = by_diff.get(d, [])
            random.shuffle(pool)
            take = pool[:per]
            selected.extend(take)
            print(f"  balanced {d}: took {len(take)} (requested {per})")
    else:
        # Proportional: preserve the natural difficulty ratio.
        selected = []
        for d in diff_order:
            pool = by_diff.get(d, [])
            random.shuffle(pool)
            share = int(round(train_cap * (available[d] / total_available)))
            take = pool[:share]
            selected.extend(take)
            print(
                f"  proportional {d}: took {len(take)} "
                f"({available[d]/total_available:.1%} of cap {train_cap})"
            )

    if not _skip_stratify:
        random.shuffle(selected)  # mix difficulties so batches are heterogeneous
        problems = selected
        print(
            f"Stratified training set locked: {len(problems)} problems "
            f"(mode={stratify_mode}, cap={train_cap})"
        )

    # Write a manifest of the problem_ids this run trains on, so a later
    # curriculum phase can exclude them (true unseen-vs-seen split). Only the
    # base/phase-1 run needs this; controlled by write_manifest in config.
    if ablation_cfg.get("write_manifest"):
        ids = [p.problem_id for p in problems]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(ids))
        print(f"📝 Wrote manifest of {len(ids)} trained problem_ids to {manifest_path}")

    print("Initializing Qwen Tokenizer for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # --- 1. GENERATE TRAINING DATA ---
    # Run B1 (fixed_test_conditioning): instead of the model generating its own tests
    # (the run_6 self-consistency exploit), surface REAL held-out test cases to the
    # model as feedback. To avoid leakage, each problem's cases are split: `shown` are
    # revealed to the model between turns; `graded` are the held-out remainder the
    # reward is computed on. The model is NEVER graded on a case it was shown.
    fixed_tests = ablation_cfg.get("fixed_test_conditioning", False)
    n_shown = int(ablation_cfg.get("n_shown_tests", 5))  # cases revealed as feedback
    min_graded = int(ablation_cfg.get("min_graded_tests", 2))  # skip if too few remain
    import random as _rnd

    _rnd.seed(1234)  # deterministic split

    def _split_cases(inputs, outputs):
        """Return (shown_in, shown_out, graded_in, graded_out) or None if the problem
        can't be split while leaving >= min_graded cases to grade on."""
        n = len(inputs)
        if n < n_shown + min_graded:
            return None  # not enough cases to both show and grade cleanly
        # Show front + back (edge-ish), fill the middle randomly if n_shown > 2.
        idx = list(range(n))
        shown_idx = {idx[0], idx[-1]}
        pool = [i for i in idx if i not in shown_idx]
        _rnd.shuffle(pool)
        while len(shown_idx) < n_shown and pool:
            shown_idx.add(pool.pop())
        graded_idx = [i for i in idx if i not in shown_idx]
        si = sorted(shown_idx)
        return (
            [inputs[i] for i in si],
            [outputs[i] for i in si],
            [inputs[i] for i in graded_idx],
            [outputs[i] for i in graded_idx],
        )

    _skipped_fixed = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for p in problems:
            shown_in = shown_out = None
            if fixed_tests:
                split = (
                    _split_cases(p.inputs, p.outputs)
                    if (p.inputs and p.outputs)
                    else None
                )
                if split is None:
                    _skipped_fixed += 1
                    continue  # drop problems that can't be split leakage-free
                shown_in, shown_out, graded_in, graded_out = split
            else:
                graded_in, graded_out = p.inputs, p.outputs

            # 2. THE ANCHOR INJECTION (For Runs 6 & 7; not used by B1)
            if use_edge_cases and p.inputs and p.outputs and p.fn_name:
                anchor_text = (
                    "\n\n=== GROUND TRUTH ANCHOR ===\n"
                    "Use this exact syntax for your <edge_cases> block:\n"
                    f"assert {p.fn_name}({repr(p.inputs[0])}) == {repr(p.outputs[0])}\n"
                    "===========================\n"
                )
                p.question = p.question.strip() + anchor_text

            # 3. Build the dynamic prompt based on the ablation config
            messages = format_prompt(p, ablation_cfg)

            # 4. Apply Qwen's exact chat template tokens
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 5. Pack the row. `inputs`/`outputs` are the GRADED (held-out) cases the
            # reward uses. For fixed_test_conditioning, `shown_inputs`/`shown_outputs`
            # are the revealed cases train_agent surfaces as feedback — kept in a
            # SEPARATE field so the reward path can never accidentally grade on them.
            row = {
                "problem_id": p.problem_id,
                "difficulty": p.difficulty,
                "prompt": prompt_text,
                "inputs": graded_in,
                "outputs": graded_out,
                "fn_name": p.fn_name,
            }
            if fixed_tests:
                row["shown_inputs"] = shown_in
                row["shown_outputs"] = shown_out
            f.write(json.dumps(row) + "\n")

    if fixed_tests:
        print(
            f"fixed_test_conditioning: showed up to {n_shown} cases/problem as feedback, "
            f"graded on held-out remainder (>= {min_graded}); "
            f"skipped {_skipped_fixed} problems with too few cases to split."
        )

    print(
        f"\nSuccessfully exported {len(problems)} trajectory prompts to {output_file}"
    )
    if use_edge_cases:
        print(
            "Notice: Explicit 'Anchor & Extend' Ground Truth test cases were injected into the prompts."
        )
    else:
        print("Notice: Standard execution prompts generated (No edge case injection).")

    # --- 2. GENERATE STATIC EVAL DATA (IF MISSING OR STALE) ---
    # The eval file is now ARM-AGNOSTIC: it stores the raw problem (incl. `question`)
    # and evaluate.py rebuilds the full prompt per-arm at eval time using that arm's
    # max_turns AND feedback_type. This means one eval file serves every arm
    # correctly, and there is no max_turns/feedback_type to go stale.
    # We still version the schema so an OLD eval file (which baked in a one-shot
    # prompt and lacks `question`) is detected and regenerated.
    EVAL_SCHEMA_VERSION = 2  # v2 = arm-agnostic (stores raw question)
    eval_meta_file = eval_file.with_suffix(".meta")
    stale = False
    if eval_file.exists():
        ok = False
        if eval_meta_file.exists():
            try:
                ok = int(eval_meta_file.read_text().strip()) == EVAL_SCHEMA_VERSION
            except Exception:
                ok = False
        if not ok:
            stale = True  # legacy/one-shot file or wrong schema -> rebuild

    if stale:
        print("\n⚠️ Eval file uses an old schema (pre arm-agnostic); regenerating.")
        eval_file.unlink(missing_ok=True)

    if not eval_file.exists():
        print(
            f"\nGenerating arm-agnostic eval split at {eval_file} (schema v{EVAL_SCHEMA_VERSION})..."
        )
        try:
            eval_problems = load_apps_split("data/raw/APPS", split="test")
            random.seed(42)
            # Sample a KNOWN count per difficulty so eval denominators are real.
            # evaluate.py takes up to 250 per bucket; provide that many when available.
            PER_BUCKET = 250
            from collections import defaultdict

            by_diff = defaultdict(list)
            for p in eval_problems:
                by_diff[p.difficulty].append(p)
            balanced = []
            for diff in ["introductory", "interview", "competition"]:
                pool = by_diff.get(diff, [])
                random.shuffle(pool)
                take = pool[:PER_BUCKET]
                balanced.extend(take)
                print(f"  eval {diff}: {len(take)} problems")
            eval_problems = balanced

            with open(eval_file, "w", encoding="utf-8") as f:
                for p in eval_problems:
                    # Arm-agnostic: store the raw question so evaluate.py can build
                    # the correct system prompt per-arm (right max_turns + feedback_type).
                    # `prompt` retained as a one-shot fallback if question is missing.
                    fallback_messages = format_prompt(p, {"max_turns": 1})
                    fallback_prompt = tokenizer.apply_chat_template(
                        fallback_messages, tokenize=False, add_generation_prompt=True
                    )
                    row = {
                        "problem_id": p.problem_id,
                        "difficulty": p.difficulty,
                        "question": p.question,  # raw text -> per-arm rebuild at eval
                        "prompt": fallback_prompt,  # fallback only
                        "inputs": p.inputs,
                        "outputs": p.outputs,
                        "fn_name": p.fn_name,
                    }
                    f.write(json.dumps(row) + "\n")
            eval_meta_file.write_text(str(EVAL_SCHEMA_VERSION))
            print(
                f"Successfully generated {len(eval_problems)} arm-agnostic eval problems to {eval_file}"
            )
        except Exception as e:
            print(
                f"Warning: Could not generate eval split. Ensure 'test' split exists. Error: {e}"
            )


if __name__ == "__main__":
    main()
