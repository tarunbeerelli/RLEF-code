"""
clean_and_optimize_dataset.py — Diagnostic Cloud Preprocessing Suite (Multi-Solution Checking)
"""

import pathlib
import shutil
import os
from collections import Counter

from dotenv import load_dotenv
from rlef.data import load_apps_split
from rlef.reward import execution_reward

load_dotenv()


def sweep_and_purge_split(data_dir, split_name):
    print("\n==================================================")
    print(f"STARTING MULTI-SOLUTION SWEEP FOR SPLIT: {split_name.upper()}")
    print("==================================================")

    try:
        problems = load_apps_split(data_dir, split=split_name, difficulties=None)
    except Exception as e:
        print(f"Error loading split {split_name}: {e}. Skipping.")
        return

    total_problems = len(problems)
    passed_count = 0
    purged_count = 0

    error_tracker = Counter()

    base_data_path = pathlib.Path(data_dir) / split_name
    broken_base_path = pathlib.Path("data/raw/APPS_broken") / split_name

    for idx, p in enumerate(problems):
        problem_folder = base_data_path / p.problem_id

        if (
            not p.solutions
            or not isinstance(p.solutions, list)
            or len(p.solutions) == 0
        ):
            if problem_folder.exists():
                error_tracker["No Ground Truth Provided"] += 1
                broken_dir = broken_base_path / p.problem_id
                os.makedirs(broken_dir.parent, exist_ok=True)
                shutil.move(str(problem_folder), str(broken_dir))
                purged_count += 1
            continue

        # --- THE FIX: Try up to 5 different human solutions ---
        best_pass_rate = 0.0
        best_error_types = ["Unknown Failure"]

        for sol in p.solutions[:5]:
            result = execution_reward(
                code=sol,
                inputs=p.inputs,
                outputs=p.outputs,
                fn_name=p.fn_name,
                current_turn=1,
                difficulty=p.difficulty,
                ablation_cfg={
                    "use_step_credit": False,
                    "use_lint_bonus": False,
                    "use_multi_turn": False,
                },
            )

            if result.pass_rate > best_pass_rate:
                best_pass_rate = result.pass_rate
                best_error_types = result.error_types

            if best_pass_rate == 1.0:
                break  # We found a perfect solution, stop checking!

        # ------------------------------------------------------

        if best_pass_rate == 1.0:
            passed_count += 1
        else:
            if problem_folder.exists():
                if best_error_types:
                    error_tracker.update(best_error_types)

                broken_dir = broken_base_path / p.problem_id
                os.makedirs(broken_dir.parent, exist_ok=True)
                shutil.move(str(problem_folder), str(broken_dir))
                purged_count += 1

        if (idx + 1) % 250 == 0:
            print(
                f"  Processed {idx + 1}/{total_problems}... Passed: {passed_count} | Archived: {purged_count}"
            )

    print(f"\n=== SPLIT SUMMARY ({split_name.upper()}) ===")
    print(f"  Total Originally Found: {total_problems}")
    print(
        f"  Verified Perfect (Kept): {passed_count} ({passed_count/total_problems:.1%})"
    )
    print(
        f"  Broken/Empty (Archived): {purged_count} ({purged_count/total_problems:.1%})"
    )

    print("\n=== FAILURE AUTOPSY REPORT ===")
    for error_type, count in error_tracker.most_common():
        print(f"  - {error_type}: {count}")


def main():
    target_data_dir = "data/raw/APPS"

    # 1. Restore the previously archived problems so we can re-test them properly
    broken_dir = pathlib.Path("data/raw/APPS_broken/train")
    active_dir = pathlib.Path("data/raw/APPS/train")
    if broken_dir.exists():
        print("Restoring previously archived problems for multi-solution re-testing...")
        for item in broken_dir.iterdir():
            if item.is_dir():
                shutil.move(str(item), str(active_dir / item.name))

    print("Initializing global APPS diagnostic sweep...")
    sweep_and_purge_split(target_data_dir, split_name="train")


if __name__ == "__main__":
    main()
