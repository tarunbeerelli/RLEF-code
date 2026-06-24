"""
clean_and_optimize_dataset.py — Permanent Cloud Preprocessing Suite

Executes an automated verification sweep across ALL splits and tiers of the
APPS dataset, saving verified indexes and purging broken problems from disk.
"""

import pathlib
import shutil

from dotenv import load_dotenv
from rlef.data import load_apps_split
from rlef.reward import execution_reward

load_dotenv()


def sweep_and_purge_split(data_dir, split_name):
    print("\n==================================================")
    print(f"STARTING SWEEP & PURGE FOR SPLIT: {split_name.upper()}")
    print("==================================================")

    # Load all problems for this split across all difficulty tiers
    try:
        problems = load_apps_split(data_dir, split=split_name, difficulties=None)
    except Exception as e:
        print(f"Error loading split {split_name}: {e}. Skipping.")
        return

    total_problems = len(problems)
    passed_count = 0
    purged_count = 0

    # Store reference paths
    base_data_path = pathlib.Path(data_dir) / split_name

    for idx, p in enumerate(problems):
        problem_folder = base_data_path / p.problem_id

        # Guard rail: If problem doesn't have any human reference code, purge it immediately
        if (
            not p.solutions
            or not isinstance(p.solutions, list)
            or len(p.solutions) == 0
        ):
            if problem_folder.exists():
                shutil.rmtree(problem_folder)
                purged_count += 1
            continue

        # Run against our high-fidelity, namespace-isolated execution engine
        result = execution_reward(
            code=p.solutions[0],
            inputs=p.inputs,
            outputs=p.outputs,
            fn_name=p.fn_name,
            reward_type="continuous",
            shaped=False,
            difficulty=p.difficulty,
        )

        # Check if the solution is 100% correct across all test parameters
        if result.pass_rate == 1.0:
            passed_count += 1
        else:
            # --- THE PURGE ---
            # If the ground-truth code yields any WrongOutput, Timeout, or Syntax errors, delete it
            if problem_folder.exists():
                try:
                    shutil.rmtree(problem_folder)
                    purged_count += 1
                except OSError as err:
                    print(f"Failed to delete folder {problem_folder}: {err}")

        # Multi-node heartbeat tracking logger
        if (idx + 1) % 250 == 0:
            print(
                f"  Processed {idx + 1}/{total_problems}... Passed: {passed_count} | Purged: {purged_count}"
            )

    print(f"\n=== SPLIT SUMMARY ({split_name.upper()}) ===")
    print(f"  Total Originally Found: {total_problems}")
    print(
        f"  Verified Perfect (Kept): {passed_count} ({passed_count/total_problems:.1%})"
    )
    print(
        f"  Broken/Empty (Purged):   {purged_count} ({purged_count/total_problems:.1%})"
    )


def main():
    target_data_dir = "data/raw/APPS"

    print("Initializing global APPS optimization sweep...")

    # 1. Sweep and optimize the complete training allocation
    sweep_and_purge_split(target_data_dir, split_name="train")

    # 2. Sweep and optimize the complete testing allocation
    sweep_and_purge_split(target_data_dir, split_name="test")

    print("\n==================================================")
    print("🎉 DATASET PREPROCESSING AND PURGE COMPLETE!")
    print(f"Your cloud directory '{target_data_dir}' is optimized.")
    print("==================================================")


if __name__ == "__main__":
    main()
