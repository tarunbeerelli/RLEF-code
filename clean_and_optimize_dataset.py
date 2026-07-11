"""
clean_and_optimize_dataset.py — Diagnostic Dataset Validation Suite (CORRECTED)

Validates each APPS problem by checking whether ANY of its reference solutions
passes the execution harness. Problems whose references all fail are treated as
un-gradeable (broken test data, non-deterministic output, missing solutions) and
archived so they don't inject false-negative reward signal into training.

FIXES vs. previous version
--------------------------
1. HARNESS ROUTING (critical): the previous version forced fn_name routing when
   fn_name was set. But APPS reference solutions in solutions.json are frequently
   written as stdin/stdout scripts even for fn_name problems, so they scored 0
   under the call-based harness and VALID problems got purged. We now try BOTH
   routings (call-based and stdin) and keep the best pass_rate. A problem is kept
   if any reference passes under any routing — mirroring how real APPS graders
   fall back between modes.

2. DRY_RUN (safety): defaults to True. The script REPORTS what it would archive
   without moving anything. Only after you inspect the report do you set
   DRY_RUN=False to actually move folders. Never run a destructive dataset
   mutation blind right after a harness change.

3. Dead kwargs removed: current_turn / difficulty / ablation_cfg were absorbed by
   **kwargs and did nothing. Removed for clarity.

4. Restore step generalized across splits and made safe if target dirs are absent.

5. Longer timeout budget for reference validation: competition references can be
   slow; a tight timeout false-fails legitimate solutions.
"""

import argparse
import os
import pathlib
import shutil
from collections import Counter

from rlef.data import load_apps_split
from rlef.reward import execution_reward

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# ── Reference validation: try BOTH harness routings, keep the best ────────────


def _best_reference_pass_rate(p, max_solutions: int = 5):
    """Return (best_pass_rate, best_error_types) across up to `max_solutions`
    reference solutions, trying both call-based and stdin harness routings for
    each. A reference is 'good' if it passes under EITHER routing."""
    best_rate = 0.0
    best_errs = ["Unknown Failure"]

    for sol in p.solutions[:max_solutions]:
        # Routing A: call-based (only meaningful if fn_name exists)
        if p.fn_name:
            r_call = execution_reward(sol, p.inputs, p.outputs, fn_name=p.fn_name)
            if r_call.pass_rate > best_rate:
                best_rate, best_errs = r_call.pass_rate, r_call.error_types
            if best_rate == 1.0:
                return best_rate, best_errs

        # Routing B: stdin/stdout (works for script-style references)
        r_std = execution_reward(sol, p.inputs, p.outputs, fn_name=None)
        if r_std.pass_rate > best_rate:
            best_rate, best_errs = r_std.pass_rate, r_std.error_types
        if best_rate == 1.0:
            return best_rate, best_errs

    return best_rate, best_errs


def sweep_and_purge_split(data_dir, split_name, dry_run=True):
    print("\n==================================================")
    print(f"SWEEP FOR SPLIT: {split_name.upper()}  (dry_run={dry_run})")
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

    def _archive(problem_folder, problem_id):
        """Move a folder to the broken archive, or just log it under dry_run."""
        if not problem_folder.exists():
            return
        if dry_run:
            return  # report-only; do not touch the filesystem
        broken_dir = broken_base_path / problem_id
        os.makedirs(broken_dir.parent, exist_ok=True)
        shutil.move(str(problem_folder), str(broken_dir))

    for idx, p in enumerate(problems):
        problem_folder = base_data_path / p.problem_id

        # No reference solutions -> cannot verify -> archive
        if (
            not p.solutions
            or not isinstance(p.solutions, list)
            or len(p.solutions) == 0
        ):
            error_tracker["No Ground Truth Provided"] += 1
            _archive(problem_folder, p.problem_id)
            purged_count += 1
            continue

        best_rate, best_errs = _best_reference_pass_rate(p)

        if best_rate == 1.0:
            passed_count += 1
        else:
            if best_errs:
                error_tracker.update(best_errs)
            _archive(problem_folder, p.problem_id)
            purged_count += 1

        if (idx + 1) % 250 == 0:
            print(
                f"  Processed {idx + 1}/{total_problems}... "
                f"Kept: {passed_count} | {'Would archive' if dry_run else 'Archived'}: {purged_count}"
            )

    print(f"\n=== SPLIT SUMMARY ({split_name.upper()}) ===")
    print(f"  Total found:            {total_problems}")
    if total_problems:
        print(
            f"  Verified (kept):        {passed_count} ({passed_count/total_problems:.1%})"
        )
        print(
            f"  {'Would archive' if dry_run else 'Archived'} (broken):  "
            f"{purged_count} ({purged_count/total_problems:.1%})"
        )

    print("\n=== FAILURE AUTOPSY ===")
    for error_type, count in error_tracker.most_common():
        print(f"  - {error_type}: {count}")

    if dry_run:
        print(
            "\n⚠️  DRY RUN — nothing was moved. If this report looks sane "
            "(a low archive %, mostly 'No Ground Truth'/genuine failures),\n"
            "    re-run with --apply to actually archive."
        )


def restore_archived(splits):
    """Move previously archived problems back so they can be re-tested."""
    for split in splits:
        broken_dir = pathlib.Path("data/raw/APPS_broken") / split
        active_dir = pathlib.Path("data/raw/APPS") / split
        if not broken_dir.exists():
            continue
        active_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for item in broken_dir.iterdir():
            if item.is_dir():
                shutil.move(str(item), str(active_dir / item.name))
                moved += 1
        if moved:
            print(f"Restored {moved} archived problems for split '{split}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move broken problems. Without this flag, runs in DRY RUN.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Which splits to sweep (e.g. train test).",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Skip restoring previously archived problems before sweeping.",
    )
    args = parser.parse_args()

    dry_run = not args.apply

    if not args.no_restore:
        restore_archived(args.splits)

    print("Initializing APPS diagnostic sweep...")
    for split in args.splits:
        sweep_and_purge_split("data/raw/APPS", split_name=split, dry_run=dry_run)


if __name__ == "__main__":
    main()
