"""
clean_and_optimize_dataset.py — Diagnostic Dataset Validation Suite (PARALLEL)

Validates each APPS problem by checking whether ANY of its reference solutions
passes the execution harness. Problems whose references all fail are archived so
they don't inject false-negative reward into training.

PARALLELISM DESIGN (why it's safe)
----------------------------------
Previous version interleaved validation (CPU-bound) with archiving (filesystem
moves). That's slow AND unsafe to parallelize. This version splits phases:
  PHASE 1 (parallel, READ-ONLY): worker processes validate shards, return verdicts.
  PHASE 2 (serial, main only, --apply): apply archive moves for failures.
Phase 1 never mutates the filesystem -> race-free parallelism, side-effect-free dry-run.
"""

import argparse
import os
import pathlib
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from rlef.data import load_apps_split
from rlef.reward import execution_reward

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _best_reference_pass_rate(p, max_solutions: int = 1):
    """Try both call-based and stdin routings; keep if any reference passes."""
    best_rate = 0.0
    best_errs = ["Unknown Failure"]
    for sol in p.solutions[:max_solutions]:
        if p.fn_name:
            r_call = execution_reward(sol, p.inputs, p.outputs, fn_name=p.fn_name)
            if r_call.pass_rate > best_rate:
                best_rate, best_errs = r_call.pass_rate, r_call.error_types
            if best_rate == 1.0:
                return best_rate, best_errs
        r_std = execution_reward(sol, p.inputs, p.outputs, fn_name=None)
        if r_std.pass_rate > best_rate:
            best_rate, best_errs = r_std.pass_rate, r_std.error_types
        if best_rate == 1.0:
            return best_rate, best_errs
    return best_rate, best_errs


def _validate_one(p):
    """Worker task: validate one problem, return a picklable verdict dict.
    NEVER touches the filesystem (that's Phase 2)."""
    if not p.solutions or not isinstance(p.solutions, list) or len(p.solutions) == 0:
        return {
            "problem_id": p.problem_id,
            "difficulty": p.difficulty,
            "keep": False,
            "errors": ["No Ground Truth Provided"],
        }
    best_rate, best_errs = _best_reference_pass_rate(p)
    return {
        "problem_id": p.problem_id,
        "difficulty": p.difficulty,
        "keep": best_rate == 1.0,
        "errors": [] if best_rate == 1.0 else list(best_errs),
    }


def sweep_and_purge_split(data_dir, split_name, dry_run=True, workers=None):
    print("\n==================================================")
    print(
        f"SWEEP FOR SPLIT: {split_name.upper()}  (dry_run={dry_run}, workers={workers})"
    )
    print("==================================================")

    try:
        problems = load_apps_split(data_dir, split=split_name, difficulties=None)
    except Exception as e:
        print(f"Error loading split {split_name}: {e}. Skipping.")
        return

    total_problems = len(problems)
    if total_problems == 0:
        print("No problems found.")
        return

    base_data_path = pathlib.Path(data_dir) / split_name
    broken_base_path = pathlib.Path("data/raw/APPS_broken") / split_name

    # ── PHASE 1: parallel, read-only validation ──
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    verdicts = []
    diff_totals = Counter()
    diff_kept = Counter()
    error_tracker = Counter()

    print(f"Validating {total_problems} problems across {workers} process(es)...")
    if workers == 1:
        for idx, p in enumerate(problems):
            verdicts.append(_validate_one(p))
            if (idx + 1) % 250 == 0:
                print(f"  Validated {idx + 1}/{total_problems}...")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_validate_one, p) for p in problems]
            done = 0
            for fut in as_completed(futures):
                verdicts.append(fut.result())
                done += 1
                if done % 250 == 0:
                    print(f"  Validated {done}/{total_problems}...")

    for v in verdicts:
        diff_totals[v["difficulty"]] += 1
        if v["keep"]:
            diff_kept[v["difficulty"]] += 1
        else:
            error_tracker.update(v["errors"])

    passed_count = sum(1 for v in verdicts if v["keep"])
    purged_count = total_problems - passed_count

    # ── PHASE 2: serial archiving (only under --apply) ──
    if not dry_run:
        moved = 0
        for v in verdicts:
            if v["keep"]:
                continue
            problem_folder = base_data_path / v["problem_id"]
            if not problem_folder.exists():
                continue
            broken_dir = broken_base_path / v["problem_id"]
            os.makedirs(broken_dir.parent, exist_ok=True)
            shutil.move(str(problem_folder), str(broken_dir))
            moved += 1
        print(f"\nArchived {moved} broken problems to {broken_base_path}")

    # ── Report ──
    print(f"\n=== SPLIT SUMMARY ({split_name.upper()}) ===")
    print(f"  Total found:            {total_problems}")
    print(
        f"  Verified (kept):        {passed_count} ({passed_count/total_problems:.1%})"
    )
    print(
        f"  {'Would archive' if dry_run else 'Archived'} (broken):  "
        f"{purged_count} ({purged_count/total_problems:.1%})"
    )

    print("\n=== PER-DIFFICULTY RETENTION ===")
    for diff in ["introductory", "interview", "competition"]:
        tot = diff_totals.get(diff, 0)
        kept = diff_kept.get(diff, 0)
        if tot:
            print(f"  {diff:14s}: kept {kept}/{tot} ({kept/tot:.1%})")

    print("\n=== FAILURE AUTOPSY ===")
    for error_type, count in error_tracker.most_common():
        print(f"  - {error_type}: {count}")

    if dry_run:
        print(
            "\n⚠️  DRY RUN — nothing was moved. If this looks sane, re-run with --apply."
        )


def restore_archived(splits):
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
        help="Actually move broken problems. Without this, DRY RUN.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Which splits to sweep (e.g. train test).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Validation worker processes. Default: (CPU count - 1).",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Skip restoring previously archived problems first.",
    )
    args = parser.parse_args()

    dry_run = not args.apply
    if not args.no_restore:
        restore_archived(args.splits)

    print("Initializing APPS diagnostic sweep...")
    for split in args.splits:
        sweep_and_purge_split(
            "data/raw/APPS", split_name=split, dry_run=dry_run, workers=args.workers
        )


if __name__ == "__main__":
    main()
