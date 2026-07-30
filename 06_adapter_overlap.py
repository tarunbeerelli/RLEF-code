#!/usr/bin/env python3
"""
06_adapter_overlap.py  (v2 -- schema-corrected)
===============================================
Per-problem set analysis across adapter arms (attention-only / MLP-only / all-7).

WHY
---
The aggregate table cannot separate two very different worlds:

  (i)  SUBSUMPTION  -- MLP-only solves (nearly) everything attention-only solves,
                       plus more. One faculty, more capacity; the 3x parameter
                       advantage is then the whole explanation.
  (ii) DISSOCIATION -- the arms solve materially DIFFERENT problems. Two faculties,
                       and the curriculum/MLP score parity reflects two routes to
                       the same number.

This builds the paired 2x2 contingency table per arm-pair, runs McNemar's test (the
correct paired test -- more powerful than the unpaired z-test on aggregates because
it conditions on problem identity), and locates the observed overlap between the
independence and subsumption reference points.

Secondary diagnostics, all enabled by the per-turn `errors` field:
  * recovery-by-error-class  -- what kind of failure each arm recovers from
  * degeneration profile     -- InvalidFormat rate by turn (repetition collapse)
  * shortcut / hard-coding scan and code-length distributions

Standard library only. Python 3.8+.

USAGE
    python 06_adapter_overlap.py --results-dir results --inspect
    python 06_adapter_overlap.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUNS = {
    "base-single": "apps_eval_baseline_1turn_zeroshot_BASELINE.json",
    "base-loop": "apps_eval_baseline_3turn_last_failed_BASELINE.json",
    "S1-single": "apps_eval_reeval_run_1_single_shot.json",
    "A1-attn": "apps_eval_run_5_proper_phase1.json",
    "A2-all7": "apps_eval_run_A2_all_layers.json",
    "A3-mlp": "apps_eval_run_A3_mlp_only.json",
    "B1-attn": "apps_eval_run_B1_fixed_tests_attn_only.json",
    "B2-all7": "apps_eval_run_B2_fixed_tests_all_layers.json",
    "B3-mlp": "apps_eval_run_B3_fixed_tests_mlp_only.json",
    "C1-curr": "apps_eval_run_phase2_B1_curriculum.json",
}

PAIRS_OF_INTEREST = [
    ("A3-mlp", "A1-attn"),  # THE key test
    ("A3-mlp", "A2-all7"),  # why does all-7 trail its own MLP subset?
    ("A2-all7", "A1-attn"),  # does all-7 land on the attention solution?
    ("B3-mlp", "B1-attn"),  # same test under unseen real tests
    ("C1-curr", "A3-mlp"),  # parity: same problems or different routes?
    ("C1-curr", "A1-attn"),  # what the curriculum phase added
    ("B2-all7", "B1-attn"),
    ("B3-mlp", "B2-all7"),
    (
        "C1-curr",
        "B1-attn",
    ),  # staging isolated: same adapter, same objective, warm vs cold
    (
        "C1-curr",
        "B3-mlp",
    ),  # attention curriculum vs feed-forward on a matched objective
]

TIERS = ["introductory", "interview", "competition"]

ID_KEYS = ("problem_id", "id", "pid", "task_id", "index")
DIFF_KEYS = ("difficulty", "level", "tier", "bucket")
CODE_KEYS = ("completion", "code", "final_code", "solution", "generation")
HIST_KEYS = ("turn_history", "turns", "history", "trajectory")
# Record-level authoritative strict-success flags (0/1 ints in this harness).
P1_KEYS = ("pass_at_1", "pass@1", "solved_turn1")
PN_KEYS = ("pass_at_N", "pass_at_n", "pass_at_3", "pass_at_k", "pass@k", "solved")
RATE_KEYS = ("pass_rate", "reward", "score", "fraction_passed")
TURN_KEYS = ("turn", "turn_idx", "step", "attempt")
ERR_KEYS = ("errors", "error_types", "failures")

SHORTCUT_PATTERNS = [
    "if input ==",
    "if inp ==",
    "if s ==",
    "if n ==",
    "if sys.argv",
    "print('expected",
    'print("expected',
    "== 'test",
    '== "test',
]


def first_key(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def as_flag(v):
    """Interpret 0/1 int, bool, or 'true' as a strict-success flag. None if absent."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) >= 0.5
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return None


def norm_difficulty(v):
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    for t in TIERS:
        if s.startswith(t[:5]):
            return t
    return s


def turn_solved(t):
    """Strict whole-question success for one turn record (pass_rate == 1.0)."""
    if not isinstance(t, dict):
        return False
    f = as_flag(first_key(t, P1_KEYS + ("passed", "success")))
    if f is not None and not isinstance(first_key(t, RATE_KEYS), (int, float)):
        return f
    r = first_key(t, RATE_KEYS)
    try:
        return r is not None and float(r) >= 1.0 - 1e-9
    except (TypeError, ValueError):
        return False


def dominant_error(errs):
    if not errs:
        return "None"
    if isinstance(errs, str):
        return errs
    counts = {}
    for e in errs:
        counts[str(e)] = counts.get(str(e), 0) + 1
    # InvalidFormat is a protocol failure and takes precedence when present.
    if "InvalidFormat" in counts:
        return "InvalidFormat"
    return max(counts, key=counts.get)


def inspect(path: Path):
    data = json.load(path.open())
    print(f"\n=== {path.name} ===")
    print(f"top-level: {list(data.keys())}")
    s = data.get("summary", {})
    if isinstance(s, dict):
        print(f"summary  : {list(s.keys())}")
        bd = s.get("by_difficulty", {})
        if bd:
            print(f"by_diff  : {json.dumps(bd)[:300]}")
    res = data.get("results") or data.get("problems") or []
    print(f"n results: {len(res)}")
    if res:
        print(f"record   : {list(res[0].keys())}")
        h = first_key(res[0], HIST_KEYS)
        if isinstance(h, list) and h:
            nums = [first_key(t, TURN_KEYS) for t in h if isinstance(t, dict)]
            print(f"turn keys: {list(h[0].keys())}   declared turn numbers: {nums}")


def load_run(path: Path):
    data = json.load(path.open())
    res = data.get("results") or data.get("problems") or []
    out, warn = {}, 0

    # Detect turn-number base (this harness is 1-indexed).
    bases = []
    for r in res[:50]:
        h = first_key(r, HIST_KEYS)
        if isinstance(h, list):
            for t in h:
                n = first_key(t, TURN_KEYS) if isinstance(t, dict) else None
                if isinstance(n, int):
                    bases.append(n)
    turn_base = min(bases) if bases else 0

    for i, r in enumerate(res):
        pid = str(first_key(r, ID_KEYS, default=f"idx{i}"))
        tier = norm_difficulty(first_key(r, DIFF_KEYS))
        code = first_key(r, CODE_KEYS, default="") or ""

        hist = first_key(r, HIST_KEYS)
        ordered = []
        if isinstance(hist, list):

            def key(pair):
                idx, t = pair
                n = first_key(t, TURN_KEYS) if isinstance(t, dict) else None
                return int(n) if isinstance(n, int) else idx

            ordered = [t for _, t in sorted(enumerate(hist), key=key)]

        # Prefer the record-level authoritative flags.
        t0 = as_flag(first_key(r, P1_KEYS))
        lp = as_flag(first_key(r, PN_KEYS))

        inferred_t0 = turn_solved(ordered[0]) if ordered else None
        inferred_lp = any(turn_solved(t) for t in ordered) if ordered else None
        if t0 is None:
            t0 = inferred_t0 if inferred_t0 is not None else False
        if lp is None:
            lp = inferred_lp if inferred_lp is not None else t0
        if inferred_lp is not None and lp != inferred_lp:
            warn += 1

        first_turn = None
        for j, t in enumerate(ordered):
            if turn_solved(t):
                n = first_key(t, TURN_KEYS)
                first_turn = (int(n) - turn_base) if isinstance(n, int) else j
                break

        errs_by_turn = [first_key(t, ERR_KEYS, default=[]) or [] for t in ordered]

        out[pid] = {
            "tier": tier,
            "t0": bool(t0),
            "loop": bool(lp),
            "first_turn": first_turn,
            "code": code,
            "errs": errs_by_turn,
            "n_turns": len(ordered),
        }

    if warn:
        print(f"    note: {warn} records where pass_at_N disagreed with turn_history")
    return out, data.get("summary", {}), turn_base


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return ("n/a", 0.0, 1.0)
    if n <= 25:
        try:
            from math import comb

            tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
            return ("exact binomial", float(min(b, c)), min(1.0, 2 * tail))
        except Exception:
            pass
    stat = (abs(b - c) - 1) ** 2 / n
    return ("chi2 Yates 1df", stat, math.erfc(math.sqrt(stat / 2.0)))


def report_pair(na, nb, ra, rb, metric, tier):
    common = {p for p in (set(ra) & set(rb)) if ra[p]["tier"] == tier}
    if not common:
        print(f"  [{tier}] no shared ids -- check ID_KEYS")
        return
    A = {p for p in common if ra[p][metric]}
    B = {p for p in common if rb[p][metric]}
    a, b, c = len(A & B), len(A - B), len(B - A)
    n = len(common)
    d = n - a - b - c
    pa, pb = len(A) / n, len(B) / n
    exp_i = pa * pb * n
    max_s = min(len(A), len(B))
    span = max_s - exp_i
    pos = (a - exp_i) / span if span > 0 else float("nan")
    jac = a / len(A | B) if (A | B) else 0.0
    lab, stat, p = mcnemar(b, c)

    print(
        f"  [{tier:13}] n={n:4d}  {na}: {len(A):3d} ({100*pa:5.1f}%)   {nb}: {len(B):3d} ({100*pb:5.1f}%)"
    )
    print(
        f"                 both={a:3d}   {na}-only={b:3d}   {nb}-only={c:3d}   neither={d:3d}"
    )
    print(
        f"                 Jaccard={jac:.3f}   overlap obs={a}  indep={exp_i:.1f}  subsume={max_s}"
    )
    if not math.isnan(pos):
        v = (
            "SUBSUMPTION-like"
            if pos >= 0.80
            else "DISSOCIATION-like"
            if pos <= 0.45
            else "intermediate"
        )
        print(f"                 axis position={pos:+.2f}  ({v})")
    print(
        f"                 McNemar {lab}: stat={stat:.3f} p={p:.4f}{'  *SIG*' if p < .05 else ''}"
    )
    print()


def report_turns(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(
        f"FIRST-SOLVE-TURN PROFILE ({tier})   [turns labelled as in the result files]"
    )
    print("=" * 100)
    print(
        f"{'run':12} {'turn1':>7} {'turn2':>7} {'turn3':>7} {'total':>7} {'recovery':>10}"
    )
    for name, (run, _s, _b) in runs.items():
        sel = [v for v in run.values() if v["tier"] == tier]
        if not sel:
            continue
        n = len(sel)
        c = {0: 0, 1: 0, 2: 0}
        for v in sel:
            if v["loop"] and v["first_turn"] is not None:
                c[min(v["first_turn"], 2)] = c.get(min(v["first_turn"], 2), 0) + 1
        tot = sum(c.values())
        failed = n - c[0]
        rec = 100 * (tot - c[0]) / failed if failed else 0.0
        print(f"{name:12} {c[0]:7d} {c[1]:7d} {c[2]:7d} {tot:7d} {rec:9.1f}%")
    print("\nrecovery = of turn-1 failures, fraction solved later in the loop")
    print("(starting-level-controlled measure of self-correction)")


def report_recovery_by_error(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(f"RECOVERY BY TURN-1 ERROR CLASS ({tier})  -- what each arm recovers FROM")
    print("=" * 100)
    classes = ["WrongOutput", "Timeout", "RuntimeError", "InvalidFormat", "None"]
    print(f"{'run':12} " + " ".join(f"{c[:11]:>13}" for c in classes))
    for name, (run, _s, _b) in runs.items():
        sel = [v for v in run.values() if v["tier"] == tier and v["n_turns"] > 1]
        if not sel:
            continue
        num, den = {c: 0 for c in classes}, {c: 0 for c in classes}
        for v in sel:
            if v["t0"] or not v["errs"]:
                continue
            k = dominant_error(v["errs"][0])
            k = k if k in den else "None"
            den[k] += 1
            if v["loop"]:
                num[k] += 1
        cells = []
        for c in classes:
            cells.append(
                f"{100*num[c]/den[c]:6.1f}% {den[c]:4d}" if den[c] else f"{'--':>12}"
            )
        print(f"{name:12} " + " ".join(f"{x:>13}" for x in cells))
    print(
        "\nreads as  recovery% n   over problems whose turn-1 failure was that class."
    )
    print(
        "Concentration of MLP recovery in WrongOutput would indicate output/convention"
    )
    print("repair (a pattern faculty) rather than routing-driven correction.")


def report_degeneration(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(f"DEGENERATION PROFILE ({tier})  -- InvalidFormat rate by turn")
    print("=" * 100)
    print(f"{'run':12} {'turn1':>9} {'turn2':>9} {'turn3':>9}")
    for name, (run, _s, _b) in runs.items():
        sel = [v for v in run.values() if v["tier"] == tier]
        if not sel or max(v["n_turns"] for v in sel) < 2:
            continue
        row = []
        for ti in range(3):
            n = sum(1 for v in sel if v["n_turns"] > ti)
            k = sum(
                1
                for v in sel
                if v["n_turns"] > ti
                and dominant_error(v["errs"][ti]) == "InvalidFormat"
            )
            row.append(f"{100*k/n:8.1f}%" if n else f"{'--':>9}")
        print(f"{name:12} " + " ".join(row))
    print("\nA rising InvalidFormat rate across turns indicates repetition collapse /")
    print("truncation before a parseable code block is emitted.")


def report_shortcuts(runs):
    print("\n" + "=" * 100)
    print("SHORTCUT SCAN + CODE LENGTH")
    print("=" * 100)
    print(f"{'run':12} {'n':>5} {'shortcut%':>10} {'median':>8} {'p90':>7}")
    for name, (run, _s, _b) in runs.items():
        codes = [v["code"] for v in run.values() if v["code"]]
        if not codes:
            print(f"{name:12} {'--':>5}")
            continue
        hits = sum(1 for c in codes if any(p in c.lower() for p in SHORTCUT_PATTERNS))
        lens = sorted(len(c.split()) for c in codes)
        print(
            f"{name:12} {len(codes):5d} {100*hits/len(codes):9.2f}% "
            f"{lens[len(lens)//2]:8d} {lens[int(.9*(len(lens)-1))]:7d}"
        )


def reconcile(name, run, summary):
    bd = (summary or {}).get("by_difficulty", {})
    if not bd:
        return
    msgs = []
    for tier in TIERS:
        if tier not in bd:
            continue
        tot = bd[tier].get("total")
        exp1 = bd[tier].get("pass_at_1")
        got = sum(1 for v in run.values() if v["tier"] == tier and v["t0"])
        cnt = sum(1 for v in run.values() if v["tier"] == tier)
        if exp1 is not None and tot:
            want = round(exp1 * tot)
            flag = "OK" if want == got and cnt == tot else "MISMATCH"
            msgs.append(f"{tier[:5]}:{got}/{want}{'' if flag=='OK' else ' !!'}")
    if msgs:
        print("    reconcile solve@1 vs summary -> " + "  ".join(msgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--metric", default="both", choices=["t0", "loop", "both"])
    a = ap.parse_args()

    rd = Path(a.results_dir)
    if not rd.is_dir():
        sys.exit(f"not a directory: {rd}")
    found = {n: rd / f for n, f in RUNS.items() if (rd / f).exists()}
    print(
        f"found {len(found)}; missing: {[n for n in RUNS if n not in found] or 'none'}"
    )
    if a.inspect:
        for p in found.values():
            inspect(p)
        return

    runs = {}
    for name, path in found.items():
        try:
            run, summ, base = load_run(path)
            runs[name] = (run, summ, base)
            t0 = sum(v["t0"] for v in run.values())
            lp = sum(v["loop"] for v in run.values())
            print(
                f"  {name:12} n={len(run):4d} solve@1={t0:3d} solve@k={lp:3d} turn_base={base}"
            )
            reconcile(name, run, summ)
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}")

    for metric in ["t0", "loop"] if a.metric == "both" else [a.metric]:
        print("\n" + "=" * 100)
        print(
            f"PAIRED OVERLAP -- {'solve@1 (turn 1)' if metric=='t0' else 'solve@3 (within loop)'}"
        )
        print("=" * 100)
        for na, nb in PAIRS_OF_INTEREST:
            if na in runs and nb in runs:
                print(f"\n{na}  vs  {nb}")
                for tier in ("introductory", "interview"):
                    report_pair(na, nb, runs[na][0], runs[nb][0], metric, tier)

    report_turns(runs)
    report_recovery_by_error(runs)
    report_degeneration(runs)
    report_shortcuts(runs)

    print("\n" + "=" * 100)
    print("DECISION GUIDE -- A3-mlp vs A1-attn, introductory, solve@1")
    print("=" * 100)
    print("""
  A1-only near 0, axis >= +0.80        -> SUBSUMPTION. One faculty, more capacity.
                                          Keep the claim capacity-scoped (3x params).
  A1-only substantial, axis <= +0.45   -> DISSOCIATION. Two faculties. Licenses the
                                          strong mechanistic framing, and makes the
                                          curriculum/MLP parity two routes to one score.
  Otherwise                            -> partial. Print the table verbatim; McNemar's p
                                          says whether the asymmetry itself is real.
""")


if __name__ == "__main__":
    main()
