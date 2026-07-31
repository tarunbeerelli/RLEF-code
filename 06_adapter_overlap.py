#!/usr/bin/env python3
"""
06_adapter_overlap.py  (v3)
===========================
Per-problem analysis of which parameter subspace hosts *error-conditioned repair*.

TERMS
-----
"Error-conditioned repair" is the ability to convert a localised, verified failure
into a passing attempt. The environment supplies error detection and localisation
(the harness states which case failed, with expected against actual output), so
only the correction component is measured. In the standard taxonomy this is
*extrinsic* self-correction -- it uses external feedback -- and is distinct from
intrinsic self-correction, which relies on prompting alone and is widely reported
to fail.

What the ablation can and cannot establish: adapting a subspace and observing the
behaviour appear tells us where the behaviour is *learnable*, not where it is
*represented*. Attention outputs feed the whole residual stream. The claim
supported here is a learnability locus.

WHY PAIRED
----------
Aggregate deltas of a few points on n = 250 cannot separate three different worlds:

  (i)   SUBSUMPTION  -- one arm solves everything the other does, plus more.
                        One mechanism, more capacity.
  (ii)  DISSOCIATION -- the arms solve materially different problems.
                        Two mechanisms.
  (iii) CAPACITY     -- the apparent advantage is trainable-parameter count, and
                        vanishes once budgets are matched.

(iii) is why the A4 (attention rank 96) and A5 (feed-forward rank 11) arms exist:
they match the parameter budgets of A3 and A1 respectively, so subsystem and
capacity can be varied independently.

REPORTS
-------
  * paired 2x2 contingency tables with McNemar's test, per arm-pair per tier
  * capacity grid: subsystem x budget -> solve rates and conditional recovery
  * paired repair test: among problems BOTH arms failed at turn 1, who recovers?
    (the load-bearing comparison -- conditions on a shared failure set, so it
    isolates repair from single-shot capability)
  * first-solve-turn profile, recovery by turn-1 error class, degeneration,
    shortcut scan

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

# --------------------------------------------------------------------------- #
# Run registry. Adjust filenames here if yours differ.
# --------------------------------------------------------------------------- #
RUNS = {
    "base-single": "apps_eval_baseline_1turn_zeroshot_BASELINE.json",
    "base-loop": "apps_eval_baseline_3turn_last_failed_BASELINE.json",
    "S1-single": "apps_eval_reeval_run_1_single_shot.json",
    "A1-attn": "apps_eval_run_5_proper_phase1.json",
    "A2-all7": "apps_eval_run_A2_all_layers.json",
    "A3-mlp": "apps_eval_run_A3_mlp_only.json",
    "A4-attn-r96": "apps_eval_run_A4_attn_rank96.json",
    "A5-mlp-r11": "apps_eval_run_A5_mlp_rank11.json",
    "B1-attn": "apps_eval_run_B1_fixed_tests_attn_only.json",
    "B2-all7": "apps_eval_run_B2_fixed_tests_all_layers.json",
    "B3-mlp": "apps_eval_run_B3_fixed_tests_mlp_only.json",
    "C1-curr": "apps_eval_run_phase2_B1_curriculum.json",
}

# (subsystem, LoRA rank, trainable params in millions) -- for the capacity grid
ARM_META = {
    "base-single": ("--", 0, 0.00),
    "base-loop": ("--", 0, 0.00),
    "S1-single": ("attn", 32, 20.19),
    "A1-attn": ("attn", 32, 20.19),
    "A2-all7": ("all7", 32, 80.74),
    "A3-mlp": ("ffn", 32, 60.56),
    "A4-attn-r96": ("attn", 96, 60.56),
    "A5-mlp-r11": ("ffn", 11, 20.82),
    "B1-attn": ("attn", 32, 20.19),
    "B2-all7": ("all7", 32, 80.74),
    "B3-mlp": ("ffn", 32, 60.56),
    "C1-curr": ("attn", 32, 20.19),
}

# A series shares one feedback objective, so the capacity grid is drawn from it.
CAPACITY_GRID = [
    "base-loop",
    "A5-mlp-r11",
    "A1-attn",
    "A3-mlp",
    "A4-attn-r96",
    "A2-all7",
]

PAIRS_OF_INTEREST = [
    # --- original dissociation tests -------------------------------------- #
    ("A3-mlp", "A1-attn"),  # rank-matched: feed-forward vs attention
    ("A3-mlp", "A2-all7"),  # why does all-7 trail its own feed-forward subset?
    ("A2-all7", "A1-attn"),  # does all-7 land on the attention solution?
    ("B3-mlp", "B1-attn"),  # same test under unseen real tests
    ("C1-curr", "A3-mlp"),  # parity: same problems or different routes?
    ("C1-curr", "A1-attn"),  # what the curriculum phase added
    ("B2-all7", "B1-attn"),  # all-7 vs attention, B regime
    ("B3-mlp", "B2-all7"),  # feed-forward vs all-7, B regime
    (
        "C1-curr",
        "B1-attn",
    ),  # staging isolated: same adapter and objective, warm vs cold
    ("C1-curr", "B3-mlp"),  # attention curriculum vs feed-forward, matched objective
    # --- capacity-controlled tests (A4 / A5) ------------------------------ #
    ("A4-attn-r96", "A3-mlp"),  # matched 60.56 M: attention vs feed-forward
    ("A5-mlp-r11", "A1-attn"),  # matched ~20 M: the mirror comparison
    ("A4-attn-r96", "A1-attn"),  # capacity sweep within attention (3x)
    ("A3-mlp", "A5-mlp-r11"),  # capacity sweep within feed-forward (3x)
    ("C1-curr", "A4-attn-r96"),  # staging at 20 M vs raw capacity at 60 M
    # --- against the untrained loop --------------------------------------- #
    ("A1-attn", "base-loop"),  # which problems attention training adds
    ("A3-mlp", "base-loop"),  # which problems feed-forward training adds
]

# Pairs for the paired repair test (shared turn-1 failure set).
REPAIR_PAIRS = [
    ("A1-attn", "A5-mlp-r11"),  # matched ~20 M -- the load-bearing comparison
    ("A4-attn-r96", "A3-mlp"),  # matched 60.56 M
    ("A1-attn", "base-loop"),  # does attention training add repair at all?
    ("A5-mlp-r11", "base-loop"),  # does feed-forward at low budget add any?
    ("A4-attn-r96", "A1-attn"),  # does 3x capacity add repair within attention?
    ("A3-mlp", "A5-mlp-r11"),  # does 3x capacity add repair within feed-forward?
    ("C1-curr", "A4-attn-r96"),  # staging vs capacity
]

TIERS = ["introductory", "interview", "competition"]

ID_KEYS = ("problem_id", "id", "pid", "task_id", "index")
DIFF_KEYS = ("difficulty", "level", "tier", "bucket")
CODE_KEYS = ("completion", "code", "final_code", "solution", "generation")
HIST_KEYS = ("turn_history", "turns", "history", "trajectory")
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


# --------------------------------------------------------------------------- #
# schema helpers
# --------------------------------------------------------------------------- #
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
    if "InvalidFormat" in counts:  # a protocol failure takes precedence
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

    # Detect the turn-number base (this harness is 1-indexed).
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

        inf_t0 = turn_solved(ordered[0]) if ordered else None
        inf_lp = any(turn_solved(t) for t in ordered) if ordered else None
        if t0 is None:
            t0 = inf_t0 if inf_t0 is not None else False
        if lp is None:
            lp = inf_lp if inf_lp is not None else t0
        if inf_lp is not None and lp != inf_lp:
            warn += 1

        first_turn = None
        for j, t in enumerate(ordered):
            if turn_solved(t):
                n = first_key(t, TURN_KEYS)
                first_turn = (int(n) - turn_base) if isinstance(n, int) else j
                break

        out[pid] = {
            "tier": tier,
            "t0": bool(t0),
            "loop": bool(lp),
            "first_turn": first_turn,
            "code": code,
            "errs": [first_key(t, ERR_KEYS, default=[]) or [] for t in ordered],
            "n_turns": len(ordered),
        }

    if warn:
        print(f"    note: {warn} records where pass_at_N disagreed with turn_history")
    return out, data.get("summary", {}), turn_base


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def mcnemar(b, c):
    """Paired test on discordant cells. b = A only, c = B only."""
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


def z2(x1, n1, x2, n2):
    """Unpooled-free two-proportion z-test on counts."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (x1 / n1 - x2 / n2) / se if se > 0 else 0.0


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
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


def report_capacity_grid(runs, tier="introductory"):
    """subsystem x capacity -> solve rates and conditional repair rate"""
    print("\n" + "=" * 100)
    print(
        f"CAPACITY GRID ({tier})  -- subsystem and trainable budget varied independently"
    )
    print("=" * 100)
    print(
        f"{'run':14} {'sub':5} {'rank':>5} {'params':>9} {'solve@1':>8} {'solve@3':>8} "
        f"{'gap':>6} {'repair rate':>16}"
    )
    print("-" * 100)
    for name in CAPACITY_GRID:
        if name not in runs:
            continue
        run = runs[name][0]
        sel = [v for v in run.values() if v["tier"] == tier]
        if not sel:
            continue
        n = len(sel)
        s1 = sum(1 for v in sel if v["t0"])
        s3 = sum(1 for v in sel if v["loop"])
        fails = [v for v in sel if not v["t0"]]
        rec = sum(1 for v in fails if v["loop"])
        sub, rank, prm = ARM_META.get(name, ("?", 0, 0.0))
        rk = f"{rank}" if rank else "--"
        pm = f"{prm:.2f}M" if prm else "--"
        print(
            f"{name:14} {sub:5} {rk:>5} {pm:>9} {100*s1/n:7.1f}% {100*s3/n:7.1f}% "
            f"{100*(s3-s1)/n:+5.1f}  {rec:3d}/{len(fails):3d} = {100*rec/len(fails):5.1f}%"
        )
    print(
        "\nrepair rate = of problems failed at turn 1, the fraction solved later in the loop."
    )
    print("It controls for starting level, which the raw gap does not.")


def report_repair_paired(na, nb, ra, rb, tier="introductory"):
    """
    Among problems BOTH arms failed at turn 1, which arm repairs more?
    Conditioning on a shared failure set isolates repair from single-shot capability.
    """
    common = {
        p
        for p in (set(ra) & set(rb))
        if ra[p]["tier"] == tier and not ra[p]["t0"] and not rb[p]["t0"]
    }
    if not common:
        print(f"  [{tier}] no shared turn-1 failures")
        return
    A = {p for p in common if ra[p]["loop"]}
    B = {p for p in common if rb[p]["loop"]}
    a, b, c = len(A & B), len(A - B), len(B - A)
    n = len(common)
    lab, stat, p = mcnemar(b, c)
    zu = z2(len(A), n, len(B), n)
    print(f"  [{tier:13}] shared turn-1 failures = {n}")
    print(
        f"                 {na} repaired {len(A):3d} ({100*len(A)/n:5.1f}%)    "
        f"{nb} repaired {len(B):3d} ({100*len(B)/n:5.1f}%)"
    )
    print(f"                 both={a:3d}   {na}-only={b:3d}   {nb}-only={c:3d}")
    print(
        f"                 McNemar {lab}: stat={stat:.3f} p={p:.4f}{'  *SIG*' if p < .05 else ''}"
        f"   (unpaired z = {zu:+.2f})"
    )
    print()


def report_turns(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(
        f"FIRST-SOLVE-TURN PROFILE ({tier})   [turns labelled as in the result files]"
    )
    print("=" * 100)
    print(
        f"{'run':14} {'turn1':>7} {'turn2':>7} {'turn3':>7} {'total':>7} {'repair':>10}"
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
        print(f"{name:14} {c[0]:7d} {c[1]:7d} {c[2]:7d} {tot:7d} {rec:9.1f}%")


def report_recovery_by_error(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(f"REPAIR BY TURN-1 ERROR CLASS ({tier})  -- what each arm repairs FROM")
    print("=" * 100)
    classes = ["WrongOutput", "Timeout", "RuntimeError", "InvalidFormat", "None"]
    print(f"{'run':14} " + " ".join(f"{c[:11]:>13}" for c in classes))
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
        cells = [
            f"{100*num[c]/den[c]:6.1f}% {den[c]:4d}" if den[c] else f"{'--':>12}"
            for c in classes
        ]
        print(f"{name:14} " + " ".join(f"{x:>13}" for x in cells))


def report_degeneration(runs, tier="introductory"):
    print("\n" + "=" * 100)
    print(f"DEGENERATION PROFILE ({tier})  -- InvalidFormat rate by turn")
    print("=" * 100)
    print(f"{'run':14} {'turn1':>9} {'turn2':>9} {'turn3':>9}")
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
        print(f"{name:14} " + " ".join(row))


def report_shortcuts(runs):
    print("\n" + "=" * 100)
    print("SHORTCUT SCAN + CODE LENGTH")
    print("=" * 100)
    print(f"{'run':14} {'n':>5} {'shortcut%':>10} {'median':>8} {'p90':>7}")
    for name, (run, _s, _b) in runs.items():
        codes = [v["code"] for v in run.values() if v["code"]]
        if not codes:
            print(f"{name:14} {'--':>5}")
            continue
        hits = sum(1 for c in codes if any(p in c.lower() for p in SHORTCUT_PATTERNS))
        lens = sorted(len(c.split()) for c in codes)
        print(
            f"{name:14} {len(codes):5d} {100*hits/len(codes):9.2f}% "
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
        tot, exp1 = bd[tier].get("total"), bd[tier].get("pass_at_1")
        got = sum(1 for v in run.values() if v["tier"] == tier and v["t0"])
        cnt = sum(1 for v in run.values() if v["tier"] == tier)
        if exp1 is not None and tot:
            want = round(exp1 * tot)
            msgs.append(
                f"{tier[:5]}:{got}/{want}"
                + ("" if want == got and cnt == tot else " !!")
            )
    if msgs:
        print("    reconcile solve@1 vs summary -> " + "  ".join(msgs))


# --------------------------------------------------------------------------- #
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
                f"  {name:14} n={len(run):4d} solve@1={t0:3d} solve@k={lp:3d} turn_base={base}"
            )
            reconcile(name, run, summ)
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}")

    report_capacity_grid(runs)

    print("\n" + "=" * 100)
    print("PAIRED REPAIR TEST  -- conditioned on problems BOTH arms failed at turn 1")
    print("=" * 100)
    for na, nb in REPAIR_PAIRS:
        if na in runs and nb in runs:
            print(f"\n{na}  vs  {nb}")
            for tier in ("introductory", "interview"):
                report_repair_paired(na, nb, runs[na][0], runs[nb][0], tier)

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
    print("READING GUIDE")
    print("=" * 100)
    print("""
  CAPACITY GRID -- the primary result. Read the repair-rate column down each
      subsystem: if attention holds its rate across a 3x budget change while
      feed-forward's scales with budget, repair is attention-hosted and the
      apparent feed-forward advantage on solve@1 was capacity.

  PAIRED REPAIR TEST -- the load-bearing comparison, because it conditions on a
      shared turn-1 failure set and so cannot be explained by one arm simply
      solving more at first attempt.

  PAIRED OVERLAP, axis position:
      >= +0.80  one arm nearly subsumes the other -- one mechanism, more capacity
      <= +0.45  the arms solve materially different problems -- two mechanisms
      between   partial; report the contingency table and let McNemar's p carry it

  NOISE FLOOR -- repeat evaluation of one checkpoint moved individual tier metrics
      by about one problem (~0.4 pp), driven by timeout non-determinism. Differences
      of one or two problems are not interpretable.
""")


if __name__ == "__main__":
    main()
