# %% [markdown]
# # Eval Analysis
#
# Reads baseline and trained eval results from results/.
# Produces per-difficulty breakdown, error analysis, before/after table.
#
# Run eval.py before this notebook:
#   python -m rlef.eval --checkpoint none --tag baseline --benchmark all
#   python -m rlef.eval --checkpoint ./checkpoints/final --tag trained --benchmark all

# %%
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

Path("../analysis").mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

BLUE, ORANGE, GREEN, GRAY = "#2563EB", "#EA580C", "#16A34A", "#6B7280"


# %%
def load_eval(path: str) -> tuple[dict, pd.DataFrame]:
    """Load eval result JSON into summary dict and examples DataFrame."""
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path} — using synthetic data.")
        return _synthetic_eval(path)

    with open(p) as f:
        data = json.load(f)

    summary = data["summary"]
    df = pd.DataFrame(data["results"])
    print(f"Loaded {len(df)} results from {path}")
    print(f"  pass@1: {summary['pass_at_1']:.1%}")
    return summary, df


def _synthetic_eval(path: str) -> tuple[dict, pd.DataFrame]:
    """Synthetic stand-in so notebook renders without real results."""
    is_trained = "trained" in path
    base_rate = 0.38 if not is_trained else 0.47

    rows = []
    for i in range(200):
        diff = ["introductory", "interview", "competition"][i % 3]
        rate = base_rate + (
            0.15 if diff == "introductory" else 0.05 if diff == "interview" else 0.0
        )
        rows.append(
            {
                "problem_id": str(i),
                "difficulty": diff,
                "pass_rate": float(np.random.random() < rate),
                "error_types": ["WrongOutput"] if np.random.random() > rate else [],
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "tag": "synthetic",
        "pass_at_1": df["pass_rate"].mean(),
        "solved": int(df["pass_rate"].sum()),
        "total": len(df),
        "by_difficulty": {
            d: {
                "pass_at_1": df[df["difficulty"] == d]["pass_rate"].mean(),
                "solved": int(df[df["difficulty"] == d]["pass_rate"].sum()),
                "total": len(df[df["difficulty"] == d]),
            }
            for d in df["difficulty"].unique()
        },
    }
    return summary, df


# %%
# ── Load results ──────────────────────────────────────────────────────────────

base_summary, base_df = load_eval("../results/apps_eval_baseline.json")
trd_summary, trd_df = load_eval("../results/apps_eval_trained.json")

# %%
# ── Plot 1: Overall + per-difficulty pass@1 ───────────────────────────────────

difficulties = ["introductory", "interview", "competition"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# overall
ax = axes[0]
labels = ["Baseline", "RLVR trained"]
values = [base_summary["pass_at_1"], trd_summary["pass_at_1"]]
bars = ax.bar(labels, values, color=[BLUE, ORANGE], alpha=0.85, width=0.4)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.005,
        f"{val:.1%}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, max(values) * 1.25)
ax.set_title("Overall pass@1 — APPS test", fontweight="bold")
ax.set_ylabel("pass@1")

# per difficulty
ax = axes[1]
x = np.arange(len(difficulties))
w = 0.32
base_vals = [
    base_summary["by_difficulty"].get(d, {}).get("pass_at_1", 0) for d in difficulties
]
trd_vals = [
    trd_summary["by_difficulty"].get(d, {}).get("pass_at_1", 0) for d in difficulties
]

ax.bar(x - w / 2, base_vals, width=w, color=BLUE, alpha=0.85, label="Baseline")
ax.bar(x + w / 2, trd_vals, width=w, color=ORANGE, alpha=0.85, label="RLVR trained")

for i, (b, t) in enumerate(zip(base_vals, trd_vals)):
    delta = t - b
    color = GREEN if delta > 0 else "red"
    ax.annotate(
        f"{delta:+.1%}",
        xy=(x[i] + w / 2, t + 0.005),
        ha="center",
        fontsize=9,
        color=color,
        fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels([d.capitalize() for d in difficulties])
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("pass@1 by difficulty", fontweight="bold")
ax.set_ylabel("pass@1")
ax.legend()

plt.tight_layout()
plt.savefig("../analysis/eval_breakdown.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/eval_breakdown.png")

# %%
# ── Plot 2: Error type breakdown ──────────────────────────────────────────────
# Which errors does RLVR fix? Which persist?
# SyntaxError should decrease (lint helps). WrongOutput is the hard problem.


def count_errors(df: pd.DataFrame) -> dict:
    counts: Counter = Counter()
    for errors in df["error_types"].dropna():
        if isinstance(errors, list):
            counts.update(errors)
    return dict(counts)


base_errors = count_errors(base_df)
trd_errors = count_errors(trd_df)
all_types = sorted(set(base_errors) | set(trd_errors))

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(all_types))
w = 0.32
ax.bar(
    x - w / 2,
    [base_errors.get(t, 0) for t in all_types],
    width=w,
    color=BLUE,
    alpha=0.85,
    label="Baseline",
)
ax.bar(
    x + w / 2,
    [trd_errors.get(t, 0) for t in all_types],
    width=w,
    color=ORANGE,
    alpha=0.85,
    label="RLVR trained",
)
ax.set_xticks(x)
ax.set_xticklabels(all_types, rotation=15)
ax.set_title("Error type counts — failures only", fontweight="bold")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig("../analysis/error_breakdown.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/error_breakdown.png")

# %%
# ── Plot 3: Paired outcome analysis ───────────────────────────────────────────
# For each problem: did baseline solve it? Did RLVR solve it?
# Fixed/broke/both correct/both wrong

merged = base_df[["problem_id", "difficulty", "pass_rate"]].merge(
    trd_df[["problem_id", "pass_rate"]],
    on="problem_id",
    suffixes=("_base", "_rlvr"),
)


def outcome(row):
    b = row["pass_rate_base"] == 1.0
    r = row["pass_rate_rlvr"] == 1.0
    if b and r:
        return "Both correct"
    if not b and r:
        return "RLVR fixed"
    if b and not r:
        return "RLVR broke"
    return "Both wrong"


merged["outcome"] = merged.apply(outcome, axis=1)
counts = merged["outcome"].value_counts()
print("\nOutcome breakdown:")
print(counts)
print(
    f"\nNet gain: {counts.get('RLVR fixed',0) - counts.get('RLVR broke',0):+d} problems"
)

fig, ax = plt.subplots(figsize=(7, 4))
colors_map = {
    "Both correct": GREEN,
    "RLVR fixed": ORANGE,
    "RLVR broke": "red",
    "Both wrong": GRAY,
}
bars = ax.bar(
    counts.index,
    counts.values,
    color=[colors_map.get(k, GRAY) for k in counts.index],
    alpha=0.85,
)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        str(val),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
ax.set_title("Paired outcome analysis", fontweight="bold")
ax.set_ylabel("Number of problems")

plt.tight_layout()
plt.savefig("../analysis/paired_outcomes.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/paired_outcomes.png")

# %%
# ── Summary table ─────────────────────────────────────────────────────────────

summary_rows = [
    [
        "Overall pass@1",
        f"{base_summary['pass_at_1']:.1%}",
        f"{trd_summary['pass_at_1']:.1%}",
        f"{trd_summary['pass_at_1']-base_summary['pass_at_1']:+.1%}",
    ],
]
for d in difficulties:
    b = base_summary["by_difficulty"].get(d, {}).get("pass_at_1", 0)
    t = trd_summary["by_difficulty"].get(d, {}).get("pass_at_1", 0)
    summary_rows.append(
        [f"{d.capitalize()} pass@1", f"{b:.1%}", f"{t:.1%}", f"{t-b:+.1%}"]
    )

summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Baseline", "RLVR", "Delta"])
print("\n=== Summary table ===")
print(summary_df.to_string(index=False))
summary_df.to_csv("../analysis/summary_table.csv", index=False)
print("Saved: analysis/summary_table.csv")
