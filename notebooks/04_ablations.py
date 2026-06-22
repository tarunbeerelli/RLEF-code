# %% [markdown]
# # Ablation Comparison
#
# Compares ablation runs side by side.
# Run after all ablation training jobs complete.
# Each ablation result must be in results/apps_eval_{tag}.json

# %%
import json
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

COLORS = ["#2563EB", "#EA580C", "#16A34A", "#7C3AED", "#DB2777", "#6B7280"]

# %%
# ── Load all ablation results ─────────────────────────────────────────────────
# Add tags here as you complete each ablation run

ABLATION_TAGS = {
    "Baseline (no training)": "baseline",
    "Single-turn RLVR (ours)": "trained",
    # add these after running ablations:
    # "Binary reward":            "ablation_binary_reward",
    # "No shaping":               "ablation_no_shaping",
    # "Trajectory credit":        "ablation_traj_credit",
    # "Multi-turn (3 turns)":     "ablation_multi_turn",
}


def load_summary(tag: str) -> dict | None:
    path = Path(f"../results/apps_eval_{tag}.json")
    if not path.exists():
        print(f"Missing: {path} — skipping.")
        return None
    with open(path) as f:
        return json.load(f)["summary"]


summaries = {label: load_summary(tag) for label, tag in ABLATION_TAGS.items()}
summaries = {k: v for k, v in summaries.items() if v is not None}

# %%
# ── Main comparison table ─────────────────────────────────────────────────────

difficulties = ["introductory", "interview", "competition"]
rows = []

for label, s in summaries.items():
    row = {"Condition": label, "Overall": s["pass_at_1"]}
    for d in difficulties:
        row[d.capitalize()] = s["by_difficulty"].get(d, {}).get("pass_at_1", 0.0)
    rows.append(row)

df = pd.DataFrame(rows).sort_values("Overall", ascending=False)
print("=== Ablation comparison ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.1%}"))
df.to_csv("../analysis/ablation_comparison.csv", index=False)
print("Saved: analysis/ablation_comparison.csv")

# %%
# ── Bar chart comparison ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df))
w = 0.18
metrics = ["Overall"] + [d.capitalize() for d in difficulties]
offsets = np.linspace(
    -(len(metrics) - 1) * w / 2, (len(metrics) - 1) * w / 2, len(metrics)
)

for metric, offset, color in zip(metrics, offsets, COLORS):
    vals = df[metric].values
    bars = ax.bar(x + offset, vals, width=w, alpha=0.85, color=color, label=metric)

ax.set_xticks(x)
ax.set_xticklabels(df["Condition"].values, rotation=12, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Ablation comparison — APPS test pass@1", fontweight="bold")
ax.set_ylabel("pass@1")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("../analysis/ablation_comparison.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/ablation_comparison.png")

# %%
# ── Delta vs baseline ─────────────────────────────────────────────────────────

baseline_overall = summaries.get("Baseline (no training)", {}).get("pass_at_1", 0)

print("\n=== Delta vs baseline (overall pass@1) ===")
for label, s in summaries.items():
    delta = s["pass_at_1"] - baseline_overall
    marker = "✓" if delta > 0 else "✗"
    print(f"  {marker} {label:<35} {delta:+.1%}")
