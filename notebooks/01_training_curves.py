# %% [markdown]
# # Training Curves
#
# Pulls reward and loss curves from WandB for the main training run
# and any ablation runs. Run this during or after training.
#
# Requires: wandb login, a completed or in-progress training run.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

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

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#16A34A"
GRAY = "#6B7280"

# %%
# ── Config ────────────────────────────────────────────────────────────────────
# Fill these in after your first training run

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "your-wandb-username")
WANDB_PROJECT = "rlef-code"

# runs to compare — add ablation run names here later
RUNS = {
    "Single-turn RLVR": "rlef-code",  # default run name from train.yaml
}


# %%
def fetch_history(run_name: str, keys: list[str]) -> pd.DataFrame:
    """Pull metric history from WandB by run name."""
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"display_name": run_name},
    )
    if not runs:
        print(f"Run '{run_name}' not found — using synthetic data for layout preview.")
        return _synthetic_run(run_name, keys)

    run = runs[0]
    history = run.history(keys=keys, pandas=True)
    print(f"Fetched {len(history)} steps for '{run_name}'")
    return history


def _synthetic_run(name: str, keys: list[str]) -> pd.DataFrame:
    """
    Synthetic data so the notebook renders before you have real results.
    Replace this with real WandB data after training.
    """
    steps = np.arange(200)
    data = {"_step": steps, "run": name}
    for key in keys:
        if "reward" in key:
            data[key] = (
                0.05 + 0.20 * (1 - np.exp(-steps / 80)) + np.random.normal(0, 0.01, 200)
            )
        elif "loss" in key:
            data[key] = (
                2.5 - 0.6 * (1 - np.exp(-steps / 100)) + np.random.normal(0, 0.03, 200)
            )
        else:
            data[key] = np.random.uniform(0, 1, 200)
    return pd.DataFrame(data)


def smooth(series: pd.Series, window: int = 15) -> pd.Series:
    """Rolling mean to reduce step-level noise. Show both raw and smoothed."""
    return series.rolling(window, min_periods=1).mean()


# %%
# ── Fetch data ────────────────────────────────────────────────────────────────

METRICS = [
    "episode/final_reward",
    "episode/pass_rate",
    "train/loss",
    "tools/execute",
    "tools/lint",
    "tools/generate_tests",
]

histories = {
    label: fetch_history(run_name, METRICS) for label, run_name in RUNS.items()
}

# %%
# ── Plot 1: Reward + loss ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = [BLUE, ORANGE, GREEN, GRAY]

for ax, metric, title, ylabel in [
    (axes[0], "episode/final_reward", "Reward over training", "Mean episode reward"),
    (axes[1], "episode/pass_rate", "Pass rate over training", "Pass rate"),
    (axes[2], "train/loss", "Loss over training", "Training loss"),
]:
    for (label, df), color in zip(histories.items(), colors):
        if metric not in df.columns:
            continue
        ax.plot(df["_step"], df[metric], alpha=0.15, color=color, lw=0.8)
        ax.plot(
            df["_step"], smooth(df[metric]), alpha=1.0, color=color, lw=2.0, label=label
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)

    if "rate" in metric or "reward" in metric:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig("../analysis/training_curves.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/training_curves.png")

# %%
# ── Plot 2: Tool usage over training ─────────────────────────────────────────
# Only meaningful for multi-turn runs — shows whether the model learns
# to use lint and test generation strategically over time.

fig, ax = plt.subplots(figsize=(10, 4))

for (label, df), color in zip(histories.items(), colors):
    tool_cols = ["tools/execute", "tools/lint", "tools/generate_tests"]
    tool_cols = [c for c in tool_cols if c in df.columns]
    if not tool_cols:
        continue

    for col, ls in zip(tool_cols, ["-", "--", "-."]):
        tool_name = col.split("/")[1]
        ax.plot(
            df["_step"],
            smooth(df[col], window=20),
            color=color,
            linestyle=ls,
            lw=1.8,
            label=f"{label} — {tool_name}",
        )

ax.set_xlabel("Training step")
ax.set_ylabel("Tool calls per episode")
ax.set_title("Tool usage over training", fontweight="bold")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../analysis/tool_usage.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved: analysis/tool_usage.png")

# %%
# ── Summary stats ─────────────────────────────────────────────────────────────

for label, df in histories.items():
    if "episode/final_reward" not in df.columns:
        continue
    r = df["episode/final_reward"]
    print(f"\n{label}")
    print(f"  Steps:          {len(df)}")
    print(f"  Initial reward: {r.iloc[:10].mean():.3f}")
    print(f"  Final reward:   {r.iloc[-20:].mean():.3f}")
    print(f"  Peak reward:    {r.max():.3f}")
    delta = r.iloc[-20:].mean() - r.iloc[:10].mean()
    print(f"  Improvement:    {delta:+.3f}")
