"""
01_training_curves.py
Pulls training metrics from the Weights & Biases API.
Uses exact TAG matching to locate the runs dynamically, excludes evals,
and uses the verified custom metric keys to pull the history.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_PATH = "tarunbeerelli-northeastern-university/rlef-code"

# The complete, verified 8-run project tags
TARGET_RUNS = {
    "R1: Sparse Baseline": "sparse_baseline",
    "R2: Dense Baseline": "dense_baseline",
    "R3: Multi-Turn Standard": "multi_turn_baseline",
    "R4: Macro Heuristic": "macro_heuristic",
    "R5: Multi-Turn Targeted": "targeted_feedback",
    "R6: Cold-Start TDD": "anchored_tdd",
    "R7: Curriculum Phase 1": "phase_1_execution",
    "R7: Curriculum Phase 2": "phase_2_tdd",
}


def fetch_run_history():
    api = wandb.Api()
    all_data = []

    for label, tag_name in TARGET_RUNS.items():
        try:
            # Match the highly specific architecture tag, strictly exclude eval runs
            filters = {"tags": {"$in": [tag_name], "$nin": ["eval"]}}

            runs = api.runs(PROJECT_PATH, filters=filters)
            if not runs:
                print(f"⚠️ No training runs found with tag: {tag_name}")
                continue

            valid_run_found = False

            for run in runs:
                # Double-check it's not an eval run just in case
                if "eval" in run.tags:
                    continue

                # The EXACT keys verified from your W&B dashboard
                history = run.history(
                    keys=["train/avg_reward", "metrics/success_rate", "train/loss"],
                    samples=150,
                )

                if not history.empty:
                    print(f"🔄 Pulled valid data for {label} (Run ID: {run.name})")
                    history["Model"] = label
                    history["Step"] = history.index
                    all_data.append(history)
                    valid_run_found = True
                    break

            if not valid_run_found:
                print(
                    f"⚠️ Found runs for tag '{tag_name}', but they were empty or evals."
                )

        except Exception as e:
            print(f"Error fetching data for tag {tag_name}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def plot_training_curves(df):
    if df.empty:
        print("❌ No training data retrieved. Cannot plot curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("husl", 8)

    # 1. Reward Dynamics
    sns.lineplot(
        data=df,
        x="Step",
        y="train/avg_reward",
        hue="Model",
        ax=axes[0],
        palette=palette,
        linewidth=1.5,
    )
    axes[0].set_title("Reward Dynamics", fontweight="bold", fontsize=14, pad=15)
    axes[0].set_ylabel("Reward Signal")

    # 2. Pass Rate Convergence
    sns.lineplot(
        data=df,
        x="Step",
        y="metrics/success_rate",
        hue="Model",
        ax=axes[1],
        palette=palette,
        linewidth=1.5,
    )
    axes[1].set_title(
        "Training Pass Rate Convergence", fontweight="bold", fontsize=14, pad=15
    )
    axes[1].set_ylabel("Pass Rate")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # 3. Loss Curves
    sns.lineplot(
        data=df,
        x="Step",
        y="train/loss",
        hue="Model",
        ax=axes[2],
        palette=palette,
        linewidth=1.5,
    )
    axes[2].set_title(
        "Optimization Loss Curves", fontweight="bold", fontsize=14, pad=15
    )
    axes[2].set_ylabel("Training Loss")

    for i, ax in enumerate(axes):
        ax.set_xlabel("Training Step")
        if i == 2:
            ax.legend(
                title="Model Suite",
                fontsize=10,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        else:
            ax.get_legend().remove()

    sns.despine()
    plt.tight_layout()
    plt.savefig("analysis_01_training_curves.png", dpi=300, bbox_inches="tight")
    print("\n🎉 Chart rendered successfully. Saved to: analysis_01_training_curves.png")


if __name__ == "__main__":
    print("🚀 Initiating cloud sweep using verified tags and metric keys...")
    df = fetch_run_history()
    plot_training_curves(df)
