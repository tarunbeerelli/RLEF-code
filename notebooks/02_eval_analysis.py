"""
02_eval_analysis.py
Generates the Overall Pass Rate Summary and the Pass Rates
Partitioned by Complexity Profile across ALL 8 models.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# The complete 8-run local file matrix
FILES = {
    "R1: Sparse Baseline": "results/apps_eval_run_1_sparse_baseline_final.json",
    "R2: Dense Baseline": "results/apps_eval_run_2_dense_baseline_final.json",
    "R3: Multi-Turn Standard": "results/apps_eval_run_3_multi_turn_baseline_final.json",
    "R4: Macro Heuristic": "results/apps_eval_run_4_macro_heuristic_final.json",
    "R5: Multi-Turn Targeted": "results/apps_eval_run_5_multi_turn_targeted_final.json",
    "R6: Cold-Start TDD": "results/apps_eval_run_6_anchored_tdd_final.json",
    "R7: Curriculum Phase 1": "results/apps_eval_run_7_curriculum_phase_1_final.json",
    "R7: Curriculum Phase 2": "results/apps_eval_run_7_curriculum_phase_2_final.json",
}


def load_summary_data():
    records = []

    for model_name, path in FILES.items():
        if not Path(path).exists():
            print(f"⚠️ Missing {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})

        # Overall
        records.append(
            {
                "Model": model_name,
                "Category": "Overall",
                "Pass Rate": summary.get("pass_at_1", 0.0) * 100,
            }
        )

        # By Difficulty
        by_diff = summary.get("by_difficulty", {})
        for diff in ["introductory", "interview", "competition"]:
            if diff in by_diff:
                records.append(
                    {
                        "Model": model_name,
                        "Category": diff.capitalize(),
                        "Pass Rate": by_diff[diff].get("pass_at_1", 0.0) * 100,
                    }
                )

    return pd.DataFrame(records)


def plot_complexity_profiles(df):
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", 8)

    overall_df = df[df["Category"] == "Overall"]
    diff_df = df[df["Category"] != "Overall"]

    # 1. Overall Pass Rate
    sns.barplot(data=overall_df, x="Model", y="Pass Rate", ax=axes[0], palette=palette)
    axes[0].set_title("Overall Pass Rate Summary", fontweight="bold", fontsize=16)
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=45)

    # 2. Partitioned Pass Rate
    sns.barplot(
        data=diff_df,
        x="Category",
        y="Pass Rate",
        hue="Model",
        ax=axes[1],
        palette=palette,
    )
    axes[1].set_title(
        "Pass Rates Partitioned by Complexity Profile", fontweight="bold", fontsize=16
    )
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    sns.despine()
    plt.tight_layout()
    plt.savefig("analysis_02_complexity.png", dpi=300, bbox_inches="tight")
    print("✅ Saved analysis_02_complexity.png")


if __name__ == "__main__":
    df = load_summary_data()
    plot_complexity_profiles(df)
