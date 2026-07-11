"""
04_ablations.py
Generates the Full Unified Project Horizon Strategy Matrix comparing
the zero-shot pass rates across ALL 8 RLHF iterations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# The complete 8-run local file matrix
ABLATION_FILES = {
    "R1: Sparse Baseline": "results/apps_eval_run_1_sparse_baseline_final.json",
    "R2: Dense Baseline": "results/apps_eval_run_2_dense_baseline_final.json",
    "R3: Multi-Turn Standard": "results/apps_eval_run_3_multi_turn_baseline_final.json",
    "R4: Macro Heuristic": "results/apps_eval_run_4_macro_heuristic_final.json",
    "R5: Multi-Turn Targeted": "results/apps_eval_run_5_multi_turn_targeted_final.json",
    "R6: Cold-Start TDD": "results/apps_eval_run_6_anchored_tdd_final.json",
    "R7: Curriculum Phase 1": "results/apps_eval_run_7_curriculum_phase_1_final.json",
    "R7: Curriculum Phase 2": "results/apps_eval_run_7_curriculum_phase_2_final.json",
}


def load_ablation_data():
    records = []
    for model_name, filepath in ABLATION_FILES.items():
        if not Path(filepath).exists():
            print(f"⚠️ Missing {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})

        # Aggregate
        records.append(
            {
                "Model": model_name,
                "Difficulty": "Aggregate Profile",
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
                        "Difficulty": diff.capitalize(),
                        "Pass Rate": by_diff[diff].get("pass_at_1", 0.0) * 100,
                    }
                )

    return pd.DataFrame(records)


def plot_unified_matrix(df):
    if df.empty:
        print("No data available to plot.")
        return

    plt.figure(figsize=(16, 9))
    sns.set_theme(style="whitegrid")

    # 4 clean structural groups for visual isolation
    colors = ["#2c3e50", "#2980b9", "#e67e22", "#e74c3c"]

    ax = sns.barplot(
        data=df, x="Model", y="Pass Rate", hue="Difficulty", palette=colors
    )

    plt.title(
        "Unified Project Horizon Strategy Matrix (Zero-Shot Pass Rate - All Runs)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Definitive Pass Rate (%)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right", fontsize=11)

    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.legend(title="", fontsize=11, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine()
    plt.tight_layout()
    plt.savefig("analysis_unified_matrix.png", dpi=300, bbox_inches="tight")
    print("✅ Saved analysis_unified_matrix.png")


if __name__ == "__main__":
    print("Aggregating complete ablation files...")
    df = load_ablation_data()
    plot_unified_matrix(df)
