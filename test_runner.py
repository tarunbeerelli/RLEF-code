import os
import sys

sys.path.append(os.path.abspath("src"))

import numpy as np
from rlef.data import load_apps_split
from rlef.reward import execution_reward


def run_real_baseline():
    # Load just 20 problems
    problems = load_apps_split(
        "data/apps", split="train", difficulties=["introductory"]
    )
    subset = problems[:20]

    print(f"--- Validating Reward Signal on {len(subset)} real problems ---")
    rewards = []

    for p in subset:
        # Use the FIRST solution from the dataset as the candidate code
        # This acts as our "Perfect" solution baseline
        code = p.solutions[0]

        # Call your actual reward function
        res = execution_reward(
            code=code,
            inputs=[str(x) for x in p.inputs],
            outputs=[str(x) for x in p.outputs],
            fn_name=p.fn_name,
            difficulty=p.difficulty,
            current_turn=1,
        )

        rewards.append(res.final_reward)
        print(
            f"Prob {p.problem_id}: PassRate={res.pass_rate:.2f}, RawReward={res.final_reward:.4f}"
        )

    # Verify if we have variance
    print(f"\nMean Reward: {np.mean(rewards):.4f}")
    print(f"Std Dev: {np.std(rewards):.4f}")


if __name__ == "__main__":
    run_real_baseline()
