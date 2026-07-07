"""
split_dataset.py — Run 7 Curriculum Partitioning
Physically separates the APPS dataset into disjoint splits to prevent 
catastrophic forgetting and data contamination across phases.
"""

import json
import random
from pathlib import Path

def main():
    raw_path = Path("data/openrlhf_apps_train.jsonl") # Point this to your prep script output
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run prepare_openrlhf_data.py first.")
        return

    with open(raw_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Shuffle for randomness
    random.seed(42)
    random.shuffle(dataset)

    total = len(dataset)
    midpoint = total // 2

    # 50/50 Disjoint Split
    split_a = dataset[:midpoint]
    split_b = dataset[midpoint:]

    # 10% Replay Buffer from Split A
    replay_count = max(1, int(len(split_a) * 0.10))
    replay_buffer = random.sample(split_a, replay_count)

    # Phase 2 Dataset (Split B + Replay)
    phase_2_dataset = split_b + replay_buffer
    random.shuffle(phase_2_dataset) # Mix the replay buffer in

    phase_1_path = Path("data/apps_run7_phase1.jsonl")
    phase_2_path = Path("data/apps_run7_phase2.jsonl")

    with open(phase_1_path, "w") as f:
        for row in split_a: f.write(json.dumps(row) + "\n")

    with open(phase_2_path, "w") as f:
        for row in phase_2_dataset: f.write(json.dumps(row) + "\n")

    print(f"Dataset Partitioning Complete (Total: {total}):")
    print(f"- Phase 1 (Split A): {len(split_a)} rows -> {phase_1_path}")
    print(f"- Phase 2 (Split B + 10% Replay): {len(phase_2_dataset)} rows -> {phase_2_path}")

if __name__ == "__main__":
    main()"""
split_dataset.py — Run 7 Curriculum Partitioning
Physically separates the APPS dataset into disjoint splits to prevent 
catastrophic forgetting and data contamination across phases.
"""

import json
import random
from pathlib import Path

def main():
    raw_path = Path("data/openrlhf_apps_train.jsonl") # Point this to your prep script output
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run prepare_openrlhf_data.py first.")
        return

    with open(raw_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Shuffle for randomness
    random.seed(42)
    random.shuffle(dataset)

    total = len(dataset)
    midpoint = total // 2

    # 50/50 Disjoint Split
    split_a = dataset[:midpoint]
    split_b = dataset[midpoint:]

    # 10% Replay Buffer from Split A
    replay_count = max(1, int(len(split_a) * 0.10))
    replay_buffer = random.sample(split_a, replay_count)

    # Phase 2 Dataset (Split B + Replay)
    phase_2_dataset = split_b + replay_buffer
    random.shuffle(phase_2_dataset) # Mix the replay buffer in

    phase_1_path = Path("data/apps_run7_phase1.jsonl")
    phase_2_path = Path("data/apps_run7_phase2.jsonl")

    with open(phase_1_path, "w") as f:
        for row in split_a:
            f.write(json.dumps(row) + "\n")

    with open(phase_2_path, "w") as f:
        for row in phase_2_dataset:
            f.write(json.dumps(row) + "\n")

    print(f"Dataset Partitioning Complete (Total: {total}):")
    print(f"- Phase 1 (Split A): {len(split_a)} rows -> {phase_1_path}")
    print(f"- Phase 2 (Split B + 10% Replay): {len(phase_2_dataset)} rows -> {phase_2_path}")

if __name__ == "__main__":
    main()