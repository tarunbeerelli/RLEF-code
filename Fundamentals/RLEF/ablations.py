"""
ablations.py — Ablation configuration generator

Generates config variants for each ablation condition.
Each variant is a copy of the base config with one thing changed.

Ablations:
  A. reward_type:  binary vs continuous
  B. shaped:       true vs false
  C. credit_type:  trajectory vs step
  D. max_turns:    1 vs 3
  E. tools:        none vs lint vs lint+tests (via allowed_tools flag)

Run all ablations:
  python -m rlef.ablations --base configs/train.yaml --output configs/ablations/

Then train each:
  for cfg in configs/ablations/*.yaml:
      accelerate launch src/rlef/train.py --config $cfg
"""

import argparse
import copy
from pathlib import Path

import yaml

ABLATIONS = [
    # (name, {overrides})
    ("binary_reward", {"reward_type": "binary", "shaped": False}),
    ("no_shaping", {"shaped": False}),
    ("traj_credit", {"credit_type": "trajectory"}),
    ("single_turn", {"max_turns": 1}),
    ("multi_turn", {"max_turns": 3}),
]


def generate_ablation_configs(base_cfg: dict, output_dir: Path) -> list[Path]:
    """
    Generate one YAML config per ablation condition.
    Each config is the base config with one field overridden.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # save base config too so we have the full set
    base_path = output_dir / "base.yaml"
    with open(base_path, "w") as f:
        yaml.dump(base_cfg, f, default_flow_style=False)
    paths.append(base_path)
    print(f"Saved base config: {base_path}")

    for name, overrides in ABLATIONS:
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)

        # tag the run name so WandB distinguishes them
        cfg["wandb_run_name"] = f"ablation_{name}"
        cfg["output_dir"] = f"./checkpoints/ablation_{name}"

        path = output_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"Saved ablation config: {path} — overrides: {overrides}")
        paths.append(path)

    return paths


def compare_results(results_dir: Path) -> None:
    """
    Print a comparison table of ablation results.
    Reads results/*.json files and prints pass@1 for each condition.
    """
    import json

    rows = []
    for path in sorted(results_dir.glob("apps_eval_*.json")):
        with open(path) as f:
            data = json.load(f)
        s = data["summary"]
        rows.append(
            {
                "tag": s["tag"],
                "pass@1": s["pass_at_1"],
                "solved": s["solved"],
                "total": s["total"],
            }
        )

    if not rows:
        print("No result files found in", results_dir)
        return

    print(f"\n{'Tag':<30} {'pass@1':>8} {'solved':>8} {'total':>8}")
    print("-" * 58)
    for r in sorted(rows, key=lambda x: x["pass@1"], reverse=True):
        print(f"{r['tag']:<30} {r['pass@1']:>8.1%} {r['solved']:>8} {r['total']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/train.yaml")
    parser.add_argument("--output", type=str, default="configs/ablations")
    parser.add_argument(
        "--compare", action="store_true", help="Print comparison table from results/"
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(Path("results"))
        return

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)

    generate_ablation_configs(base_cfg, Path(args.output))
    print(f"\nGenerated {len(ABLATIONS) + 1} configs in {args.output}/")
    print("Train each with:")
    print(
        "  accelerate launch --config_file accelerate_config.yaml "
        "src/rlef/train.py --config configs/ablations/<name>.yaml"
    )


if __name__ == "__main__":
    main()
