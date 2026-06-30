#!/usr/bin/env bash
# ==============================================================================
# scripts/run_openrlhf.sh
# ==============================================================================
set -eo pipefail

# 1. Start the Ray background cluster on the instance
ray stop --force 2>/dev/null || true
ray start --head --disable-usage-stats --num-gpus=2

# 2. Run OpenRLHF optimization using the hierarchical dot configuration format
python -m openrlhf.cli.train_ppo_ray \
   --actor.model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
   --algo.advantage.estimator group_norm \
   --reward.remote_url src/rlef/reward_bridge.py \
   --data.prompt_dataset ./data/openrlhf_prompts.jsonl \
   --data.input_key prompt \
   --ckpt.output_dir ./checkpoints/openrlhf_out \
   --data.label_key label \
   --train.train_batch_size 128 \
   --rollout.batch_size 512 \
   --ds.zero_stage 3 \
   --train.bf16 true \
   --actor.lora_enable true \
   --actor.lora_r 16 \
   --actor.lora_alpha 32
