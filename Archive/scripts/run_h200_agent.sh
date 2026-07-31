#!/bin/bash
# run_h200_agent.sh - Launches Trajectory PPO on a single H200 (141GB)

set -x

# 1. Start the Ray cluster locally on the H200
ray stop
ray start --head --num-gpus=1

# 2. Launch the OpenRLHF Agent pipeline
# We use PPO/GRPO Ray engine, injecting our custom env.py via agent_func_path
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain "Qwen/Qwen2.5-Coder-7B-Instruct" \
   --save_path "./checkpoints/h200_agent_rl" \
   --micro_train_batch_size 4 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 32 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data "data/openrlhf_apps_train.jsonl" \
   --input_key "prompt" \
   --enable_agent_trajectory \
   --agent_func_path "src.rlef.env.CodingEnvironment" \
   --max_agent_turns 5 \
   --vllm_gpu_memory_utilization 0.4 \
   --flash_attn \
   --gradient_checkpointing \
   --wandb_project "rlef-code-h200"
