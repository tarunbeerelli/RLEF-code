#!/bin/bash
# run_grpo_docker.sh - Optimized GRPO pipeline execution for a single-node H200 setup

set -x

# 1. Clear any residual cluster states
ray stop

# 2. Hard infrastructure overrides to prevent network handshaking stalls
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export PYTHONWARNINGS="ignore"
export PYTHONPATH=/workspace/src:/workspace:$PYTHONPATH

# 3. Spin up the cluster orchestration head
ray start --head --num-gpus=1

# 4. Fire up the optimization pipeline via direct module execution
python3 -m openrlhf.cli.train_ppo_ray \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 1 \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node 1 \
  --vllm.num_engines 1 \
  --vllm.gpu_memory_utilization 0.45 \
  --vllm.enforce_eager \
  --vllm.sync_backend nccl \
  --train.colocate_all \
  --vllm.enable_sleep \
  --ds.enable_sleep \
  --ds.attn_implementation sdpa \
  --actor.model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --data.prompt_dataset ./data/openrlhf_apps_train.jsonl \
  --max_turn 5 \
  --data.input_key prompt \
  --data.apply_chat_template \
  --ckpt.output_dir ./checkpoint/rlef-7b-grpo \
  --train.batch_size 32 \
  --train.micro_batch_size 2 \
  --rollout.batch_size 256 \
  --rollout.n_samples_per_prompt 8 \
  --algo.advantage.estimator group_norm \
  --algo.kl.use_loss \
  --algo.kl.estimator k3 \
  --data.max_len 2048 \
  --rollout.max_new_tokens 1024 \
  --actor.adam.lr 1e-6 \
  --actor.gradient_checkpointing_enable \
  --logger.wandb.project rlef-grpo-single-h200 \
  --ds.param_dtype bf16
