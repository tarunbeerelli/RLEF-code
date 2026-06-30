set -x

export PYTHONPATH=src:.:$PYTHONPATH

# Fire up Ray bound strictly to your single local H200 accelerator
ray stop
ray start --head --num-gpus=1

poetry run python3 -m openrlhf.cli.train_grpo \
   --ray_num_nodes 1 \
   --make_block_explicit_pool \
   --user_grpc_server_address "localhost:50051" \
   --pretrain "Qwen/Qwen2.5-7B-Instruct" \
   --save_path "./checkpoint/rlef-7b-grpo" \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --max_samples 750 \
   --max_len 2048 \
   --bf16 \
   --actor_num_nodes 1 \
   --actor_per_node_gpus 1 \
   --vllm_num_engines 1 \
   --vllm_per_engine_gpus 1 \
   --vllm_gpu_memory_utilization 0.75 \
   --num_episodes 1 \
   --rollout_batch_size 256 \
   --grant_relative_group_size 8 \
   --generate_max_len 1024 \
   --learning_rate 1e-6 \
   --gradient_checkpointing \
   --wandb_project "rlef-grpo-single-h200"
