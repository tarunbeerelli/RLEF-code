#!/bin/bash
set -e

# Bind local project search paths
export PYTHONPATH=src:.:$PYTHONPATH

# Default system configurations (Optimized for 1*H200)
GPUS_AVAILABLE=1
CHECKPOINT_PATH="./checkpoint/rlef-7b-grpo"
EVAL_DATASET="./data/eval_prompts.jsonl"
FILTERED_DATASET="./data/eval_prompts_balanced_500.jsonl"
OUTPUT_PATH="./results/eval_outputs.jsonl"

# Check if upgrading to the 2*H200 stack later
if [ "$1" == "--dual" ]; then
    echo "🚀 Scaling evaluation worker profile to 2*H200 NVLink stack..."
    GPUS_AVAILABLE=2
else
    echo "⚡ Initializing evaluation worker profile on single local H200..."
fi

# Ensure output directories exist structurally
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$FILTERED_DATASET")"

# Step 1: Pre-process and slice exactly 500 balanced test samples (150 intro, 250 interview, 100 comp)
echo "🧹 Extracting optimized 500-case evaluation subset from master dataset..."
poetry run python3 -c "
import json

TARGETS = {
    'introductory': 150,
    'interview': 250,
    'competition': 100
}
counts = {'introductory': 0, 'interview': 0, 'competition': 0}

with open('${EVAL_DATASET}', 'r') as infile, open('${FILTERED_DATASET}', 'w') as outfile:
    for line in infile:
        if not line.strip(): continue
        data = json.loads(line)
        diff = str(data.get('difficulty', '')).lower()

        if diff in TARGETS and counts[diff] < TARGETS[diff]:
            outfile.write(json.dumps(data) + '\n')
            counts[diff] += 1

print(f'📊 Strategy breakdown compiled: {counts}')
total_cases = sum(counts.values())
print(f'✅ Dynamic test slice created with exactly {total_cases} questions.')
"

# Safe initialization of Ray runtime pools matching active silicon
ray stop
ray start --head --num-gpus=${GPUS_AVAILABLE}

# Step 2: Fire up OpenRLHF batch generation over our extracted 500 cases
echo "🔮 Running high-throughput parallel generation via OpenRLHF vLLM engine..."
poetry run python3 -m openrlhf.cli.batch_inference \
   --eval_task generate \
   --pretrain "${CHECKPOINT_PATH}" \
   --dataset json@$(dirname "${FILTERED_DATASET}") \
   --input_key "input" \
   --output_path "${OUTPUT_PATH}" \
   --max_len 2048 \
   --flash_attn \
   --bf16 \
   --micro_batch_size 16 \
   --tp_size ${GPUS_AVAILABLE} \
   --vllm_gpu_memory_utilization 0.85 \
   --best_of 1 \
   --generate_max_len 1024 \
   --temperature 0.0

echo "✅ Evaluation inference generation complete. Outputs saved to: ${OUTPUT_PATH}"

# Step 3: Run aggregate validation analysis through your gRPC Reward Servicer
read -p "Do you want to route these completions through the gRPC reward server for full metric analytics? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📡 Scoring completions against active gRPC reward server rules..."
    PYTHONPATH=. poetry run python -c "
import json, asyncio, grpc
from collections import defaultdict
from rlef import reward_pb2, reward_pb2_grpc

async def score_eval():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = reward_pb2_grpc.RewardServiceStub(channel)
        samples = []
        difficulties = []

        with open('${OUTPUT_PATH}', 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                diff = str(data.get('difficulty', 'interview')).lower()
                difficulties.append(diff)
                samples.append(reward_pb2.RewardRequest.Sample(
                    prompt=data.get('input', ''),
                    completion=data.get('output', ''),
                    metadata_json=json.dumps({'difficulty': diff})
                ))

        if not samples:
            print('⚠️ No evaluations found to process.')
            return

        request = reward_pb2.RewardResponse(samples=samples)
        response = await stub.EvaluateBatch(request)

        # Calculate full matrix stratified performance splits
        tier_scores = defaultdict(list)
        for score, diff in zip(response.rewards, difficulties):
            tier_scores[diff].append(score)

        print('\n================== EVALUATION ANALYSIS MATRIX ==================')
        total_score = 0
        for tier, scores in tier_scores.items():
            avg_tier = sum(scores) / len(scores)
            total_score += sum(scores)
            print(f'   📈 Tier: {tier.upper():12} | Count: {len(scores):3} | Mean Reward: {avg_tier:.4f}')

        print('----------------------------------------------------------------')
        print(f'   🏆 AGGREGATE EVAL MEAN: {total_score / len(response.rewards):.4f}')
        print('================================================================\n')

asyncio.run(score_eval())
"
fi
