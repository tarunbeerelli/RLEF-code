import grpc
import torch
from rlef import reward_pb2, reward_pb2_grpc


def reward_func(queries, prompts, labels):
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = reward_pb2_grpc.RewardServiceStub(channel)

        # Build payload packing text along with its corresponding test case label metadata
        samples = [
            reward_pb2.TextSample(prompt=p, completion=q[len(p) :], metadata_json=lbl)
            for p, q, lbl in zip(prompts, queries, labels)
        ]

        request = reward_pb2.RewardRequest(samples=samples)
        response = stub.EvaluateBatch(request)

    reward_tensor = torch.tensor(response.rewards, dtype=torch.float32)
    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {"grpc_mean_reward": float(reward_tensor.mean().item())},
    }
