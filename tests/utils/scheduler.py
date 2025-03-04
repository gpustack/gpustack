from typing import List
from gpustack.policies.base import ModelInstanceScheduleCandidate


def compare_candidates(  # noqa: C901
    candidates: List[ModelInstanceScheduleCandidate], expected_candidates
):
    for i, expected in enumerate(expected_candidates):
        candidate = candidates[i]
        if "gpu_indexes" in expected:
            assert candidate.gpu_indexes == expected["gpu_indexes"]

        if "vram" in expected:
            assert candidate.computed_resource_claim.vram == expected["vram"]

        if "offload_layers" in expected:
            assert (
                candidate.computed_resource_claim.offload_layers
                == expected["offload_layers"]
            )

        if "worker_id" in expected:
            assert candidate.worker.id == expected["worker_id"]

        if "worker_name" in expected:
            assert candidate.worker.name == expected["worker_name"]

        if "is_unified_memory" in expected:
            assert (
                candidate.computed_resource_claim.is_unified_memory
                == expected["is_unified_memory"]
            )

        if "ram" in expected:
            assert candidate.computed_resource_claim.ram == expected["ram"]

        if "score" in expected:
            assert str(candidate.score)[:5] == str(expected["score"])[:5]

        if "rpc_servers" in expected:
            for i, rpc_server in enumerate(expected["rpc_servers"]):
                assert rpc_server.worker_id == expected["rpc_servers"][i].worker_id
                assert rpc_server.gpu_index == expected["rpc_servers"][i].gpu_index
                assert (
                    rpc_server.computed_resource_claim
                    == expected["rpc_servers"][i].computed_resource_claim
                )

        if "ray_actors" in expected:
            for i, ray_actor in enumerate(expected["ray_actors"]):
                assert ray_actor.worker_id == expected["ray_actors"][i].worker_id
                assert ray_actor.worker_ip == expected["ray_actors"][i].worker_ip
                assert ray_actor.total_gpus == expected["ray_actors"][i].total_gpus
                assert ray_actor.gpu_indexes == expected["ray_actors"][i].gpu_indexes
                assert (
                    ray_actor.computed_resource_claim
                    == expected["ray_actors"][i].computed_resource_claim
                )

        if "tensor_split" in expected:
            assert (
                candidate.computed_resource_claim.tensor_split
                == expected["tensor_split"]
            )
