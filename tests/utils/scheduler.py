from typing import List, Dict
from gpustack.policies.base import ModelInstanceScheduleCandidate


def compare_candidates(  # noqa: C901
    candidates: List[ModelInstanceScheduleCandidate], expected_candidates: List[Dict]
):
    assert len(candidates) == len(
        expected_candidates
    ), f"Expected {len(expected_candidates)}, but got {len(candidates)}"

    for i, expected in enumerate(expected_candidates):
        actual = candidates[i]

        if "gpu_indexes" in expected:
            assert (
                actual.gpu_indexes == expected["gpu_indexes"]
            ), f"Expected {expected['gpu_indexes']}, but got {actual.gpu_indexes}"

        if "gpu_addresses" in expected:
            assert (
                actual.gpu_addresses == expected["gpu_addresses"]
            ), f"Expected {expected['gpu_addresses']}, but got {actual.gpu_addresses}"

        if "vram" in expected:
            assert (
                actual.computed_resource_claim.vram == expected["vram"]
            ), f"Expected {expected['vram']}, but got {actual.computed_resource_claim.vram}"

        if "offload_layers" in expected:
            assert (
                actual.computed_resource_claim.offload_layers
                == expected["offload_layers"]
            ), f"Expected {expected['offload_layers']}, but got {actual.computed_resource_claim.offload_layers}"

        if "worker_id" in expected:
            assert (
                actual.worker.id == expected["worker_id"]
            ), f"Expected {expected['worker_id']}, but got {actual.worker.id}"

        if "worker_name" in expected:
            assert (
                actual.worker.name == expected["worker_name"]
            ), f"Expected {expected['worker_name']}, but got {actual.worker.name}"

        if "is_unified_memory" in expected:
            assert (
                actual.computed_resource_claim.is_unified_memory
                == expected["is_unified_memory"]
            ), f"Expected {expected['is_unified_memory']}, but got {actual.computed_resource_claim.is_unified_memory}"

        if "ram" in expected:
            assert (
                actual.computed_resource_claim.ram == expected["ram"]
            ), f"Expected {expected['ram']}, but got {actual.computed_resource_claim.ram}"

        if "score" in expected:
            assert (
                str(actual.score)[:5] == str(expected["score"])[:5]
            ), f"Expected {expected['score']}, but got {actual.score}"

        if "subordinate_workers" in expected:
            for j, expected_subworker in enumerate(expected["subordinate_workers"]):
                actual_subworker = actual.subordinate_workers[j]

                assert (
                    actual_subworker.worker_id == expected_subworker.worker_id
                ), f"Expected {expected_subworker.worker_id}, but got {actual_subworker.worker_id}"
                assert (
                    actual_subworker.worker_ip == expected_subworker.worker_ip
                ), f"Expected {expected_subworker.worker_ip}, but got {actual_subworker.worker_ip}"
                assert (
                    actual_subworker.total_gpus == expected_subworker.total_gpus
                ), f"Expected {expected_subworker.total_gpus}, but got {actual_subworker.total_gpus}"
                assert (
                    actual_subworker.gpu_indexes == expected_subworker.gpu_indexes
                ), f"Expected {expected_subworker.gpu_indexes}, but got {actual_subworker.gpu_indexes}"
                assert (
                    actual_subworker.gpu_addresses == expected_subworker.gpu_addresses
                ), f"Expected {expected_subworker.gpu_addresses}, but got {actual_subworker.gpu_addresses}"
                assert (
                    actual_subworker.computed_resource_claim
                    == expected_subworker.computed_resource_claim
                ), f"Expected {expected_subworker.computed_resource_claim}, but got {actual_subworker.computed_resource_claim}"

        if "rpc_servers" in expected:
            for j, expected_rpc_server in enumerate(expected["rpc_servers"]):
                # FIXME: I think this is should be actual.rpc_servers[j]
                actual_rpc_server = expected_rpc_server

                assert (
                    actual_rpc_server.worker_id == expected_rpc_server.worker_id
                ), f"Expected {expected_rpc_server.worker_id}, but got {actual_rpc_server.worker_id}"
                assert (
                    actual_rpc_server.gpu_index == expected_rpc_server.gpu_index
                ), f"Expected {expected_rpc_server.gpu_index}, but got {actual_rpc_server.gpu_index}"
                assert (
                    actual_rpc_server.computed_resource_claim
                    == expected_rpc_server.computed_resource_claim
                ), f"Expected {expected_rpc_server.computed_resource_claim}, but got {actual_rpc_server.computed_resource_claim}"

        if "ray_actors" in expected:
            for j, expected_ray_actor in enumerate(expected["ray_actors"]):
                # FIXME: I think this is should be actual.ray_actors[j]
                actual_ray_actor = expected_ray_actor

                assert (
                    actual_ray_actor.worker_id == expected_ray_actor.worker_id
                ), f"Expected {expected_ray_actor.worker_id}, but got {actual_ray_actor.worker_id}"
                assert (
                    actual_ray_actor.worker_ip == expected_ray_actor.worker_ip
                ), f"Expected {expected_ray_actor.worker_ip}, but got {actual_ray_actor.worker_ip}"
                assert (
                    actual_ray_actor.total_gpus == expected_ray_actor.total_gpus
                ), f"Expected {expected_ray_actor.total_gpus}, but got {actual_ray_actor.total_gpus}"
                assert (
                    actual_ray_actor.gpu_indexes == expected_ray_actor.gpu_indexes
                ), f"Expected {expected_ray_actor.gpu_indexes}, but got {actual_ray_actor.gpu_indexes}"
                assert (
                    actual_ray_actor.computed_resource_claim
                    == expected_ray_actor.computed_resource_claim
                ), f"Expected {expected_ray_actor.computed_resource_claim}, but got {actual_ray_actor.computed_resource_claim}"

        if "tensor_split" in expected:
            assert (
                actual.computed_resource_claim.tensor_split == expected["tensor_split"]
            ), f"Expected {expected['tensor_split']}, but got {actual.computed_resource_claim.tensor_split}"
