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
            ), f"Expected gpu_indexes {expected['gpu_indexes']}, but got {actual.gpu_indexes}"

        if "gpu_addresses" in expected:
            assert (
                actual.gpu_addresses == expected["gpu_addresses"]
            ), f"Expected gpu_addresses {expected['gpu_addresses']}, but got {actual.gpu_addresses}"

        if "vram" in expected:
            assert (
                actual.computed_resource_claim.vram == expected["vram"]
            ), f"Expected vram {expected['vram']}, but got {actual.computed_resource_claim.vram}"

        if "offload_layers" in expected:
            assert (
                actual.computed_resource_claim.offload_layers
                == expected["offload_layers"]
            ), f"Expected offload_layers {expected['offload_layers']}, but got {actual.computed_resource_claim.offload_layers}"

        if "worker_id" in expected:
            assert (
                actual.worker.id == expected["worker_id"]
            ), f"Expected worker_id {expected['worker_id']}, but got {actual.worker.id}"

        if "worker_name" in expected:
            assert (
                actual.worker.name == expected["worker_name"]
            ), f"Expected worker_name {expected['worker_name']}, but got {actual.worker.name}"

        if "is_unified_memory" in expected:
            assert (
                actual.computed_resource_claim.is_unified_memory
                == expected["is_unified_memory"]
            ), f"Expected is_unified_memory {expected['is_unified_memory']}, but got {actual.computed_resource_claim.is_unified_memory}"

        if "ram" in expected:
            assert (
                actual.computed_resource_claim.ram == expected["ram"]
            ), f"Expected ram {expected['ram']}, but got {actual.computed_resource_claim.ram}"

        if "score" in expected:
            assert actual.score is not None

            def truncate_2(value: float) -> float:
                return int(value * 100) / 100.0

            assert truncate_2(actual.score) == truncate_2(
                expected["score"]
            ), f"Expected score {expected['score']}, but got {actual.score}"

        if "subordinate_workers" in expected:
            for j, expected_subworker in enumerate(expected["subordinate_workers"]):
                actual_subworker = actual.subordinate_workers[j]

                assert (
                    actual_subworker.worker_id == expected_subworker.worker_id
                ), f"Expected subordinate worker worker_id {expected_subworker.worker_id}, but got {actual_subworker.worker_id}"
                assert (
                    actual_subworker.worker_ip == expected_subworker.worker_ip
                ), f"Expected subordinate worker worker_ip {expected_subworker.worker_ip}, but got {actual_subworker.worker_ip}"
                assert (
                    actual_subworker.total_gpus == expected_subworker.total_gpus
                ), f"Expected subordinate worker total_gpus {expected_subworker.total_gpus}, but got {actual_subworker.total_gpus}"
                assert (
                    actual_subworker.gpu_indexes == expected_subworker.gpu_indexes
                ), f"Expected subordinate worker gpu_indexes {expected_subworker.gpu_indexes}, but got {actual_subworker.gpu_indexes}"
                assert (
                    actual_subworker.gpu_addresses == expected_subworker.gpu_addresses
                ), f"Expected subordinate worker gpu_addresses {expected_subworker.gpu_addresses}, but got {actual_subworker.gpu_addresses}"
                assert (
                    actual_subworker.computed_resource_claim
                    == expected_subworker.computed_resource_claim
                ), f"Expected subordinate worker computed_resource_claim {expected_subworker.computed_resource_claim}, but got {actual_subworker.computed_resource_claim}"

        if "tensor_split" in expected:
            assert (
                actual.computed_resource_claim.tensor_split == expected["tensor_split"]
            ), f"Expected tensor_split {expected['tensor_split']}, but got {actual.computed_resource_claim.tensor_split}"
