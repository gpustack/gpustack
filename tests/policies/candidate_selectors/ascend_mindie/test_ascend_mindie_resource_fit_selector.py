import pytest
from unittest.mock import patch

from tests.utils.model import new_model
from gpustack.policies.candidate_selectors import AscendMindIEResourceFitSelector
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
    GPUSelector,
)
from tests.fixtures.workers.fixtures import (
    linux_ascend_1_910b_64gx8,
    linux_ascend_2_910b_64gx8,
    linux_ascend_3_910b_64gx8,
    linux_ascend_4_910b_64gx8,
)
from tests.utils.scheduler import compare_candidates


def expected_candidate(
    worker_id, worker_name, gpu_indexes, vram, subworkers=None, ram=None
):
    candidate = {
        "worker_id": worker_id,
        "worker_name": worker_name,
        "gpu_indexes": gpu_indexes,
        "is_unified_memory": False,
        "vram": vram,
        "subordinate_workers": subworkers or [],
    }
    if ram is not None:
        candidate["ram"] = ram
    return candidate


@pytest.mark.parametrize(
    "m, expected",
    [
        # Manual single worker selection.
        # Check point:
        # - Unavailable workers.
        (
            new_model(
                id=1,
                name="manual_single_worker_selection",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_2:cann:0",
                        "ascend_4:cann:0",  # Unavailable worker.
                    ],
                    gpus_per_replica=2,
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [],
        ),
        # Manual single worker selection 2.
        # Check point:
        # - Unavailable devices.
        (
            new_model(
                id=1,
                name="manual_single_worker_selection_2",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_2:cann:0",
                        "ascend_3:cann:0",  # Unavailable device.
                    ],
                    gpus_per_replica=2,
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [],
        ),
        # Automatic single worker selection.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - There are two candidates be selected.
        (
            new_model(
                id=1,
                name="automatic_single_worker_selection",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.39"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.41"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
            ],
        ),
        # Semi-automatic multi-workers selection.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Specify tensor parallel size to enforce the selection of multi-devices.
        # - There are two candidates be selected.
        (
            new_model(
                id=1,
                name="semi_automatic_multi_workers_selection",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--tensor-parallel-size=2",
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.39"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.41"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 54975581388},
                            ),
                        )
                    ],
                }
            ],
        ),
        # Semi-automatic multi-workers selection 2.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Specify tensor parallel size to exceeds all available devices count.
        # - There are no candidates be selected.
        (
            new_model(
                id=1,
                name="semi_automatic_multi_workers_selection_2",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--tensor-parallel-size=4",
                    "--trust-remote-code",
                ],
            ),
            [],
        ),
        # Semi-automatic multi-workers selection 3.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   specify NPU memory fraction to satisfy the selection.
        # - Specify pipeline parallel size to select available devices count.
        (
            new_model(
                id=1,
                name="semi_automatic_multi_workers_selection_3",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--npu-memory-fraction=0.5",
                    "--pipeline-parallel-size=3",
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.39"],
                    "ram": 536870912,
                    "vram": {0: 34359738368},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.41"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 34359738368},
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=1,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.42"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 34359738368},
                            ),
                        ),
                    ],
                }
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_3x_64gx1_1x_64gx0(config, m, expected):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 2nd worker to 40%.
        if worker.id == 2:
            for dev in worker.status.gpu_devices:
                dev.memory.utilization_rate = 40.0
                dev.memory.used = 24758800785
                dev.memory.allocated = 21474836480

    workers = [
        linux_ascend_1_910b_64gx8(return_device=1),
        linux_ascend_2_910b_64gx8(return_device=1, callback=adjust_memory),
        linux_ascend_3_910b_64gx8(return_device=1),
        linux_ascend_4_910b_64gx8(return_device=0),  # No devices.
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected)


@pytest.mark.parametrize(
    "m, expected",
    [
        # Automatic multi-workers selection.
        # Check point:
        # - Although 1st and 2nd workers have total 8 devices,
        #   [0, 2] devices on 1st worker have allocated 60%,
        #   [0, 2] devices on 2nd worker have allocated 40%,
        #   Therefore, 4 workers should be selected to provide 8 devices.
        (
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.5",
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0, 1],
                    "gpu_addresses": ["29.17.48.41", "29.17.57.32"],
                    "ram": 536870912,
                    "vram": {0: 34359738368, 1: 34359738368},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=4,
                            worker_ip="192.168.50.6",
                            total_gpus=2,
                            gpu_indexes=[0, 1],
                            gpu_addresses=["29.18.48.41", "29.18.57.33"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 34359738368, 1: 34359738368},
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=1,
                            worker_ip="192.168.50.3",
                            total_gpus=4,
                            gpu_indexes=[1, 3],
                            gpu_addresses=["29.17.57.31", "29.17.48.40"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={1: 34359738368, 3: 34359738368},
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=4,
                            gpu_indexes=[1, 3],
                            gpu_addresses=["29.17.57.33", "29.17.48.42"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={1: 34359738368, 3: 34359738368},
                            ),
                        ),
                    ],
                },
            ],
        )
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_2x_64gx4_2x_64gx2(config, m, expected):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 1st and 2nd workers.
        # The 0th and 2nd devices of 1st worker have allocated 60%,
        # while the 0th and 2nd devices of 2nd worker have allocated 40%.
        if worker.id == 1:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 2]:
                    dev.memory.utilization_rate = 60.0
                    dev.memory.used = 44667659879
                    dev.memory.allocated = 41231686042
        elif worker.id == 2:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 2]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695

    workers = [
        linux_ascend_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_3_910b_64gx8(return_device=2),
        linux_ascend_4_910b_64gx8(return_device=2),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)
    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected)


@pytest.mark.parametrize(
    "m, expected",
    [
        # Automatic single worker selection.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - There are two candidates be selected
        (
            new_model(
                id=1,
                name="automatic_single_worker_selection",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.39"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.41"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
            ],
        ),
        # Semi-automatic multi-workers selection.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Specify tensor parallel size to enforce the selection of multi-devices.
        (
            new_model(
                id=1,
                name="semi_automatic_multi_workers_selection",
                replicas=1,
                model_scope_model_id="Qwen/Qwen3-4B",
                cpu_offloading=False,
                backend_parameters=[
                    "--tensor-parallel-size=4",
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0, 1],
                    "gpu_addresses": ["29.17.48.39", "29.17.57.31"],
                    "ram": 536870912,
                    "vram": {0: 54975581388, 1: 54975581388},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=2,
                            gpu_indexes=[0, 1],
                            gpu_addresses=["29.17.48.41", "29.17.57.32"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 54975581388, 1: 54975581388},
                            ),
                        )
                    ],
                }
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_3x_64gx2(config, m, expected):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 2nd worker to 40%.
        if worker.id == 2:
            for dev in worker.status.gpu_devices:
                dev.memory.utilization_rate = 40.0
                dev.memory.used = 24758800785
                dev.memory.allocated = 21474836480

    workers = [
        linux_ascend_1_910b_64gx8(return_device=2),
        linux_ascend_2_910b_64gx8(return_device=2, callback=adjust_memory),
        linux_ascend_3_910b_64gx8(return_device=2),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected)


@pytest.mark.parametrize(
    "m, expected",
    [
        # Manual single worker selection.
        # Check point:
        # - Specify GPU selector to enforce the selection of single worker.
        (
            new_model(
                id=1,
                name="manual_single_worker_selection",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-8B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_2:cann:0",
                    ],
                    gpus_per_replica=1,
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.41"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
            ],
        ),
        # Manual single worker selection.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   specify NPU memory fraction to satisfy the selection.
        (
            new_model(
                id=1,
                name="manual_single_worker_selection",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-14B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:1",
                    ],
                    gpus_per_replica=2,
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 0],
                    "gpu_addresses": ["29.17.57.31", "29.17.48.39"],
                    "ram": 536870912,
                    "vram": {1: 54975581388, 0: 54975581388},
                },
            ],
        ),
        # Automatic single worker selection.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   it has sorted to the list end.
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - There are two candidates be selected.
        (
            new_model(
                id=1,
                name="automatic_single_worker_selection",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-8B",
                cpu_offloading=False,
                backend_parameters=[
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1],
                    "gpu_addresses": ["29.17.57.31"],
                    "ram": 536870912,
                    "vram": {1: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.41"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                },
            ],
        ),
        # Semi-automatic single worker selection.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   it has sorted to the list end.
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Specify tensor parallel size to enforce the selection of multi-devices.
        # - There are two candidates be selected.
        (
            new_model(
                id=1,
                name="semi_automatic_single_worker_selection",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-14B",
                cpu_offloading=False,
                backend_parameters=[
                    "--tensor-parallel-size=2",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 54975581388, 2: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0, 1],
                    "gpu_addresses": ["29.17.48.41", "29.17.57.32"],
                    "ram": 536870912,
                    "vram": {0: 54975581388, 1: 54975581388},
                },
            ],
        ),
        # Semi-automatic single worker selection 2.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   it has sorted to the list end.
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Specify data parallel size to enforce the selection of multi-devices.
        # - There are two candidates be selected.
        (
            new_model(
                id=1,
                name="semi_automatic_single_worker_selection_2",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-14B",
                cpu_offloading=False,
                backend_parameters=[
                    "--data-parallel-size=2",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 54975581388, 2: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0, 1],
                    "gpu_addresses": ["29.17.48.41", "29.17.57.32"],
                    "ram": 536870912,
                    "vram": {0: 54975581388, 1: 54975581388},
                },
            ],
        ),
        # Manual multi-workers selection.
        # Check point:
        # - Specify GPU selector to enforce the selection of multi-workers with multiple devices.
        (
            new_model(
                id=1,
                name="manual_multi_workers_selection",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-32B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:1",
                        "ascend_0:cann:2",
                        "ascend_2:cann:3",
                        "ascend_2:cann:4",
                    ],
                    gpus_per_replica=4,
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 54975581388, 2: 54975581388},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=2,
                            gpu_indexes=[3, 4],
                            gpu_addresses=["29.17.49.41", "29.17.45.216"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={3: 54975581388, 4: 54975581388},
                                vram_utilization=0.8,
                            ),
                        ),
                    ],
                }
            ],
        ),
        # Manual multi-workers selection 2.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   specify NPU memory fraction to satisfy the selection.
        (
            new_model(
                id=1,
                name="manual_multi_workers_selection_2",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-14B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_2:cann:0",
                    ],
                    gpus_per_replica=2,
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.39"],
                    "ram": 536870912,
                    "vram": {0: 54975581388},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.41"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 54975581388},
                                vram_utilization=0.8,
                            ),
                        ),
                    ],
                },
            ],
        ),
        # Manual multi-workers selection with gpus per replica 2.
        # Check point:
        # - Specify GPU selector to enforce the selection of multi-workers with multiple devices.
        (
            new_model(
                id=1,
                name="manual_multi_workers_select_with_gpus_per_replica_2",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen3-32B",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:1",
                        "ascend_0:cann:2",
                        "ascend_2:cann:3",
                        "ascend_2:cann:4",
                    ],
                    gpus_per_replica=2,
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 54975581388, 2: 54975581388},
                },
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [3, 4],
                    "gpu_addresses": ["29.17.49.41", "29.17.45.216"],
                    "ram": 536870912,
                    "vram": {3: 54975581388, 4: 54975581388},
                },
            ],
        ),
        # Automatic multi-workers selection.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   specify NPU memory fraction to satisfy the selection.
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        (
            new_model(
                id=1,
                name="automatic_multi_workers_selection",
                replicas=1,
                huggingface_repo_id="deepseek-ai/DeepSeek-V2-Chat",
                cpu_offloading=False,
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2, 3, 4, 5, 6, 7, 0],
                    "gpu_addresses": [
                        "29.17.57.31",
                        "29.17.51.57",
                        "29.17.48.40",
                        "29.17.45.215",
                        "29.17.67.76",
                        "29.17.114.31",
                        "29.17.105.70",
                        "29.17.48.39",
                    ],
                    "ram": 536870912,
                    "vram": {
                        0: 54975581388,
                        1: 54975581388,
                        2: 54975581388,
                        3: 54975581388,
                        4: 54975581388,
                        5: 54975581388,
                        6: 54975581388,
                        7: 54975581388,
                    },
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                "29.17.48.41",
                                "29.17.57.32",
                                "29.17.51.78",
                                "29.17.49.41",
                                "29.17.45.216",
                                "29.17.67.77",
                                "29.17.114.32",
                                "29.17.105.71",
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 54975581388,
                                    1: 54975581388,
                                    2: 54975581388,
                                    3: 54975581388,
                                    4: 54975581388,
                                    5: 54975581388,
                                    6: 54975581388,
                                    7: 54975581388,
                                },
                            ),
                        )
                    ],
                }
            ],
        ),
        # Automatic multi-workers selection 2.
        # Check point:
        # - All devices of 2nd worker have allocated 40%,
        #   can not satisfy the selection even indicates NPU memory fraction.
        # - There are 3 workers available,
        #   which cannot satisfy the attention head size.
        (
            new_model(
                id=1,
                name="automatic_multi_workers_selection_2",
                replicas=1,
                huggingface_repo_id="deepseek-ai/DeepSeek-V2-Chat",
                cpu_offloading=False,
                backend_parameters=[
                    "--npu-memory-fraction=0.6",
                    "--trust-remote-code",
                ],
            ),
            [],
        ),
        # Automatic multi-workers selection 3.
        # Check point:
        # - The 0th device of 1st worker has allocated 15%,
        #   specify NPU memory fraction to satisfy the selection.
        # - All devices of 2nd worker have allocated 40%,
        #   it should not be selected.
        # - Quantization support.
        (
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                model_scope_model_id="vllm-ascend/DeepSeek-V3-W8A8",
                cpu_offloading=False,
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2, 3, 4, 5, 6, 7, 0],
                    "gpu_addresses": [
                        "29.17.57.31",
                        "29.17.51.57",
                        "29.17.48.40",
                        "29.17.45.215",
                        "29.17.67.76",
                        "29.17.114.31",
                        "29.17.105.70",
                        "29.17.48.39",
                    ],
                    "ram": 536870912,
                    "vram": {
                        0: 54975581388,
                        1: 54975581388,
                        2: 54975581388,
                        3: 54975581388,
                        4: 54975581388,
                        5: 54975581388,
                        6: 54975581388,
                        7: 54975581388,
                    },
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                "29.17.48.41",
                                "29.17.57.32",
                                "29.17.51.78",
                                "29.17.49.41",
                                "29.17.45.216",
                                "29.17.67.77",
                                "29.17.114.32",
                                "29.17.105.71",
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 54975581388,
                                    1: 54975581388,
                                    2: 54975581388,
                                    3: 54975581388,
                                    4: 54975581388,
                                    5: 54975581388,
                                    6: 54975581388,
                                    7: 54975581388,
                                },
                            ),
                        )
                    ],
                }
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_3x_64gx8(config, m, expected):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 1st and 2nd workers.
        # The 0th device of 1st worker has allocated 15%,
        # while all devices of 2nd worker have allocated 40%.
        if worker.id == 1:
            worker.status.gpu_devices[0].memory.utilization_rate = 15.0
            worker.status.gpu_devices[0].memory.used = 13743895347
            worker.status.gpu_devices[0].memory.allocated = 10307921510
        elif worker.id == 2:
            for dev in worker.status.gpu_devices:
                dev.memory.utilization_rate = 40.0
                dev.memory.used = 24758800785
                dev.memory.allocated = 21474836480

    workers = [
        linux_ascend_1_910b_64gx8(callback=adjust_memory),
        linux_ascend_2_910b_64gx8(callback=adjust_memory),
        linux_ascend_3_910b_64gx8(),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected)


@pytest.mark.parametrize(
    "m, expected",
    [
        # Manual multi-workers selection.
        (
            new_model(
                id=1,
                name="manual_multi_workers_selection",
                replicas=1,
                model_scope_model_id="deepseek-ai/DeepSeek-R1-0528",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:1",
                        "ascend_0:cann:2",
                        "ascend_0:cann:3",
                        "ascend_0:cann:4",
                        "ascend_0:cann:5",
                        "ascend_0:cann:6",
                        "ascend_0:cann:7",
                        "ascend_1:cann:0",
                        "ascend_1:cann:1",
                        "ascend_1:cann:2",
                        "ascend_1:cann:3",
                        "ascend_1:cann:4",
                        "ascend_1:cann:5",
                        "ascend_1:cann:6",
                        "ascend_1:cann:7",
                        "ascend_2:cann:0",
                        "ascend_2:cann:1",
                        "ascend_2:cann:2",
                        "ascend_2:cann:3",
                        "ascend_2:cann:4",
                        "ascend_2:cann:5",
                        "ascend_2:cann:6",
                        "ascend_2:cann:7",
                        "ascend_3:cann:0",
                        "ascend_3:cann:1",
                        "ascend_3:cann:2",
                        "ascend_3:cann:3",
                        "ascend_3:cann:4",
                        "ascend_3:cann:5",
                        "ascend_3:cann:6",
                        "ascend_3:cann:7",
                    ],
                    gpus_per_replica=32,
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.95",
                    "--data-parallel-size=4",
                    "--tensor-parallel-size=8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                    "gpu_addresses": [
                        '29.17.48.39',
                        '29.17.57.31',
                        '29.17.51.57',
                        '29.17.48.40',
                        '29.17.45.215',
                        '29.17.67.76',
                        '29.17.114.31',
                        '29.17.105.70',
                    ],
                    "ram": 536870912,
                    "vram": {
                        0: 65283502899,
                        1: 65283502899,
                        2: 65283502899,
                        3: 65283502899,
                        4: 65283502899,
                        5: 65283502899,
                        6: 65283502899,
                        7: 65283502899,
                    },
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.42',
                                '29.17.57.33',
                                '29.17.51.79',
                                '29.17.48.42',
                                '29.17.45.217',
                                '29.17.67.78',
                                '29.17.114.33',
                                '29.17.105.72',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                                vram_utilization=0.95,
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.41',
                                '29.17.57.32',
                                '29.17.51.78',
                                '29.17.49.41',
                                '29.17.45.216',
                                '29.17.67.77',
                                '29.17.114.32',
                                '29.17.105.71',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                                vram_utilization=0.95,
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=4,
                            worker_ip="192.168.50.6",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.18.48.41',
                                '29.18.57.33',
                                '29.18.51.78',
                                '29.18.48.41',
                                '29.18.45.216',
                                '29.18.67.77',
                                '29.18.114.32',
                                '29.18.105.71',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                                vram_utilization=0.95,
                            ),
                        ),
                    ],
                }
            ],
        ),
        # Manual multi-workers selection with 16 gpus per replica.
        (
            new_model(
                id=1,
                name="manual_multi_workers_selection_with_16_gpus_per_replica",
                replicas=1,
                model_scope_model_id="deepseek-ai/DeepSeek-R1-0528",
                cpu_offloading=False,
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:1",
                        "ascend_0:cann:2",
                        "ascend_0:cann:3",
                        "ascend_0:cann:4",
                        "ascend_0:cann:5",
                        "ascend_0:cann:6",
                        "ascend_0:cann:7",
                        "ascend_1:cann:0",
                        "ascend_1:cann:1",
                        "ascend_1:cann:2",
                        "ascend_1:cann:3",
                        "ascend_1:cann:4",
                        "ascend_1:cann:5",
                        "ascend_1:cann:6",
                        "ascend_1:cann:7",
                        "ascend_2:cann:0",
                        "ascend_2:cann:1",
                        "ascend_2:cann:2",
                        "ascend_2:cann:3",
                        "ascend_2:cann:4",
                        "ascend_2:cann:5",
                        "ascend_2:cann:6",
                        "ascend_2:cann:7",
                        "ascend_3:cann:0",
                        "ascend_3:cann:1",
                        "ascend_3:cann:2",
                        "ascend_3:cann:3",
                        "ascend_3:cann:4",
                        "ascend_3:cann:5",
                        "ascend_3:cann:6",
                        "ascend_3:cann:7",
                    ],
                    gpus_per_replica=32,
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.95",
                    "--data-parallel-size=4",
                    "--tensor-parallel-size=8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                    "gpu_addresses": [
                        '29.17.48.39',
                        '29.17.57.31',
                        '29.17.51.57',
                        '29.17.48.40',
                        '29.17.45.215',
                        '29.17.67.76',
                        '29.17.114.31',
                        '29.17.105.70',
                    ],
                    "ram": 536870912,
                    "vram": {
                        0: 65283502899,
                        1: 65283502899,
                        2: 65283502899,
                        3: 65283502899,
                        4: 65283502899,
                        5: 65283502899,
                        6: 65283502899,
                        7: 65283502899,
                    },
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.42',
                                '29.17.57.33',
                                '29.17.51.79',
                                '29.17.48.42',
                                '29.17.45.217',
                                '29.17.67.78',
                                '29.17.114.33',
                                '29.17.105.72',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                                vram_utilization=0.95,
                            ),
                        ),
                    ],
                }
            ],
        ),
        # Automatic multi-workers selection.
        (
            new_model(
                id=1,
                name="automatic_multi_workers_selection",
                replicas=1,
                model_scope_model_id="deepseek-ai/DeepSeek-R1-0528",
                cpu_offloading=False,
                backend_parameters=[
                    "--npu-memory-fraction=0.95",
                    "--data-parallel-size=4",
                    "--tensor-parallel-size=8",
                    "--trust-remote-code",
                ],
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
                    "gpu_addresses": [
                        '29.17.48.39',
                        '29.17.57.31',
                        '29.17.51.57',
                        '29.17.48.40',
                        '29.17.45.215',
                        '29.17.67.76',
                        '29.17.114.31',
                        '29.17.105.70',
                    ],
                    "ram": 536870912,
                    "vram": {
                        0: 65283502899,
                        1: 65283502899,
                        2: 65283502899,
                        3: 65283502899,
                        4: 65283502899,
                        5: 65283502899,
                        6: 65283502899,
                        7: 65283502899,
                    },
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.42',
                                '29.17.57.33',
                                '29.17.51.79',
                                '29.17.48.42',
                                '29.17.45.217',
                                '29.17.67.78',
                                '29.17.114.33',
                                '29.17.105.72',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.41',
                                '29.17.57.32',
                                '29.17.51.78',
                                '29.17.49.41',
                                '29.17.45.216',
                                '29.17.67.77',
                                '29.17.114.32',
                                '29.17.105.71',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                            ),
                        ),
                        ModelInstanceSubordinateWorker(
                            worker_id=4,
                            worker_ip="192.168.50.6",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.18.48.41',
                                '29.18.57.33',
                                '29.18.51.78',
                                '29.18.48.41',
                                '29.18.45.216',
                                '29.18.67.77',
                                '29.18.114.32',
                                '29.18.105.71',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={
                                    0: 65283502899,
                                    1: 65283502899,
                                    2: 65283502899,
                                    3: 65283502899,
                                    4: 65283502899,
                                    5: 65283502899,
                                    6: 65283502899,
                                    7: 65283502899,
                                },
                            ),
                        ),
                    ],
                }
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_4x_64gx8(config, m, expected):
    workers = [
        linux_ascend_1_910b_64gx8(),
        linux_ascend_2_910b_64gx8(),
        linux_ascend_3_910b_64gx8(),
        linux_ascend_4_910b_64gx8(),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected)


@pytest.mark.parametrize(
    "index, m, expect_msg",
    [
        (
            1,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.2",
                    "--trust-remote-code",
                ],
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.2, all GPUs combined need to provide at least 781.20 GiB of total VRAM and each GPU needs 20% of allocatable VRAM.
- The optimal combination ['ascend_0', 'ascend_1'] provides 102.4 GiB of allocatable VRAM.
- Cannot find a suitable worker combination to run the model in distributed mode. If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and devices."""
            ],
        ),
        (
            2,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.2",
                    "--trust-remote-code",
                ],
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.2, all GPUs combined need to provide at least 781.20 GiB of total VRAM and each GPU needs 20% of allocatable VRAM.
- The optimal combination ['ascend_2', 'ascend_3', 'ascend_1'] provides 76.8 GiB of allocatable VRAM. There are 1 worker that can provide 2 devices, as the workers in the combination, but some devices among them fail to meet requirements.
- Cannot find a suitable worker combination to run the model in distributed mode. If you are confident that the resources are sufficient, you may manually schedule the model by selecting the workers and devices."""
            ],
        ),
        (
            3,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.2",
                    "--trust-remote-code",
                ],
                distributed_inference_across_workers=False,
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.2, all GPUs combined need to provide at least 781.20 GiB of total VRAM and each GPU needs 20% of allocatable VRAM.
- The largest available worker has 51.2 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 10.24 GiB of allocatable VRAM."""
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_2x_64gx4_2x_64gx2_check_msg(
    config, index, m, expect_msg
):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 1st and 2nd workers.
        # The 0th and 2nd devices of 1st worker have allocated 60%,
        # while the 0th and 2nd devices of 2nd worker have allocated 40%.
        if worker.id == 1:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 2]:
                    dev.memory.utilization_rate = 60.0
                    dev.memory.used = 44667659879
                    dev.memory.allocated = 41231686042
        elif worker.id == 2:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 2]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695

    workers = [
        linux_ascend_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_3_910b_64gx8(return_device=2),
        linux_ascend_4_910b_64gx8(return_device=2),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    if index == 2:
        for device in workers[0].status.gpu_devices:
            device.type = "unknown"

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        await resource_fit_selector.select_candidates(workers)
        assert resource_fit_selector.get_messages() == expect_msg


@pytest.mark.parametrize(
    "index, m, expect_msg",
    [
        (
            1,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.9",
                    "--trust-remote-code",
                ],
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:2",
                        "ascend_1:cann:0",
                        "ascend_1:cann:2",
                        "ascend_2:cann:0",
                        "ascend_2:cann:2",
                        "ascend_3:cann:0",
                        "ascend_3:cann:2",
                    ],
                    gpus_per_replica=8,
                ),
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.9, all GPUs combined need to provide at least 173.60 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Selected GPUs have 307.20 GiB allocatable VRAM, 0/8 of GPUs meet the VRAM utilization ratio, providing 276.48 GiB of allocatable VRAM."""
            ],
        ),
        (
            2,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.1",
                    "--trust-remote-code",
                ],
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:2",
                        "ascend_1:cann:0",
                        "ascend_1:cann:2",
                        "ascend_2:cann:0",
                        "ascend_2:cann:2",
                        "ascend_3:cann:0",
                        "ascend_3:cann:2",
                    ],
                    gpus_per_replica=8,
                ),
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.1, all GPUs combined need to provide at least 1562.40 GiB of total VRAM and each GPU needs 10% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Selected GPUs have 307.20 GiB allocatable VRAM, 8/8 of GPUs meet the VRAM utilization ratio, providing 30.72 GiB of allocatable VRAM."""
            ],
        ),
        (
            3,
            new_model(
                id=1,
                name="automatic_multi_workers_selection_3",
                replicas=1,
                huggingface_repo_id="Qwen/Qwen2.5-72B-Instruct",
                cpu_offloading=False,
                backend_parameters=[
                    "--max-seq-len=32768",
                    "--npu-memory-fraction=0.9",
                    "--trust-remote-code",
                ],
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:cann:0",
                        "ascend_0:cann:2",
                        "ascend_1:cann:0",
                        "ascend_1:cann:2",
                        "ascend_2:cann:0",
                        "ascend_2:cann:2",
                        "ascend_3:cann:0",
                        "ascend_3:cann:2",
                    ],
                    gpus_per_replica=8,
                ),
            ),
            [
                """- The model requires approximately 156.24 GiB VRAM and 0.5 GiB RAM.
- With --npu-memory-fraction=0.9, all GPUs combined need to provide at least 173.60 GiB of total VRAM and each GPU needs 90% of allocatable VRAM.
- Manual GPU selection resulted in resource overcommit.
- Selected GPUs have 307.20 GiB allocatable VRAM, 0/8 of GPUs meet the VRAM utilization ratio, providing 276.48 GiB of allocatable VRAM."""
            ],
        ),
        (
            1,
            new_model(
                id=1,
                name="vocab_size_tp_divisibility_check",
                replicas=1,
                huggingface_repo_id="openai-community/gpt2",
                cpu_offloading=False,
                backend_parameters=[
                    "--tensor-parallel-size=4",
                    "--trust-remote-code",
                ],
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:npu:0",
                        "ascend_0:npu:1",
                        "ascend_1:npu:0",
                        "ascend_1:npu:1",
                    ],
                    gpus_per_replica=4,
                ),
                backend=BackendEnum.ASCEND_MINDIE.value,
            ),
            [
                "Model's vocabulary size (50257) must be divisible by the --tensor-parallel-size (4).",
                "",
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates_4x_64gx4_manually_check_msg(  # noqa: C901
    config, index, m, expect_msg
):
    def adjust_memory(worker):
        # Adjust the memory utilization of the 1st and 2nd workers.
        # The 0th and 2nd devices of 1st worker have allocated 60%,
        # while the 0th and 2nd devices of 2nd worker have allocated 40%.
        if worker.id == 1:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 1, 2, 3]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695
        elif worker.id == 2:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 1, 2, 3]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695
        elif worker.id == 3:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 1, 2, 3]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695
        elif worker.id == 4:
            for dev in worker.status.gpu_devices:
                if dev.index in [0, 1, 2, 3]:
                    dev.memory.utilization_rate = 40.0
                    dev.memory.used = 309847529063
                    dev.memory.allocated = 27487790695

    workers = [
        linux_ascend_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_3_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_ascend_4_910b_64gx8(return_device=4, callback=adjust_memory),
    ]
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)
    if index == 3:
        for worker in workers:
            worker.system_reserved.ram = worker.status.memory.total - 500

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        await resource_fit_selector.select_candidates(workers)
        assert resource_fit_selector.get_messages() == expect_msg


@pytest.mark.parametrize(
    "case_name, m, workers, expected_candidates",
    [
        # Auto schedule for DeepSeekV32 model
        (
            "auto_select_deepseekv32_model",
            new_model(
                1,
                "test_name",
                1,
                huggingface_repo_id="deepseek-ai/DeepSeek-V3.2",
                backend_version="0.11.0",
            ),
            [linux_ascend_1_910b_64gx8(), linux_ascend_2_910b_64gx8()],
            [
                expected_candidate(
                    1,
                    "ascend_0",
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    {
                        0: 54975581388,
                        1: 54975581388,
                        2: 54975581388,
                        3: 54975581388,
                        4: 54975581388,
                        5: 54975581388,
                        6: 54975581388,
                        7: 54975581388,
                    },
                    ram=536870912,
                    subworkers=[
                        ModelInstanceSubordinateWorker(
                            worker_id=2,
                            worker_ip="192.168.50.2",
                            total_gpus=8,
                            gpu_indexes=[0, 1, 2, 3, 4, 5, 6, 7],
                            gpu_addresses=[
                                '29.17.48.42',
                                '29.17.57.33',
                                '29.17.51.79',
                                '29.17.48.42',
                                '29.17.45.217',
                                '29.17.67.78',
                                '29.17.114.33',
                                '29.17.105.72',
                            ],
                            computed_resource_claim=ComputedResourceClaim(
                                is_unified_memory=False,
                                vram={
                                    0: 54975581388,
                                    1: 54975581388,
                                    2: 54975581388,
                                    3: 54975581388,
                                    4: 54975581388,
                                    5: 54975581388,
                                    6: 54975581388,
                                    7: 54975581388,
                                },
                                ram=536870912,
                            ),
                        )
                    ],
                )
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_select_candidates(config, case_name, m, workers, expected_candidates):
    model_instances = [
        ModelInstance(
            id=worker.id * 10 + gpu.index,
            worker_id=worker.id,
            gpu_indexes=[gpu.index],
            computed_resource_claim=ComputedResourceClaim(
                vram={gpu.index: gpu.memory.allocated}
            ),
        )
        for worker in workers
        for gpu in worker.status.gpu_devices
        if gpu.memory.allocated
    ]

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m, model_instances)

    with (
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            'gpustack.policies.candidate_selectors.base_candidate_selector.get_worker_model_instances',
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        actual = await resource_fit_selector.select_candidates(workers)
        compare_candidates(actual, expected_candidates)
