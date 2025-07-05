import copy

import pytest
from unittest.mock import patch, AsyncMock

from tests.utils.model import new_model
from gpustack.policies.candidate_selectors import AscendMindIEResourceFitSelector
from gpustack.schemas.models import (
    ModelInstance,
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
    GPUSelector,
)
from tests.fixtures.workers.fixtures import (
    linux_huawei_1_910b_64gx8,
    linux_huawei_2_910b_64gx8,
    linux_huawei_3_910b_64gx8,
    linux_huawei_4_910b_64gx8,
)
from tests.utils.scheduler import compare_candidates


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
                        "ascend_2:npu:0",
                        "ascend_4:npu:0",  # Unavailable worker.
                    ],
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
                        "ascend_2:npu:0",
                        "ascend_3:npu:0",  # Unavailable device.
                    ],
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
        # - There are two candidates be selected,
        #   but with quick fit, it should only select the first one.
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
                    "vram": {0: 61847529062},
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
                    "vram": {0: 61847529062},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=1,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.41"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 61847529062},
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
        linux_huawei_1_910b_64gx8(return_device=1),
        linux_huawei_2_910b_64gx8(return_device=1, callback=adjust_memory),
        linux_huawei_3_910b_64gx8(return_device=1),
        linux_huawei_4_910b_64gx8(return_device=0),  # No devices.
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
        linux_huawei_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_3_910b_64gx8(return_device=2),
        linux_huawei_4_910b_64gx8(return_device=2),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)
    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
        # - There are two candidates be selected,
        #   but with quick fit, it should only select the first one.
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
                    "vram": {0: 61847529062},
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
                    "vram": {0: 61847529062, 1: 61847529062},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=2,
                            gpu_indexes=[0, 1],
                            gpu_addresses=["29.17.48.41", "29.17.57.32"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 61847529062, 1: 61847529062},
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
        linux_huawei_1_910b_64gx8(return_device=2),
        linux_huawei_2_910b_64gx8(return_device=2, callback=adjust_memory),
        linux_huawei_3_910b_64gx8(return_device=2),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
                        "ascend_2:npu:0",
                    ],
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 3,
                    "worker_name": "ascend_2",
                    "gpu_indexes": [0],
                    "gpu_addresses": ["29.17.48.41"],
                    "ram": 536870912,
                    "vram": {0: 61847529062},
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
                        "ascend_0:npu:0",
                        "ascend_0:npu:1",
                    ],
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
                    "--trust-remote-code",
                ],
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
        # - There are two candidates be selected,
        #   but with quick fit, it should only select the first one.
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
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1],
                    "gpu_addresses": ["29.17.57.31"],
                    "ram": 536870912,
                    "vram": {1: 61847529062},
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
        # - There are two candidates be selected,
        #   but with quick fit, it should only select the first one.
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
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 61847529062, 2: 61847529062},
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
        # - There are two candidates be selected,
        #   but with quick fit, it should only select the first one.
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
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 61847529062, 2: 61847529062},
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
                        "ascend_0:npu:1",
                        "ascend_0:npu:2",
                        "ascend_2:npu:3",
                        "ascend_2:npu:4",
                    ],
                ),
                backend_parameters=[
                    "--trust-remote-code",
                ],
            ),
            [
                {
                    "worker_id": 1,
                    "worker_name": "ascend_0",
                    "gpu_indexes": [1, 2],
                    "gpu_addresses": ["29.17.57.31", "29.17.51.57"],
                    "ram": 536870912,
                    "vram": {1: 61847529062, 2: 61847529062},
                    "subordinate_workers": [
                        ModelInstanceSubordinateWorker(
                            worker_id=3,
                            worker_ip="192.168.50.4",
                            total_gpus=8,
                            gpu_indexes=[3, 4],
                            gpu_addresses=["29.17.49.41", "29.17.45.216"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={3: 61847529062, 4: 61847529062},
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
                        "ascend_0:npu:0",
                        "ascend_2:npu:0",
                    ],
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.8",
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
                            total_gpus=8,
                            gpu_indexes=[0],
                            gpu_addresses=["29.17.48.41"],
                            computed_resource_claim=ComputedResourceClaim(
                                ram=536870912,
                                vram={0: 54975581388},
                            ),
                        ),
                    ],
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
        linux_huawei_1_910b_64gx8(callback=adjust_memory),
        linux_huawei_2_910b_64gx8(callback=adjust_memory),
        linux_huawei_3_910b_64gx8(),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
                        "ascend_0:npu:0",
                        "ascend_0:npu:1",
                        "ascend_0:npu:2",
                        "ascend_0:npu:3",
                        "ascend_0:npu:4",
                        "ascend_0:npu:5",
                        "ascend_0:npu:6",
                        "ascend_0:npu:7",
                        "ascend_1:npu:0",
                        "ascend_1:npu:1",
                        "ascend_1:npu:2",
                        "ascend_1:npu:3",
                        "ascend_1:npu:4",
                        "ascend_1:npu:5",
                        "ascend_1:npu:6",
                        "ascend_1:npu:7",
                        "ascend_2:npu:0",
                        "ascend_2:npu:1",
                        "ascend_2:npu:2",
                        "ascend_2:npu:3",
                        "ascend_2:npu:4",
                        "ascend_2:npu:5",
                        "ascend_2:npu:6",
                        "ascend_2:npu:7",
                        "ascend_3:npu:0",
                        "ascend_3:npu:1",
                        "ascend_3:npu:2",
                        "ascend_3:npu:3",
                        "ascend_3:npu:4",
                        "ascend_3:npu:5",
                        "ascend_3:npu:6",
                        "ascend_3:npu:7",
                    ],
                ),
                backend_parameters=[
                    "--npu-memory-fraction=0.95",
                    "--data-parallel-size=4",
                    "--tensor-parallel-size=8",
                    "--trust-remote-code",
                ],
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
        linux_huawei_1_910b_64gx8(),
        linux_huawei_2_910b_64gx8(),
        linux_huawei_3_910b_64gx8(),
        linux_huawei_4_910b_64gx8(),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
async def test_select_candidates_2x_64gx4_2x_64gx2_check_msg(config, m, expected):
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
        linux_huawei_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_3_910b_64gx8(return_device=2),
        linux_huawei_4_910b_64gx8(return_device=2),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)

    resource_fit_selector._serving_params.npu_memory_fraction = 0.2

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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
        compare_candidates(actual, [])

        expect_msg = [
            """- The model requires approximately 0.5 GiB RAM and 156.24 GiB VRAM.
- With --npu-memory-fraction=0.2, All GPUs combined need to provide at least 781.20 GiB of total VRAM.
- The largest available worker has 102.4 GiB allocatable VRAM, 4/4 of GPUs meet the VRAM utilization ratio, providing 20.48 GiB of allocatable VRAM."""
        ]
        assert resource_fit_selector.get_messages() == expect_msg


@pytest.mark.parametrize(
    "m, expected",
    [
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
                gpu_selector=GPUSelector(
                    gpu_ids=[
                        "ascend_0:npu:0",
                        "ascend_0:npu:2",
                        "ascend_1:npu:0",
                        "ascend_1:npu:2",
                        "ascend_2:npu:0",
                        "ascend_2:npu:2",
                        "ascend_3:npu:0",
                        "ascend_3:npu:2",
                    ],
                ),
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
async def test_select_candidates_4x_64gx4_manually_check_msg(  # noqa: C901
    config, m, expected
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
        linux_huawei_1_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_2_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_3_910b_64gx8(return_device=4, callback=adjust_memory),
        linux_huawei_4_910b_64gx8(return_device=4, callback=adjust_memory),
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

    resource_fit_selector = AscendMindIEResourceFitSelector(config, m)
    resource_fit_selector._serving_params.npu_memory_fraction = 0.9

    resource_fit_selector2 = AscendMindIEResourceFitSelector(config, m)
    resource_fit_selector2._serving_params.npu_memory_fraction = 0.1

    resource_fit_selector3 = copy.deepcopy(resource_fit_selector)

    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
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

        expect_msg = [
            """- The model requires approximately 0.5 GiB RAM and 156.24 GiB VRAM.
- With --npu-memory-fraction=0.9, All GPUs combined need to provide at least 173.60 GiB of total VRAM.
- Worker ascend_0 GPU indexes [0, 2] and other 3 workers fails to meet the 90.0% allocatable VRAM ratio."""
        ]
        assert resource_fit_selector.get_messages() == expect_msg

        # case 2
        await resource_fit_selector2.select_candidates(workers)

        expect_msg2 = [
            """- The model requires approximately 0.5 GiB RAM and 156.24 GiB VRAM.
- With --npu-memory-fraction=0.1, All GPUs combined need to provide at least 1562.40 GiB of total VRAM.
- Selected GPUs have 51.2 GiB allocatable VRAM, 8/8 of GPUs meet the VRAM utilization ratio, providing 5.12 GiB of allocatable VRAM."""
        ]

        assert resource_fit_selector2.get_messages() == expect_msg2

    # case 3
    with (
        patch("sqlmodel.ext.asyncio.session.AsyncSession", AsyncMock()),
        patch(
            "gpustack.policies.utils.get_worker_model_instances",
            return_value=model_instances,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all",
            return_value=workers,
        ),
    ):
        for worker in workers:
            worker.system_reserved.ram = worker.status.memory.total - 500

        await resource_fit_selector3.select_candidates(workers)

        expect_msg3 = [
            """- The model requires approximately 0.5 GiB RAM and 156.24 GiB VRAM.
- With --npu-memory-fraction=0.9, All GPUs combined need to provide at least 173.60 GiB of total VRAM.
- Worker ['ascend_0', 'ascend_1']...(more 2) fails to meet the required RAM.
- Worker ascend_0 GPU indexes [0, 2] and other 3 workers fails to meet the 90.0% allocatable VRAM ratio."""
        ]

        assert resource_fit_selector3.get_messages() == expect_msg3
