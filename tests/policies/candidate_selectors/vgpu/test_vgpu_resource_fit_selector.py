from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.candidate_selectors import VGPUResourceFitSelector
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile,
    GPUInstanceTypeDetail,
    GPUInstanceTypeStatusPublic,
)
from gpustack.schemas.models import GPUTypeSelector
from gpustack.utils.resource_usage import parse_quantity_to_mib
from tests.fixtures.workers.fixtures import (
    linux_nvidia_5_a100_80gx2,
    linux_nvidia_6_a100_80gx2,
    linux_rocm_1_7800_16gx1,
)
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model

_GIB = 1024**3

_PATCH_SESSION = (
    "gpustack.policies.candidate_selectors.vgpu_resource_fit_selector.async_session"
)
_PATCH_ALL_BY_FIELDS = (
    "gpustack.policies.candidate_selectors.vgpu_resource_fit_selector."
    "GPUInstanceType.all_by_fields"
)
_PATCH_ESTIMATE = (
    "gpustack.policies.candidate_selectors.vgpu_resource_fit_selector."
    "estimate_model_vram"
)
_PATCH_RAM = (
    "gpustack.policies.candidate_selectors.vgpu_resource_fit_selector."
    "get_model_ram_claim"
)


def make_instance_type(
    name="pool-a100",
    manufacturer="nvidia",
    product="A100-80GB-PCIe",
    memory="81920Mi",
    profiles=None,
):
    sliced_detail = None
    if profiles is not None:
        sliced_detail = GPUInstanceTypeAcceleratorSlicedDetail(
            physical=GPUInstanceTypeAcceleratorSlicedPhysicalDetail(
                count=len(profiles),
                profiles=[
                    GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile(
                        name=p_name, count=1, memory_mib=p_mib
                    )
                    for p_name, p_mib in profiles
                ],
            )
        )
    return GPUInstanceType(
        id=1,
        cluster_id=1,
        name=name,
        status=GPUInstanceTypeStatusPublic(
            detail=GPUInstanceTypeDetail(
                manufacturer=manufacturer,
                product=product,
                memory=memory,
                sliced_detail=sliced_detail,
            )
        ),
    )


def make_devices(
    node,
    *,
    manufacturer="nvidia",
    product="NVIDIA A100 80GB PCIe",
    logical_count=None,
    physical_profiles=None,
    remaining=800000,
    group_id="pool",
):
    """One node's ``Devices``, the shape the cluster's operator publishes:
    which slicing modes the node has enabled, and what each accelerator has
    left. ``logical_count=None`` / ``physical_profiles=None`` mean that mode is
    off on this node."""
    detail = {}
    if logical_count is not None:
        detail["logical"] = {"count": logical_count}
    if physical_profiles is not None:
        detail["physical"] = {
            "count": len(physical_profiles),
            "profiles": [
                {"name": p_name, "count": 1, "memoryMib": p_mib}
                for p_name, p_mib in physical_profiles
            ],
        }
    return {
        "metadata": {"name": node},
        "spec": {
            "groups": [
                {
                    "id": group_id,
                    "manufacturer": manufacturer,
                    "name": product,
                    "acceleratorSlicedDetail": detail,
                }
            ]
        },
        "status": {
            "groups": [
                {
                    "id": group_id,
                    "manufacturer": manufacturer,
                    "accelerators": [{"index": 0, "remaining": remaining}],
                }
            ]
        },
    }


def make_vgpu_model(selector: GPUTypeSelector, distributed=False):
    return new_model(
        1,
        "test",
        1,
        huggingface_repo_id="deepseek-ai/DeepSeek-R1",
        distributed_inference_across_workers=distributed,
        cluster_id=1,
        gpu_type_selector=selector,
    )


async def run_selector(
    config, model, workers, instance_types, vram_claim, devices=None
):
    """``devices`` stands in for the cluster read, keyed by node name. ``None``
    is the "cannot tell" case (no cluster, or the read failed), where the fit
    falls back to every pool worker."""

    async def fake_all_by_fields(session, fields):
        # Mirror the targeted name filter the selector now applies.
        if "name" in fields:
            return [it for it in instance_types if it.name == fields["name"]]
        return instance_types

    with (
        patch(_PATCH_SESSION, return_value=mock_async_session()),
        patch(
            _PATCH_ALL_BY_FIELDS,
            new=fake_all_by_fields,
        ),
        patch(_PATCH_ESTIMATE, new=AsyncMock(return_value=vram_claim)),
        patch(_PATCH_RAM, return_value=0),
        patch.object(
            VGPUResourceFitSelector,
            "_load_cluster_devices",
            new=AsyncMock(return_value=devices),
        ),
    ):
        selector = VGPUResourceFitSelector(config, model, [])
        candidates = await selector.select_candidates(workers)
    return selector, candidates


@pytest.mark.asyncio
async def test_sliced_single_worker_fit(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    workers = [linux_nvidia_5_a100_80gx2(), linux_rocm_1_7800_16gx1()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 30 * _GIB
    )

    # Only the NVIDIA worker matches the pool; the ROCm worker is excluded.
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.worker.name == workers[0].name
    assert candidate.gpu_indexes is None
    assert candidate.gpu_type == "cuda"
    assert candidate.computed_resource_claim.vram == {0: 40 * _GIB}
    assert selector.get_messages() == []


@pytest.mark.asyncio
async def test_sliced_oversized_without_distributed(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    workers = [linux_nvidia_5_a100_80gx2()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 70 * _GIB
    )

    assert candidates == []
    messages = selector.get_messages()
    assert len(messages) == 1
    assert "70.00 GiB" in messages[0] or "70" in messages[0]
    assert "pool-a100" in messages[0]


@pytest.mark.asyncio
async def test_sliced_distributed_across_workers(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        ),
        distributed=True,
    )
    workers = [linux_nvidia_5_a100_80gx2(), linux_nvidia_6_a100_80gx2()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 70 * _GIB
    )

    # 70 GiB over 40 GiB slices needs 2 workers, one slice each.
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.gpu_indexes is None
        assert candidate.computed_resource_claim.vram == {0: 40 * _GIB}
        assert len(candidate.subordinate_workers) == 1
        subordinate = candidate.subordinate_workers[0]
        assert subordinate.gpu_indexes is None
        assert subordinate.computed_resource_claim.vram == {0: 40 * _GIB}
        assert subordinate.worker_name != candidate.worker.name
    assert selector.get_messages() == []


@pytest.mark.asyncio
async def test_distributed_not_enough_workers(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        ),
        distributed=True,
    )
    workers = [linux_nvidia_5_a100_80gx2()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 70 * _GIB
    )

    assert candidates == []
    assert "needs 2 slices" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_whole_card_mode_uses_full_card_vram(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=0,
            accelerator_sliced_cores_percentage=0,
        )
    )
    workers = [linux_nvidia_5_a100_80gx2()]

    _, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 70 * _GIB
    )

    assert len(candidates) == 1
    assert candidates[0].computed_resource_claim.vram == {0: 80 * _GIB}


@pytest.mark.asyncio
async def test_partitioned_profile_uses_profile_memory(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_partitioned_profile="1g.40gb",
        )
    )
    worker = linux_nvidia_5_a100_80gx2()
    instance_type = make_instance_type(profiles=[("1g.40gb", 40960)])

    _, candidates = await run_selector(
        config,
        model,
        [worker],
        [instance_type],
        30 * _GIB,
        devices={
            worker.name: make_devices(
                worker.name, physical_profiles=[("1g.40gb", 40960)]
            )
        },
    )

    assert len(candidates) == 1
    assert candidates[0].computed_resource_claim.vram == {0: 40960 * 1024 * 1024}


@pytest.mark.asyncio
async def test_profile_claim_picks_only_the_node_with_partitioning(config):
    # One InstanceType spans both nodes — same product — but the slicing mode is
    # per node. A profile claim must land on the node with partitioning on;
    # picking the other one leaves the workload Pending on a resource that node
    # never advertises.
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_partitioned_profile="1g.40gb",
        )
    )
    logical_node = linux_nvidia_5_a100_80gx2()
    physical_node = linux_nvidia_6_a100_80gx2()
    instance_type = make_instance_type(profiles=[("1g.40gb", 40960)])

    _, candidates = await run_selector(
        config,
        model,
        [logical_node, physical_node],
        [instance_type],
        30 * _GIB,
        devices={
            logical_node.name: make_devices(logical_node.name, logical_count=128),
            physical_node.name: make_devices(
                physical_node.name, physical_profiles=[("1g.40gb", 40960)]
            ),
        },
    )

    assert [c.worker.name for c in candidates] == [physical_node.name]


@pytest.mark.asyncio
async def test_profile_claim_needs_the_profile_offered_by_the_node(config):
    # Partitioning is on, but this node cannot cut the profile asked for.
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_partitioned_profile="1g.40gb",
        )
    )
    worker = linux_nvidia_5_a100_80gx2()
    instance_type = make_instance_type(profiles=[("1g.40gb", 40960)])

    selector, candidates = await run_selector(
        config,
        model,
        [worker],
        [instance_type],
        30 * _GIB,
        devices={
            worker.name: make_devices(
                worker.name, physical_profiles=[("2g.80gb", 81920)]
            )
        },
    )

    assert candidates == []
    assert "profile '1g.40gb'" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_sliced_claim_skips_node_without_logical_slicing(config):
    # The converse: a node in a hardware-partitioning mode offers no software
    # slice, so a ratio claim must not be sent to it.
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    worker = linux_nvidia_5_a100_80gx2()

    selector, candidates = await run_selector(
        config,
        model,
        [worker],
        [make_instance_type()],
        30 * _GIB,
        devices={
            worker.name: make_devices(
                worker.name, physical_profiles=[("1g.40gb", 40960)]
            )
        },
    )

    assert candidates == []
    assert "software slicing" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_claim_skips_node_with_nothing_remaining(config):
    # The mode is enabled but every accelerator is already fully claimed.
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    worker = linux_nvidia_5_a100_80gx2()

    selector, candidates = await run_selector(
        config,
        model,
        [worker],
        [make_instance_type()],
        30 * _GIB,
        devices={
            worker.name: make_devices(worker.name, logical_count=128, remaining=0)
        },
    )

    assert candidates == []
    assert "capacity left" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_claim_skips_worker_the_cluster_reports_no_devices_for(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    worker = linux_nvidia_5_a100_80gx2()

    _, candidates = await run_selector(
        config, model, [worker], [make_instance_type()], 30 * _GIB, devices={}
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_unreadable_devices_keeps_every_pool_worker(config):
    # A cluster read that could not answer must not block scheduling; the
    # node-side scheduler still has the final say.
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    worker = linux_nvidia_5_a100_80gx2()

    _, candidates = await run_selector(
        config, model, [worker], [make_instance_type()], 30 * _GIB, devices=None
    )

    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_unknown_partitioned_profile(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_partitioned_profile="2g.80gb",
        )
    )
    workers = [linux_nvidia_5_a100_80gx2()]
    instance_type = make_instance_type(profiles=[("1g.40gb", 40960)])

    selector, candidates = await run_selector(
        config, model, workers, [instance_type], 30 * _GIB
    )

    assert candidates == []
    assert "2g.80gb" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_unknown_instance_type(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-h100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    workers = [linux_nvidia_5_a100_80gx2()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 30 * _GIB
    )

    assert candidates == []
    assert "pool-h100" in selector.get_messages()[0]


@pytest.mark.asyncio
async def test_no_matching_workers(config):
    model = make_vgpu_model(
        GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    workers = [linux_rocm_1_7800_16gx1()]

    selector, candidates = await run_selector(
        config, model, workers, [make_instance_type()], 30 * _GIB
    )

    assert candidates == []
    assert "No workers have GPUs matching" in selector.get_messages()[0]


def test_parse_pool_memory_quantity():
    assert parse_quantity_to_mib("81920Mi") == 81920
    assert parse_quantity_to_mib("80Gi") == 80 * 1024


@pytest.mark.asyncio
async def test_evaluation_spec_without_cluster_attr(config):
    """Evaluation-time ModelSpec carries gpu_type_selector but may have no
    cluster stamped: the selector must fall back to an unscoped projection
    read instead of raising AttributeError."""
    from gpustack.schemas.model_sets import ModelSpec
    from gpustack.schemas.models import SourceEnum

    spec = ModelSpec(
        name="eval-spec",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        gpu_type_selector=GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        ),
    )
    spec.cluster_id = None
    workers = [linux_nvidia_5_a100_80gx2()]

    captured = {}

    async def fake_all_by_fields(session, fields):
        captured["fields"] = fields
        return [make_instance_type()]

    with (
        patch(_PATCH_SESSION, return_value=mock_async_session()),
        patch(_PATCH_ALL_BY_FIELDS, new=fake_all_by_fields),
        patch(_PATCH_ESTIMATE, new=AsyncMock(return_value=30 * _GIB)),
        patch(_PATCH_RAM, return_value=0),
    ):
        selector = VGPUResourceFitSelector(config, spec, [])
        candidates = await selector.select_candidates(workers)

    assert "cluster_id" not in captured["fields"]
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_evaluation_spec_with_stamped_cluster(config):
    """The evaluation route stamps the request's cluster onto each spec;
    the selector must scope the InstanceType read by it."""
    from gpustack.schemas.model_sets import ModelSpec
    from gpustack.schemas.models import SourceEnum

    spec = ModelSpec(
        name="eval-spec",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        gpu_type_selector=GPUTypeSelector(
            type="pool-a100",
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        ),
    )
    spec.cluster_id = 42
    workers = [linux_nvidia_5_a100_80gx2()]

    captured = {}

    async def fake_all_by_fields(session, fields):
        captured["fields"] = fields
        return [make_instance_type()]

    with (
        patch(_PATCH_SESSION, return_value=mock_async_session()),
        patch(_PATCH_ALL_BY_FIELDS, new=fake_all_by_fields),
        patch(_PATCH_ESTIMATE, new=AsyncMock(return_value=30 * _GIB)),
        patch(_PATCH_RAM, return_value=0),
    ):
        selector = VGPUResourceFitSelector(config, spec, [])
        candidates = await selector.select_candidates(workers)

    assert captured["fields"]["cluster_id"] == 42
    assert len(candidates) == 1
