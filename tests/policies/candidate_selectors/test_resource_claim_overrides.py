from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.candidate_selectors.ascend_mindie_resource_fit_selector import (
    AscendMindIEResourceFitSelector,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.models import (
    BackendEnum,
    GPUSelector,
    Model,
    ROUTER_DEFAULT_MEMORY,
)
from tests.fixtures.workers.fixtures import (
    linux_ascend_1_910b_64gx8,
    linux_nvidia_1_4090_24gx1,
)

GIB = 1024**3


def deployment(backend, env=None, **kwargs):
    return Model(
        name="resource-overrides",
        source="huggingface",
        huggingface_repo_id="test/resource-overrides",
        backend=backend,
        categories=["llm"],
        env=env or {},
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection", ["auto", "manual", "manual_replicas", "cpu_fallback"]
)
@pytest.mark.parametrize(
    "ram", [None, "0", str(8 * GIB), str(128 * GIB), "invalid", "-1"]
)
@pytest.mark.parametrize("vram", [None, 20 * GIB])
async def test_custom_ram_override_controls_admission(config, selection, ram, vram):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    if selection == "cpu_fallback":
        worker.status.gpu_devices = []
    env = {} if vram is None else {"GPUSTACK_MODEL_VRAM_CLAIM": str(vram)}
    if ram is not None:
        env["GPUSTACK_MODEL_RAM_CLAIM"] = ram
    manual = selection.startswith("manual")
    m = deployment(
        BackendEnum.CUSTOM,
        env,
        replicas=2 if selection == "manual_replicas" else 1,
        cpu_offloading=selection == "cpu_fallback",
        gpu_selector=(
            GPUSelector(gpu_ids=[f"{worker.name}:cuda:0"], gpus_per_replica=1)
            if manual
            else None
        ),
    )
    selector = CustomBackendResourceFitSelector(
        config, m, [], resource_view=ResourceView([])
    )
    with patch(
        "gpustack.policies.utils.get_model_weight_size", return_value=10 * GIB
    ) as get_weights:
        candidates = await selector.select_candidates([worker])
    assert get_weights.call_count == (1 if vram is None else 0)
    expected_ram = int(ram) if ram in ("0", str(8 * GIB), str(128 * GIB)) else 2 * GIB
    assert selector.get_resource_claim().ram == expected_ram
    assert selector.get_resource_claim().vram == (14 * GIB if vram is None else vram)
    shortage = expected_ram > 64 * GIB
    assert bool(candidates) == (not shortage or selection == "manual")
    if candidates:
        assert all(c.computed_resource_claim.ram == expected_ram for c in candidates)
        assert all(bool(c.overcommit) == shortage for c in candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role_ram,env_ram,expected",
    [
        (None, None, ROUTER_DEFAULT_MEMORY),
        (None, "invalid", ROUTER_DEFAULT_MEMORY),
        (None, "-1", ROUTER_DEFAULT_MEMORY),
        (None, "0", 0),
        (None, str(8 * GIB), 8 * GIB),
        (None, str(128 * GIB), 128 * GIB),
        (4 * GIB, str(128 * GIB), 4 * GIB),
        (128 * GIB, "0", 128 * GIB),
        (0, str(128 * GIB), 0),
    ],
)
async def test_custom_cpu_only_role_resources_take_priority(
    config, role_ram, env_ram, expected
):
    worker = linux_nvidia_1_4090_24gx1()
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    env = {} if env_ram is None else {"GPUSTACK_MODEL_RAM_CLAIM": env_ram}
    selector = CustomBackendResourceFitSelector(
        config,
        deployment(BackendEnum.CUSTOM, env),
        [],
        cpu_only=True,
        ram_claim=role_ram,
        resource_view=ResourceView([]),
    )
    with patch("gpustack.policies.utils.get_model_weight_size") as get_weights:
        candidates = await selector.select_candidates([worker])
    get_weights.assert_not_called()
    assert selector.get_resource_claim().ram == expected
    assert selector.get_resource_claim().vram == 0
    assert bool(candidates) == (expected <= 64 * GIB)
    if candidates:
        assert candidates[0].computed_resource_claim.ram == expected
        assert not candidates[0].gpu_indexes


@pytest.fixture
def mindie_metadata():
    metadata = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=1024,
        vocab_size=32768,
        max_position_embeddings=2048,
        torch_dtype="bfloat16",
    )
    with (
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.get_pretrained_config_with_workers",
            new=AsyncMock(return_value=metadata),
        ),
        patch(
            "gpustack.policies.candidate_selectors.ascend_mindie_resource_fit_selector.get_model_weight_size",
            return_value=10 * GIB,
        ),
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("ram", [None, "0", str(8 * GIB), "invalid", "-1"])
@pytest.mark.parametrize("vram", [None, str(6 * GIB), "0", "invalid", "-1"])
async def test_mindie_resource_overrides_are_independent(
    config, mindie_metadata, ram, vram
):
    m = deployment(
        BackendEnum.ASCEND_MINDIE,
        backend_parameters=["--cpu-mem-size=1", "--max-preempt-count=1"],
    )
    selector = AscendMindIEResourceFitSelector(
        config, m, [], resource_view=ResourceView([])
    )
    await selector._init_model_parameters([])
    baseline = await selector._estimate_usage([])
    assert baseline.ram == GIB + GIB // 2
    m.env = {
        key: value
        for key, value in (
            ("GPUSTACK_MODEL_RAM_CLAIM", ram),
            ("GPUSTACK_MODEL_VRAM_CLAIM", vram),
        )
        if value is not None
    }
    claim = await selector._estimate_usage([])
    assert claim.ram == (int(ram) if ram in ("0", str(8 * GIB)) else baseline.ram)
    assert claim.vram == (6 * GIB if vram == str(6 * GIB) else baseline.vram)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["auto", "manual", "manual_replicas"])
@pytest.mark.parametrize("ram", [8, 128])
async def test_mindie_ram_only_override_controls_admission(
    config, mindie_metadata, selection, ram
):
    worker = linux_ascend_1_910b_64gx8(return_device=1)
    worker.status.memory.total = 64 * GIB
    worker.system_reserved.ram = 0
    m = deployment(
        BackendEnum.ASCEND_MINDIE,
        {"GPUSTACK_MODEL_RAM_CLAIM": str(ram * GIB)},
        replicas=2 if selection == "manual_replicas" else 1,
        gpu_selector=(
            GPUSelector(gpu_ids=[f"{worker.name}:cann:0"], gpus_per_replica=1)
            if selection.startswith("manual")
            else None
        ),
    )
    selector = AscendMindIEResourceFitSelector(
        config, m, [], resource_view=ResourceView([])
    )
    candidates = await selector.select_candidates([worker])
    assert bool(candidates) == (ram <= 64 or selection == "manual")
    if candidates:
        assert candidates[0].computed_resource_claim.ram == ram * GIB
        assert bool(candidates[0].overcommit) == (ram > 64)
