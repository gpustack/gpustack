import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpustack.scheduler import evaluator
from gpustack.schemas.catalog_source import (
    KIND_MODEL_SET,
    CatalogModelEntry,
    build_catalog_entries,
    normalize_catalog_yaml,
)
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.schemas.workers import GPUDeviceStatus
from gpustack.server import catalog
from gpustack.utils.command import find_parameter
from gpustack.utils.compat_importlib import pkg_resources


@pytest.fixture(params=["model-catalog.yaml", "model-catalog-modelscope.yaml"])
def packaged_catalog(request, monkeypatch):
    path = pkg_resources.files("gpustack.assets").joinpath(request.param)
    content = normalize_catalog_yaml(catalog.read_builtin_catalog_text(str(path)))
    entries = [
        CatalogModelEntry(
            kind=entry.kind,
            name=entry.name,
            payload=entry.payload,
            source_name="builtin",
            source_type=SourceTypeEnum.BUILTIN,
        )
        for entry in build_catalog_entries(
            [SourceContent("builtin", SourceTypeEnum.BUILTIN, content)]
        )
        if entry.kind == KIND_MODEL_SET
    ]
    monkeypatch.setattr(catalog, "_model_set_entries", AsyncMock(return_value=entries))
    return request.param


def make_model(catalog_file, name="MiniMax-M3.1-Flash", **kwargs):
    if catalog_file == "model-catalog.yaml":
        org = "Nanbeige" if name.startswith("Nanbeige") else "MiniMaxAI"
        source = dict(source="huggingface", huggingface_repo_id=f"{org}/{name}")
    else:
        org = "nanbeige" if name.startswith("Nanbeige") else "MiniMax"
        source = dict(source="model_scope", model_scope_model_id=f"{org}/{name}")
    return ModelSpec(**source, backend="vLLM", **kwargs)


def make_worker(name="hopper", vendor="nvidia", cc="9.0", arch_family=None):
    return SimpleNamespace(
        name=name,
        labels={"accelerator": name},
        status=SimpleNamespace(
            gpu_devices=[
                GPUDeviceStatus(
                    vendor=vendor,
                    compute_capability=cc,
                    arch_family=arch_family,
                    type="cann" if vendor == "ascend" else "cuda",
                    index=0,
                )
            ]
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cc,cache_dtype",
    [
        ("9.0", "fp8"),
        ("10.0", "nvfp4"),
        ("10.3", "nvfp4"),
        ("12.0", "fp8"),
        ("12.1", "fp8"),
    ],
)
async def test_evaluation_uses_gpu_specific_defaults(
    packaged_catalog, monkeypatch, cc, cache_dtype
):
    model = make_model(packaged_catalog)
    # Stop after default selection, before metadata or resource evaluation.
    monkeypatch.setattr(evaluator, "set_gguf_model_file_path", AsyncMock())
    monkeypatch.setattr(
        evaluator, "evaluate_model_input", AsyncMock(return_value=(False, []))
    )
    result = await evaluator.evaluate_model(None, None, model, [make_worker(cc=cc)], [])
    assert result.default_spec is not None
    parameters = result.default_spec.backend_parameters
    assert find_parameter(parameters, ["kv-cache-dtype"]) == cache_dtype
    assert ("--moe-backend=marlin" in parameters) is (cache_dtype == "fp8")
    attention = json.loads(find_parameter(parameters, ["attention-config"]))
    assert attention["indexer_kv_dtype"] == "fp8"


@pytest.mark.asyncio
async def test_hopper_and_blackwell_rtx_share_default_recipe(packaged_catalog):
    model = make_model(packaged_catalog)
    workers = [make_worker(), make_worker("rtx-pro-6000", cc="12.0")]
    assert await evaluator.set_default_spec(None, model, workers)
    assert find_parameter(model.backend_parameters, ["kv-cache-dtype"]) == "fp8"
    assert "--moe-backend=marlin" in model.backend_parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["gpu", "worker"])
async def test_manual_selection_narrows_default_hardware(packaged_catalog, selection):
    model = make_model(
        packaged_catalog,
        **(
            {"gpu_selector": {"gpu_ids": ["hopper:cuda:0"]}}
            if selection == "gpu"
            else {"worker_selector": {"accelerator": "hopper"}}
        ),
    )
    workers = [make_worker("blackwell", cc="10.0"), make_worker()]
    await evaluator.set_default_spec(None, model, workers)
    assert find_parameter(model.backend_parameters, ["kv-cache-dtype"]) == "fp8"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workers",
    [
        [],
        [make_worker(cc=None)],
        [make_worker(cc="8.0")],
        [make_worker(cc="11.0")],
        [make_worker(cc="13.0")],
        [make_worker(vendor="amd", cc="10.0")],
        [make_worker(), make_worker("blackwell", cc="10.0")],
        [make_worker(), make_worker("unsupported", cc="8.0")],
    ],
)
async def test_no_hardware_default_for_missing_unsupported_or_mixed_gpus(
    packaged_catalog, workers
):
    model = make_model(packaged_catalog)
    assert not await evaluator.set_default_spec(None, model, workers)
    assert model.backend_parameters is None


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["gpu", "worker"])
async def test_unmatched_selection_does_not_use_other_devices(
    packaged_catalog, selection
):
    model = make_model(
        packaged_catalog,
        **(
            {"gpu_selector": {"gpu_ids": ["missing:cuda:0"]}}
            if selection == "gpu"
            else {"worker_selector": {"accelerator": "missing"}}
        ),
    )
    await evaluator.set_default_spec(None, model, [make_worker()])
    assert model.backend_parameters is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workers",
    [[], [make_worker(vendor="ascend", cc=None, arch_family="Ascend910B3")]],
)
async def test_unrestricted_recipe_is_available_on_ascend_or_without_gpus(
    packaged_catalog, workers
):
    model = make_model(packaged_catalog, name="Nanbeige4.3-33B")
    assert await evaluator.set_default_spec(None, model, workers)
    assert "--reasoning-parser=qwen3" in model.backend_parameters
    assert find_parameter(model.backend_parameters, ["dtype"]) is None
    assert find_parameter(model.backend_parameters, ["gpu-memory-utilization"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("parameters", [[], ["--max-model-len=8192"]])
async def test_explicit_parameters_are_preserved(packaged_catalog, parameters):
    model = make_model(packaged_catalog, backend_parameters=parameters, env={})
    assert not await evaluator.set_default_spec(None, model, [make_worker()])
    assert model.backend_parameters == parameters
    assert model.env == {}


@pytest.mark.asyncio
async def test_explicit_backend_does_not_receive_another_backend_recipe(
    packaged_catalog,
):
    model = make_model(packaged_catalog)
    model.backend = "SGLang"
    assert not await evaluator.set_default_spec(None, model, [make_worker()])
    assert model.backend_parameters is None
