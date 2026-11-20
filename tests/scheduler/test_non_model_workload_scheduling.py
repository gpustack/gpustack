from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from gpustack.policies.base import Allocated
from gpustack.policies.resource_view import ResourceView
from gpustack.policies.worker_filters import backend_framework_filter
from gpustack.scheduler import evaluator, scheduler, port_budget
from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import (
    GPUSelector,
    GPUTypeSelector,
    Model,
    ModelInstanceStateEnum,
)
from gpustack.schemas.port_reservations import WorkerPortAvailability
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1
from tests.utils.mock import mock_async_session
from tests.utils.model import new_model_instance

GIB = 1024**3


@pytest.fixture
def scheduling(monkeypatch):
    worker = linux_nvidia_1_4090_24gx1()
    worker.cluster_id = 1
    worker.status.memory.total = 8 * GIB
    worker.system_reserved.ram = 0
    monkeypatch.setattr(backend_framework_filter, "async_session", mock_async_session)
    monkeypatch.setattr(
        backend_framework_filter.RunnerOverrideEntry, "all", AsyncMock(return_value=[])
    )
    backend = InferenceBackend(
        backend_name="custom-frontend",
        owner_principal_id=None,
        version_configs=VersionConfigDict(
            root={
                framework: {
                    "image_name": "test:same-image",
                    "custom_framework": framework,
                }
                for framework in ("cpu", "cuda")
            }
        ),
    )
    monkeypatch.setattr(InferenceBackend, "all", AsyncMock(return_value=[backend]))
    monkeypatch.setattr("gpustack.envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE", 0)
    monkeypatch.setattr(
        port_budget,
        "worker_port_snapshots",
        AsyncMock(
            return_value={
                worker.id: WorkerPortAvailability(
                    worker_id=worker.id, generation=uuid4(), port_range="40000-40255"
                )
            }
        ),
    )
    estimate = AsyncMock(return_value=3 * GIB)
    monkeypatch.setattr(
        "gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector.estimate_model_vram",
        estimate,
    )
    return worker, estimate


def deployment(**kwargs):
    return Model(
        id=1,
        name="frontend",
        source="local_path",
        local_path="/models/tokenizer",
        cluster_id=1,
        **{
            "backend": "custom-frontend",
            "backend_version": "cpu",
            "env": {"GPUSTACK_NON_MODEL_WORKLOAD": "true"},
            **kwargs,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["cpu", "cuda", "direct-image"])
@pytest.mark.parametrize("cpu_offloading", [False, True])
@pytest.mark.parametrize("gpu_full", [False, True])
@pytest.mark.parametrize("ram", [None, 4, 16])
async def test_non_model_placement_uses_only_ram(
    config, scheduling, version, cpu_offloading, gpu_full, ram
):
    worker, estimate = scheduling
    env = {"GPUSTACK_NON_MODEL_WORKLOAD": "true"}
    if ram is not None:
        env["GPUSTACK_MODEL_RAM_CLAIM"] = str(ram * GIB)
    model = deployment(
        backend="Custom" if version == "direct-image" else "custom-frontend",
        backend_version=None if version == "direct-image" else version,
        image_name="test:same-image" if version == "direct-image" else None,
        run_command="python -m frontend",
        env=env,
        cpu_offloading=cpu_offloading,
    )
    reservations = {
        worker.id: Allocated(
            ram=0,
            vram={0: worker.status.gpu_devices[0].memory.total} if gpu_full else {},
        )
    }
    candidate, messages = await scheduler.find_candidate(
        None,
        config,
        model,
        [worker],
        [],
        resource_view=ResourceView([], reservations=reservations),
    )
    estimate.assert_not_awaited()
    if ram == 16:
        assert candidate is None
        assert any("RAM" in message for message in messages)
        return
    assert candidate is not None, messages
    assert candidate.gpu_indexes is None
    assert candidate.gpu_type is None
    assert candidate.computed_resource_claim.vram == {}
    assert candidate.computed_resource_claim.ram == (ram or 2) * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["cpu", "cuda"])
@pytest.mark.parametrize("flag", [None, "false"])
async def test_cpu_framework_without_non_model_opt_in_keeps_existing_gpu_scheduling(
    config, scheduling, version, flag
):
    worker, estimate = scheduling
    model = deployment(
        backend_version=version,
        env={} if flag is None else {"GPUSTACK_NON_MODEL_WORKLOAD": flag},
        cpu_offloading=False,
    )
    candidate, messages = await scheduler.find_candidate(
        None, config, model, [worker], [], resource_view=ResourceView([])
    )
    estimate.assert_awaited_once()
    assert candidate is not None, messages
    assert candidate.gpu_type == "cuda"
    assert candidate.gpu_indexes == [0]
    assert candidate.computed_resource_claim.vram == {0: 3 * GIB}


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ["gpu_selector", "gpu_type_selector"])
async def test_invalid_saved_gpu_selection_fails_even_without_workers(config, selector):
    value = (
        GPUSelector(gpu_ids=["missing:cuda:0"])
        if selector == "gpu_selector"
        else GPUTypeSelector(type="test-slice")
    )
    candidate, messages = await scheduler.find_candidate(
        None,
        config,
        deployment(**{selector: value}),
        [],
        [],
        resource_view=ResourceView([]),
    )
    assert candidate is None
    assert "not supported" in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["tokenizer", "model.gguf"])
async def test_evaluation_accepts_non_model_files_without_architecture_or_weight_parsing(
    config, scheduling, monkeypatch, tmp_path, filename
):
    worker, estimate = scheduling
    source = tmp_path / filename
    source.touch()
    model = ModelSpec(
        source="local_path",
        local_path=str(source),
        cluster_id=1,
        backend="Custom",
        image_name="test:frontend",
        run_command="python -m frontend",
        env={"GPUSTACK_NON_MODEL_WORKLOAD": "true"},
    )
    parsing = AsyncMock(side_effect=AssertionError("must not inspect model weights"))
    monkeypatch.setattr(scheduler, "evaluate_gguf_model", parsing)
    monkeypatch.setattr(scheduler, "evaluate_pretrained_config", parsing)
    monkeypatch.setattr(evaluator, "get_catalog_spec_by_source_key", parsing)
    result = await evaluator.evaluate_model(
        config, None, model, [worker], [], resource_view=ResourceView([])
    )
    assert result.compatible, result
    assert result.resource_claim.ram == 2 * GIB
    assert result.resource_claim.vram == 0
    estimate.assert_not_awaited()
    parsing.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,repo_field,filename_field,api_method",
    [
        (
            "huggingface",
            "huggingface_repo_id",
            "huggingface_filename",
            "HfApi.list_repo_files",
        ),
        (
            "model_scope",
            "model_scope_model_id",
            "model_scope_file_path",
            "HubApi.get_model_files",
        ),
    ],
)
async def test_non_model_gguf_filename_completion_keeps_zero_vram(
    config, scheduling, monkeypatch, source, repo_field, filename_field, api_method
):
    worker, estimate = scheduling
    filename = "model-Q4_K_M.gguf"
    files = [filename, "tokenizer.json"]
    list_files = MagicMock(
        return_value=files if source == "huggingface" else [{"Path": f} for f in files]
    )
    monkeypatch.setattr(f"gpustack.utils.hub.{api_method}", list_files)
    monkeypatch.setattr(evaluator, "is_repo_cached", lambda *_: True)
    parsing = AsyncMock(side_effect=AssertionError("must not inspect model weights"))
    monkeypatch.setattr(scheduler, "evaluate_gguf_model", parsing)
    monkeypatch.setattr(scheduler, "evaluate_pretrained_config", parsing)
    model = ModelSpec(
        source=source,
        **{repo_field: "test/frontend-GGUF"},
        cluster_id=1,
        backend="Custom",
        image_name="test:frontend",
        run_command="python -m frontend",
        env={"GPUSTACK_NON_MODEL_WORKLOAD": "true"},
    )

    result = await evaluator.evaluate_model(
        config, None, model, [worker], [], resource_view=ResourceView([])
    )

    assert result.compatible, result
    assert getattr(model, filename_field) == filename
    assert result.resource_claim.ram == 2 * GIB
    assert result.resource_claim.vram == 0
    list_files.assert_called_once()
    parsing.assert_not_awaited()
    estimate.assert_not_awaited()


@pytest.mark.asyncio
async def test_instance_analysis_queues_non_model_service_without_loading_model_metadata(
    monkeypatch,
):
    model = deployment()
    instance = new_model_instance(
        1, "frontend", 1, state=ModelInstanceStateEnum.ANALYZING
    )
    instance.created_at = instance.updated_at = datetime.now(timezone.utc) - timedelta(
        minutes=5
    )
    instance_scheduler = scheduler.Scheduler.__new__(scheduler.Scheduler)
    instance_scheduler._queue = SimpleNamespace(put=AsyncMock())
    monkeypatch.setattr(scheduler, "async_session", mock_async_session)
    monkeypatch.setattr(
        scheduler.ModelInstance, "one_by_id", AsyncMock(return_value=instance)
    )
    monkeypatch.setattr(scheduler.Model, "one_by_id", AsyncMock(return_value=model))
    worker_lookup = AsyncMock(side_effect=AssertionError("metadata is not required"))
    monkeypatch.setattr(scheduler.Worker, "all", worker_lookup)
    monkeypatch.setattr(scheduler, "ModelInstanceService", MagicMock())
    await instance_scheduler._evaluate_instance(instance)
    instance_scheduler._queue.put.assert_awaited_once_with(instance)
    worker_lookup.assert_not_awaited()
