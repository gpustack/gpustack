from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from gpustack.scheduler import evaluator
from gpustack.policies import utils as policy_utils
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import DisaggregationSpec, RoleSpec, SourceEnum
from gpustack.schemas.port_reservations import WorkerPortAvailability
from gpustack.schemas.workers import CPUInfo, WorkerStateEnum
from tests.fixtures.workers.fixtures import linux_nvidia_22_H100_80gx8
from tests.utils.mock import mock_async_session


def mock_evaluation_inventory(monkeypatch, workers):
    monkeypatch.setattr(
        evaluator.Worker, "all_by_fields", AsyncMock(return_value=workers)
    )
    monkeypatch.setattr(
        evaluator.ModelInstance, "all_by_fields", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        evaluator.scheduler,
        "prioritize_workers_with_model_files",
        AsyncMock(return_value=workers),
    )
    monkeypatch.setattr(evaluator, "cache_instances_in", AsyncMock(return_value=[]))
    monkeypatch.setattr(evaluator, "evaluate_cache", {})
    monkeypatch.setattr(evaluator, "evaluate_model_limiter", evaluator.AsyncLimiter(1))


@pytest.mark.asyncio
@pytest.mark.parametrize("pd", [False, True])
async def test_opt_out_keeps_preview_defaults_and_input_validation(monkeypatch, pd):
    model = ModelSpec(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        env={"GPUSTACK_SKIP_MODEL_EVALUATION": "1"},
        roles=(
            [RoleSpec(name=role) for role in ("prefill", "decode", "router")]
            if pd
            else None
        ),
        disaggregation=DisaggregationSpec(mode="vllm-nixl") if pd else None,
    )
    params = ["--port=18000"]
    catalog_spec = model.model_copy(update={"backend_parameters": params})
    mock_evaluation_inventory(monkeypatch, [])
    monkeypatch.setattr(
        evaluator,
        "get_catalog_spec_by_source_key",
        AsyncMock(return_value=catalog_spec),
    )
    monkeypatch.setattr(evaluator, "set_gguf_model_file_path", AsyncMock())
    metadata = AsyncMock()
    monkeypatch.setattr(evaluator, "evaluate_model_metadata", metadata)

    (result,) = await evaluator.evaluate_models(None, None, [model])

    assert result.default_spec.backend_parameters == params
    assert not result.compatible
    assert any("--port" in msg for msg in result.compatibility_messages)
    metadata.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('fits', [False, True])
@pytest.mark.parametrize('tolerate_metadata_errors', [False, True])
@pytest.mark.parametrize(
    'mode,pd',
    [
        ('internal', False),
        ('external', False),
        ('hybrid', False),
        ('multi-port-external', False),
        ('external', True),
        ('hybrid', True),
    ],
)
async def test_unsaved_spec_completes_real_resource_evaluation(
    config, monkeypatch, tmp_path, mode, pd, fits, tolerate_metadata_errors
):
    from gpustack.scheduler import port_budget

    fixtures = Path(__file__).parents[1] / 'fixtures' / 'model_configs'
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    shutil.copyfile(fixtures / 'qwen3_moe.json', tmp_path / 'config.json')
    worker = linux_nvidia_22_H100_80gx8()
    worker.status.cpu = CPUInfo(total=16)
    if not fits:
        worker.status.gpu_devices = worker.status.gpu_devices[:1]
    worker.cluster_id = 1
    worker.state = WorkerStateEnum.READY
    worker.system_principal_id = 101
    parameters = [
        '--distributed-executor-backend=mp',
        '--data-parallel-backend=mp',
        '--tensor-parallel-size=1',
        '--data-parallel-size=2',
    ]
    if mode != 'internal':
        parameters.append(f'--data-parallel-{mode}-lb')
    if mode == 'hybrid':
        parameters.append('--data-parallel-size-local=2')
    model = ModelSpec(
        source=SourceEnum.LOCAL_PATH,
        local_path=str(tmp_path),
        backend='vLLM',
        backend_version='0.32.0',
        cluster_id=1,
        env=(
            {'GPUSTACK_SKIP_MODEL_EVALUATION': '0'}
            if tolerate_metadata_errors
            else None
        ),
        backend_parameters=[] if pd else parameters,
        roles=(
            [
                RoleSpec(name='prefill', backend_parameters=parameters),
                RoleSpec(name='decode', backend_parameters=parameters),
                RoleSpec(name='router'),
            ]
            if pd
            else None
        ),
        disaggregation=DisaggregationSpec(mode='vllm-nixl') if pd else None,
    )
    monkeypatch.setattr(
        evaluator, 'get_catalog_spec_by_source_key', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        evaluator.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(topology=None)),
    )
    monkeypatch.setattr(
        'gpustack.validation.pd.runtime.cluster_vendors',
        AsyncMock(return_value=['nvidia']),
    )
    monkeypatch.setattr(
        policy_utils,
        'get_local_model_weight_size',
        AsyncMock(return_value=20 * 1024**3),
    )
    monkeypatch.setattr(
        'gpustack.policies.worker_filters.backend_framework_filter.async_session',
        mock_async_session,
    )
    monkeypatch.setattr(
        'gpustack.policies.worker_filters.backend_framework_filter.BackendFrameworkFilter._has_supported_runners',
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr('gpustack.envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE', 0)
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(
            return_value={
                worker.id: WorkerPortAvailability(
                    worker_id=worker.id,
                    generation=uuid4(),
                    port_range='40000-40255',
                )
            }
        ),
    )

    mock_evaluation_inventory(monkeypatch, [worker])
    (result,) = await evaluator.evaluate_models(config, None, [model], cluster_id=1)

    assert not result.error, result.error_message
    assert result.compatible == fits, (
        result.compatibility_messages,
        result.scheduling_messages,
    )
    assert not hasattr(model, 'id')
    assert model.name is None
    if not fits:
        assert result.resource_claim is None
        assert result.scheduling_messages
        assert all('has no attribute' not in msg for msg in result.scheduling_messages)
        return
    assert result.resource_claim is not None
    assert result.resource_claim.vram > 0
    if pd:
        assert {claim.role for claim in result.role_resource_claims} == {
            'prefill',
            'decode',
            'router',
        }


@pytest.mark.asyncio
async def test_cached_evaluation_uses_the_same_reservations_as_its_cache_key(
    monkeypatch,
):
    first = [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 10})]
    changed = [SimpleNamespace(worker_id=1, computed_resource_claim={"ram": 20})]
    model = ModelSpec(source=SourceEnum.LOCAL_PATH, local_path="/models/model")
    evaluate = AsyncMock(return_value=evaluator.ModelEvaluationResult())
    monkeypatch.setattr(evaluator, "evaluate_cache", {})
    monkeypatch.setattr(
        evaluator, "visible_cache_service", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        evaluator,
        "cache_instances_in",
        AsyncMock(
            side_effect=[first, first, changed],
        ),
    )
    monkeypatch.setattr(evaluator, "evaluate_model", evaluate)

    workers = [SimpleNamespace(id=1, model_dump=lambda **kwargs: {"id": 1})]
    for _ in range(3):
        await evaluator.evaluate_model_with_cache(
            SimpleNamespace(), None, model, workers, []
        )

    assert evaluator.cache_instances_in.call_args.kwargs["worker_ids"] == [1]
    assert evaluate.await_count == 2
    for call, rows, ram in zip(evaluate.await_args_list, [first, changed], [10, 20]):
        assert call.kwargs["cache_instances"] is rows
        assert call.kwargs["resource_view"].allocated(1).ram == ram


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster_id", [None, 3])
async def test_cache_reservations_are_scoped_to_evaluation_workers(
    monkeypatch, cluster_id
):
    from gpustack.scheduler.group_schedule import cache_instances_in
    from gpustack.schemas.cache_services import CacheServiceInstance

    rows = AsyncMock(return_value=[])
    monkeypatch.setattr(CacheServiceInstance, "all_by_fields", rows)
    await cache_instances_in(object(), cluster_id, worker_ids=[1, 2])
    rows.assert_awaited_once()
    assert rows.call_args.args[1] == ({"cluster_id": 3} if cluster_id else {})
    condition = rows.call_args.kwargs["extra_conditions"][0]
    assert next(iter(condition.compile().params.values())) == [1, 2]
    rows.reset_mock()
    assert await cache_instances_in(object(), cluster_id, worker_ids=[]) == []
    rows.assert_not_awaited()
