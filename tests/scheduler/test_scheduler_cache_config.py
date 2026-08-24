"""Schedule-time shared-cache re-resolution.

When an instance is assigned its worker, the scheduler re-resolves the
model's shared-cache snapshot with that worker, so worker-dependent
injection (e.g. the client's own local_hostname) binds to the instance's
node. Models without a shared cache keep their create-time snapshot
untouched.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.scheduler.scheduler import Scheduler
from gpustack.schemas.cache_services import CacheConfigSnapshot
from gpustack.schemas.models import (
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    ModelInstanceStateEnum,
)
from tests.utils.model import new_model


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _model(extended_kv_cache=None):
    return new_model(
        1,
        "m",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        extended_kv_cache=extended_kv_cache,
    )


def _shared_cache_model():
    return _model(
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=5
        )
    )


def _instance():
    # the scheduler assigns distributed_servers then reads the real
    # ModelInstance.spans_workers property; the fake mirrors the
    # single-worker placement (_candidate has no subordinate workers)
    return SimpleNamespace(
        id=11,
        name="m-1",
        model_id=1,
        state=ModelInstanceStateEnum.PENDING,
        state_message="",
        worker_id=None,
        spans_workers=False,
        cache_config="unchanged-sentinel",
    )


def _candidate(worker):
    return SimpleNamespace(
        worker=worker,
        computed_resource_claim=None,
        gpu_type=None,
        gpu_indexes=None,
        gpu_addresses=None,
        subordinate_workers=None,
    )


def _schedule_patches(model, model_instance, candidate, resolve_mock, service_mock):
    worker = candidate.worker
    return (
        patch(
            "gpustack.scheduler.scheduler.async_session",
            lambda: _FakeSessionCtx(),
        ),
        patch(
            "gpustack.scheduler.scheduler.Worker.all",
            AsyncMock(return_value=[worker]),
        ),
        patch(
            "gpustack.scheduler.scheduler.Model.one_by_id",
            AsyncMock(return_value=model),
        ),
        patch(
            "gpustack.scheduler.scheduler.ModelInstance.one_by_id",
            AsyncMock(return_value=model_instance),
        ),
        patch(
            "gpustack.scheduler.scheduler.ModelInstance.all",
            AsyncMock(return_value=[]),
        ),
        patch(
            "gpustack.scheduler.scheduler.find_candidate",
            AsyncMock(return_value=(candidate, [])),
        ),
        patch(
            "gpustack.scheduler.scheduler.resolve_instance_cache_config_safe",
            resolve_mock,
        ),
        patch("gpustack.scheduler.scheduler.ModelInstanceService", service_mock),
    )


def _worker():
    return SimpleNamespace(
        id=7,
        name="node-a",
        ip="10.0.0.7",
        advertise_address="10.0.0.7",
        ifname="eth0",
    )


@pytest.mark.asyncio
async def test_schedule_assignment_reresolves_shared_cache_config():
    scheduler = Scheduler(SimpleNamespace(cache_dir=None))
    model = _shared_cache_model()
    model_instance = _instance()
    worker = _worker()
    candidate = _candidate(worker)
    snapshot = CacheConfigSnapshot(cache_service_id=5, injected=True)
    resolve_mock = AsyncMock(return_value=snapshot)
    service_mock = MagicMock()
    service_mock.return_value.update = AsyncMock()

    with ExitStack() as stack:
        for ctx in _schedule_patches(
            model, model_instance, candidate, resolve_mock, service_mock
        ):
            stack.enter_context(ctx)
        await scheduler._schedule_one(SimpleNamespace(id=11, name="m-1", model_id=1))

    resolve_mock.assert_awaited_once()
    assert resolve_mock.await_args.args[1] is model
    assert resolve_mock.await_args.kwargs["worker"] is worker
    assert model_instance.cache_config is snapshot
    assert model_instance.worker_id == 7
    assert model_instance.state == ModelInstanceStateEnum.SCHEDULED
    service_mock.return_value.update.assert_awaited_once_with(model_instance)


@pytest.mark.asyncio
async def test_schedule_assignment_skips_models_without_shared_cache():
    scheduler = Scheduler(SimpleNamespace(cache_dir=None))
    model = _model()
    model_instance = _instance()
    candidate = _candidate(_worker())
    resolve_mock = AsyncMock()
    service_mock = MagicMock()
    service_mock.return_value.update = AsyncMock()

    with ExitStack() as stack:
        for ctx in _schedule_patches(
            model, model_instance, candidate, resolve_mock, service_mock
        ):
            stack.enter_context(ctx)
        await scheduler._schedule_one(SimpleNamespace(id=11, name="m-1", model_id=1))

    resolve_mock.assert_not_called()
    assert model_instance.cache_config == "unchanged-sentinel"
    assert model_instance.state == ModelInstanceStateEnum.SCHEDULED
