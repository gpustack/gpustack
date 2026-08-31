"""Managed cache-service reconciliation.

The controller drives each managed service's CacheServiceInstance rows to
the desired worker set (singleton: the user-picked worker; per_node: every
active worker of the service's cluster, narrowed by the service's
worker_selector labels when set) and folds instance states back into the
service-level aggregate.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.cache_services import (
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.controllers import CacheServiceController


def _provider(topology="singleton") -> CacheProvider:
    return CacheProvider(
        name="LMCache",
        supported_modes=["managed"],
        topology=topology,
        default_version="v1",
        versions={"v1": {"image": "lmcache:v1"}},
    )


def _service(**overrides):
    fields = dict(
        id=9,
        name="svc",
        provider_name="LMCache",
        provider_version="v0.5.2",
        config=None,
        mode=CacheServiceModeEnum.MANAGED,
        cluster_id=1,
        worker_id=5,
        worker_selector=None,
        state=CacheServiceStateEnum.PENDING,
        state_message=None,
        healthy=None,
        deleted_at=None,
        update=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _instance(**overrides):
    fields = dict(
        id=21,
        name="svc-abcde",
        cache_service_id=9,
        worker_id=5,
        cluster_id=1,
        state=CacheServiceStateEnum.PENDING,
        spec_digest=None,
        delete=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _worker(id, cluster_id=1, deleted_at=None, labels=None):
    return SimpleNamespace(
        id=id, cluster_id=cluster_id, deleted_at=deleted_at, labels=labels or {}
    )


def _patch_reconcile(
    monkeypatch,
    provider,
    workers=None,
    worker=None,
    instance_lists=None,
):
    """Back the reconcile lookups. ``instance_lists`` are consecutive
    CacheServiceInstance.all_by_fields results (reconcile pass, then
    aggregate pass)."""
    monkeypatch.setattr(
        "gpustack.server.controllers.get_cache_provider", lambda name: provider
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Worker.all_by_fields",
        AsyncMock(return_value=workers or []),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Worker.one_by_id",
        AsyncMock(return_value=worker),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
        AsyncMock(side_effect=list(instance_lists or [[], []])),
    )
    create = AsyncMock()
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheServiceInstance.create", create
    )
    return create


@pytest.mark.asyncio
async def test_singleton_creates_one_instance_on_picked_worker(monkeypatch):
    service = _service(worker_id=5)
    created_instance = _instance(worker_id=5)
    create = _patch_reconcile(
        monkeypatch,
        _provider("singleton"),
        worker=_worker(5),
        instance_lists=[[], [created_instance]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    create.assert_awaited_once()
    created = create.await_args.args[1]
    assert created.cache_service_id == 9
    assert created.worker_id == 5
    assert created.cluster_id == 1
    assert created.state == CacheServiceStateEnum.PENDING
    # Display name: parent service's name plus a short random suffix,
    # following the model-instance convention.
    assert created.name.startswith("svc-")
    assert len(created.name) == len("svc-") + 5
    # The one PENDING instance keeps the aggregate at PENDING (no write:
    # the service already is PENDING).
    service.update.assert_not_called()


@pytest.mark.asyncio
async def test_per_node_creates_instance_per_active_worker(monkeypatch):
    service = _service(worker_id=None)
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[_worker(5), _worker(6), _worker(7)],
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    assert create.await_count == 3
    assert [call.args[1].worker_id for call in create.await_args_list] == [5, 6, 7]
    assert all(call.args[1].cluster_id == 1 for call in create.await_args_list)
    # Each instance gets its own service-name-prefixed display name.
    names = [call.args[1].name for call in create.await_args_list]
    assert all(name.startswith("svc-") for name in names)
    assert len(set(names)) == 3


@pytest.mark.asyncio
async def test_per_node_only_fills_missing_workers(monkeypatch):
    service = _service(worker_id=None)
    existing = _instance(worker_id=5)
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[_worker(5), _worker(6)],
        instance_lists=[[existing], [existing]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    create.assert_awaited_once()
    assert create.await_args.args[1].worker_id == 6
    existing.delete.assert_not_called()


@pytest.mark.asyncio
async def test_per_node_deletes_instance_of_departed_worker(monkeypatch):
    service = _service(worker_id=None)
    kept = _instance(id=21, worker_id=5, state=CacheServiceStateEnum.RUNNING)
    orphan = _instance(id=22, worker_id=6, state=CacheServiceStateEnum.RUNNING)
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[_worker(5)],
        instance_lists=[[kept, orphan], [kept]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    orphan.delete.assert_awaited_once()
    kept.delete.assert_not_called()
    create.assert_not_called()
    # All remaining instances RUNNING -> the aggregate follows.
    service.update.assert_awaited_once()
    assert service.update.await_args.args[1] == {
        "state": CacheServiceStateEnum.RUNNING,
        "state_message": None,
        "healthy": True,
    }


@pytest.mark.asyncio
async def test_per_node_selector_scopes_to_matching_workers(monkeypatch):
    """Only workers carrying all of the selector's labels get instances."""
    service = _service(worker_id=None, worker_selector={"gpu": "a100"})
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[
            _worker(5, labels={"gpu": "a100"}),
            _worker(6, labels={"gpu": "h100"}),
            _worker(7, labels={"gpu": "a100", "zone": "z1"}),
        ],
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    assert [call.args[1].worker_id for call in create.await_args_list] == [5, 7]


@pytest.mark.asyncio
async def test_per_node_selector_requires_all_labels(monkeypatch):
    """A multi-key selector is an AND: a worker matching only a subset of
    the labels stays out of the desired set."""
    service = _service(worker_id=None, worker_selector={"gpu": "a100", "zone": "z1"})
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[
            _worker(5, labels={"gpu": "a100"}),
            _worker(6, labels={"zone": "z1"}),
            _worker(7, labels={"gpu": "a100", "zone": "z1", "extra": "x"}),
        ],
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    create.assert_awaited_once()
    assert create.await_args.args[1].worker_id == 7


@pytest.mark.asyncio
async def test_per_node_empty_selector_targets_all_workers(monkeypatch):
    service = _service(worker_id=None, worker_selector={})
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[_worker(5), _worker(6, labels={"gpu": "a100"})],
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    assert [call.args[1].worker_id for call in create.await_args_list] == [5, 6]


@pytest.mark.asyncio
async def test_per_node_selector_change_moves_instances(monkeypatch):
    """A service-row selector edit reconciles like a worker change: the
    now-unmatched worker's instance is deleted and the newly matched
    worker gets one."""
    service = _service(worker_id=None, worker_selector={"gpu": "h100"})
    outdated = _instance(id=21, worker_id=5, state=CacheServiceStateEnum.RUNNING)
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[
            _worker(5, labels={"gpu": "a100"}),
            _worker(6, labels={"gpu": "h100"}),
        ],
        instance_lists=[[outdated], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    outdated.delete.assert_awaited_once()
    create.assert_awaited_once()
    assert create.await_args.args[1].worker_id == 6


@pytest.mark.asyncio
async def test_per_node_selector_matching_no_worker_parks_service_in_error(
    monkeypatch,
):
    service = _service(worker_id=None, worker_selector={"gpu": "b200"})
    orphan = _instance(worker_id=5, state=CacheServiceStateEnum.RUNNING)
    create = _patch_reconcile(
        monkeypatch,
        _provider("per_node"),
        workers=[_worker(5, labels={"gpu": "a100"})],
        instance_lists=[[orphan], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    # A selector matching nothing is an authoritative empty set: labels
    # change only by explicit edits, so the instances follow (the selector
    # can scale the service to zero) and the ERROR message says why.
    orphan.delete.assert_awaited_once()
    create.assert_not_called()
    service.update.assert_awaited_once()
    updated = service.update.await_args.args[1]
    assert updated["state"] == CacheServiceStateEnum.ERROR
    assert "No workers match the worker selector" in updated["state_message"]
    assert updated["healthy"] is False


@pytest.mark.asyncio
async def test_singleton_missing_worker_parks_service_in_error(monkeypatch):
    service = _service(worker_id=5)
    orphan = _instance(worker_id=5)
    _patch_reconcile(
        monkeypatch,
        _provider("singleton"),
        worker=None,
        instance_lists=[[orphan], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    orphan.delete.assert_not_awaited()
    service.update.assert_awaited_once()
    assert service.update.await_args.args[1] == {
        "state": CacheServiceStateEnum.ERROR,
        "state_message": "Assigned worker no longer exists.",
        "healthy": False,
    }


@pytest.mark.asyncio
async def test_singleton_rejects_worker_from_other_cluster(monkeypatch):
    service = _service(worker_id=5, cluster_id=1)
    _patch_reconcile(
        monkeypatch,
        _provider("singleton"),
        worker=_worker(5, cluster_id=2),
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    service.update.assert_awaited_once()
    assert service.update.await_args.args[1]["state"] == CacheServiceStateEnum.ERROR


# ---- aggregate transitions ----


def _patch_aggregate_instances(monkeypatch, instances):
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
        AsyncMock(return_value=instances),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "states, expected",
    [
        (
            [CacheServiceStateEnum.RUNNING, CacheServiceStateEnum.RUNNING],
            {
                "state": CacheServiceStateEnum.RUNNING,
                "state_message": None,
                "healthy": True,
            },
        ),
        (
            [CacheServiceStateEnum.RUNNING, CacheServiceStateEnum.ERROR],
            {
                "state": CacheServiceStateEnum.RUNNING,
                "state_message": "1/2 instances running",
                "healthy": False,
            },
        ),
        (
            [CacheServiceStateEnum.PENDING, CacheServiceStateEnum.STARTING],
            {
                "state": CacheServiceStateEnum.STARTING,
                "state_message": None,
                "healthy": None,
            },
        ),
        (
            [CacheServiceStateEnum.PENDING, CacheServiceStateEnum.PENDING],
            {
                "state": CacheServiceStateEnum.PENDING,
                "state_message": None,
                "healthy": None,
            },
        ),
        (
            [CacheServiceStateEnum.ERROR, CacheServiceStateEnum.UNREACHABLE],
            {
                "state": CacheServiceStateEnum.ERROR,
                "state_message": "0/2 instances running",
                "healthy": False,
            },
        ),
        (
            [],
            {
                "state": CacheServiceStateEnum.ERROR,
                "state_message": "no instances running",
                "healthy": False,
            },
        ),
    ],
)
async def test_aggregate_transitions(monkeypatch, states, expected):
    service = _service(state=CacheServiceStateEnum.UNREACHABLE)
    _patch_aggregate_instances(
        monkeypatch,
        [_instance(id=21 + i, worker_id=5 + i, state=s) for i, s in enumerate(states)],
    )

    controller = CacheServiceController(MagicMock())
    await controller._sync_service_aggregate(MagicMock(), service)

    service.update.assert_awaited_once()
    assert service.update.await_args.args[1] == expected


@pytest.mark.asyncio
async def test_aggregate_writes_only_on_change(monkeypatch):
    service = _service(
        state=CacheServiceStateEnum.RUNNING, state_message=None, healthy=True
    )
    _patch_aggregate_instances(
        monkeypatch, [_instance(state=CacheServiceStateEnum.RUNNING)]
    )

    controller = CacheServiceController(MagicMock())
    await controller._sync_service_aggregate(MagicMock(), service)

    service.update.assert_not_called()


# ---- instance-event fan-in ----


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


async def _run_instance_event(monkeypatch, service, event_type):
    """Drive one instance event through the watch body by faking the
    subscription stream; returns the (reconcile, aggregate) mocks."""
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheService.one_by_id",
        AsyncMock(return_value=service),
    )
    controller = CacheServiceController(MagicMock())
    reconcile = AsyncMock()
    aggregate = AsyncMock()
    monkeypatch.setattr(controller, "_reconcile_service", reconcile)
    monkeypatch.setattr(controller, "_sync_service_aggregate", aggregate)

    from gpustack.server.bus import Event

    async def fake_subscribe(**kwargs):
        yield Event(type=event_type, data=_instance())

    with patch(
        "gpustack.server.controllers.CacheServiceInstance.subscribe",
        side_effect=lambda **kwargs: fake_subscribe(**kwargs),
    ):
        await controller._watch_instances()

    return reconcile, aggregate


@pytest.mark.asyncio
async def test_instance_event_skips_external_and_deleted_services(monkeypatch):
    from gpustack.server.bus import EventType

    external = _service(mode=CacheServiceModeEnum.EXTERNAL)
    reconcile, aggregate = await _run_instance_event(
        monkeypatch, external, EventType.UPDATED
    )

    reconcile.assert_not_called()
    aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_instance_update_event_only_syncs_aggregate(monkeypatch):
    from gpustack.server.bus import EventType

    service = _service()
    reconcile, aggregate = await _run_instance_event(
        monkeypatch, service, EventType.UPDATED
    )

    aggregate.assert_awaited_once()
    assert aggregate.await_args.args[1] is service
    reconcile.assert_not_called()


@pytest.mark.asyncio
async def test_instance_delete_event_reconciles_parent_service(monkeypatch):
    """An instance deletion reconciles the whole parent service, so a
    deleted instance whose worker is still in the desired set is
    replaced with a fresh PENDING row right away."""
    from gpustack.server.bus import EventType

    service = _service()
    reconcile, aggregate = await _run_instance_event(
        monkeypatch, service, EventType.DELETED
    )

    reconcile.assert_awaited_once()
    assert reconcile.await_args.args[1] is service
    aggregate.assert_not_called()


def _attached_model(model_id=3, service_id=9):
    from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum

    return SimpleNamespace(
        id=model_id,
        deleted_at=None,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True,
            mode=KVCacheModeEnum.SHARED,
            cache_service_id=service_id,
        ),
    )


def _model_instance(state, injected=False, reason="not ready", worker_id=5):
    from gpustack.schemas.models import CacheConfigSnapshot

    return SimpleNamespace(
        id=31,
        model_id=3,
        worker_id=worker_id,
        state=state,
        spans_workers=False,
        cache_config=CacheConfigSnapshot(
            cache_service_id=9, injected=injected, reason=reason
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        # every state whose engine has not consumed the snapshot yet: the
        # serve process re-reads the row at the STARTING transition
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
    ],
)
async def test_refresh_rewrites_snapshot_before_engine_start(state):
    """A cache instance turning RUNNING re-resolves degraded snapshots of
    instances whose engine has not started yet — closing the
    create-service-then-model window."""
    from gpustack.schemas.models import CacheConfigSnapshot

    mi = _model_instance(state)
    fresh = CacheConfigSnapshot(cache_service_id=9, injected=True)
    update_mock = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.all_by_fields",
            AsyncMock(return_value=[_attached_model()]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_fields",
            AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(
                return_value=[SimpleNamespace(id=5, ip="10.0.0.5", deleted_at=None)]
            ),
        ),
        patch(
            "gpustack.server.controllers.resolve_instance_cache_config_safe",
            AsyncMock(return_value=fresh),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            lambda session: SimpleNamespace(update=update_mock),
        ),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._refresh_attached_snapshots(MagicMock(), _service())

    assert mi.cache_config is fresh
    update_mock.assert_awaited_once_with(mi)


@pytest.mark.asyncio
async def test_refresh_hints_running_degraded_instance_without_touching_config():
    """A RUNNING engine's snapshot records what it actually started with:
    the refresher must not flip injected, only add a restart hint when a
    fresh resolve would now attach."""
    from gpustack.schemas.models import CacheConfigSnapshot

    mi = _model_instance(ModelInstanceStateEnum.RUNNING, reason="was not ready")
    fresh = CacheConfigSnapshot(cache_service_id=9, injected=True)
    update_mock = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.all_by_fields",
            AsyncMock(return_value=[_attached_model()]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_fields",
            AsyncMock(return_value=[mi]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(
                return_value=[SimpleNamespace(id=5, ip="10.0.0.5", deleted_at=None)]
            ),
        ),
        patch(
            "gpustack.server.controllers.resolve_instance_cache_config_safe",
            AsyncMock(return_value=fresh),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            lambda session: SimpleNamespace(update=update_mock),
        ),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._refresh_attached_snapshots(MagicMock(), _service())
        # Second pass is idempotent: the hint is only appended once.
        await controller._refresh_attached_snapshots(MagicMock(), _service())

    assert mi.cache_config.injected is False
    assert mi.cache_config.reason.startswith("was not ready; ")
    assert "restart the instance to attach" in mi.cache_config.reason
    assert mi.cache_config.reason.count("restart the instance to attach") == 1
    update_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_skips_takeover_and_starting_instances():
    """No hint when a fresh resolve stays degraded (e.g. user takeover —
    a restart would not fix it), and no rewrite once the engine is
    launching (STARTING races the container create)."""
    from gpustack.schemas.models import CacheConfigSnapshot

    running = _model_instance(ModelInstanceStateEnum.RUNNING, reason="taken over")
    starting = _model_instance(ModelInstanceStateEnum.STARTING)
    still_degraded = CacheConfigSnapshot(
        cache_service_id=9, injected=False, reason="taken over"
    )
    update_mock = AsyncMock()
    with (
        patch(
            "gpustack.server.controllers.Model.all_by_fields",
            AsyncMock(return_value=[_attached_model()]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_fields",
            AsyncMock(return_value=[running, starting]),
        ),
        patch(
            "gpustack.server.controllers.Worker.all_by_fields",
            AsyncMock(
                return_value=[SimpleNamespace(id=5, ip="10.0.0.5", deleted_at=None)]
            ),
        ),
        patch(
            "gpustack.server.controllers.resolve_instance_cache_config_safe",
            AsyncMock(return_value=still_degraded),
        ),
        patch(
            "gpustack.server.controllers.ModelInstanceService",
            lambda session: SimpleNamespace(update=update_mock),
        ),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._refresh_attached_snapshots(MagicMock(), _service())

    update_mock.assert_not_awaited()
    assert "restart" not in (running.cache_config.reason or "")


@pytest.mark.asyncio
async def test_refresh_tracks_endpoint_liveness_on_running_instance():
    """An engine that started attached keeps its snapshot (the record of
    its startup config), but endpoint_live tracks the present: off when
    the recorded endpoint stops being what a fresh resolve yields (cache
    gone, or moved to another port), and back on upon recovery — so
    "attached" indicators never report a cache that is not there."""
    from gpustack.schemas.cache_services import (
        CacheConfigSnapshot,
        CacheServiceEndpoint,
    )

    endpoint = CacheServiceEndpoint(host="10.0.0.5", port=9000)
    mi = SimpleNamespace(
        id=31,
        model_id=3,
        worker_id=5,
        state=ModelInstanceStateEnum.RUNNING,
        spans_workers=False,
        cache_config=CacheConfigSnapshot(
            cache_service_id=9,
            injected=True,
            endpoint=endpoint,
            env={"PYTHONHASHSEED": "0"},
        ),
    )
    resolve_mock = AsyncMock()
    update_mock = AsyncMock()

    async def run_pass():
        with (
            patch(
                "gpustack.server.controllers.Model.all_by_fields",
                AsyncMock(return_value=[_attached_model()]),
            ),
            patch(
                "gpustack.server.controllers.ModelInstance.all_by_fields",
                AsyncMock(return_value=[mi]),
            ),
            patch(
                "gpustack.server.controllers.Worker.all_by_fields",
                AsyncMock(
                    return_value=[SimpleNamespace(id=5, ip="10.0.0.5", deleted_at=None)]
                ),
            ),
            patch(
                "gpustack.server.controllers.resolve_instance_cache_config_safe",
                resolve_mock,
            ),
            patch(
                "gpustack.server.controllers.ModelInstanceService",
                lambda session: SimpleNamespace(update=update_mock),
            ),
        ):
            controller = CacheServiceController(MagicMock())
            await controller._refresh_attached_snapshots(MagicMock(), _service())

    # cache gone: a fresh resolve degrades -> liveness flips off, the
    # startup record (env/endpoint/injected) stays untouched
    resolve_mock.return_value = CacheConfigSnapshot(
        cache_service_id=9, injected=False, reason="gone"
    )
    await run_pass()
    assert mi.cache_config.injected is True
    assert mi.cache_config.endpoint_live is False
    assert mi.cache_config.env == {"PYTHONHASHSEED": "0"}
    assert mi.cache_config.endpoint == endpoint

    # cache moved to another port: attachable again, but not at the
    # endpoint this engine started with -> stays off
    resolve_mock.return_value = CacheConfigSnapshot(
        cache_service_id=9,
        injected=True,
        endpoint=CacheServiceEndpoint(host="10.0.0.5", port=9100),
    )
    await run_pass()
    assert mi.cache_config.endpoint_live is False

    # cache back on the recorded endpoint -> recovers
    resolve_mock.return_value = CacheConfigSnapshot(
        cache_service_id=9, injected=True, endpoint=endpoint
    )
    await run_pass()
    assert mi.cache_config.endpoint_live is True
    assert update_mock.await_count == 2  # unchanged states write nothing


@pytest.mark.asyncio
async def test_aggregate_flags_spec_drift():
    """A spec edit leaves running containers untouched by design, so the
    aggregate must say so: instances created from an older spec flag the
    service; pre-digest rows (None) never do."""
    from gpustack.schemas.cache_services import cache_service_spec_digest

    service = _service(update=AsyncMock())
    stale = _instance(state=CacheServiceStateEnum.RUNNING, spec_digest="0" * 16)
    with patch(
        "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
        AsyncMock(return_value=[stale]),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._sync_service_aggregate(MagicMock(), service)

    args = service.update.await_args.args[1]
    assert args["state"] == CacheServiceStateEnum.RUNNING
    assert "configuration changed" in args["state_message"]

    # a fresh instance (current digest) and a pre-digest row are clean
    service2 = _service(update=AsyncMock())
    current = _instance(
        state=CacheServiceStateEnum.RUNNING,
        spec_digest=cache_service_spec_digest(service2),
    )
    legacy = _instance(
        id=22, worker_id=6, state=CacheServiceStateEnum.RUNNING, spec_digest=None
    )
    with patch(
        "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
        AsyncMock(return_value=[current, legacy]),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._sync_service_aggregate(MagicMock(), service2)

    args = service2.update.await_args.args[1]
    assert args["state_message"] is None
