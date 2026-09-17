"""Managed cache-service reconciliation.

The controller drives each managed service's CacheServiceInstance rows to
the desired worker set (replicas: pinned or scheduler-placed; per_node: every
active worker of the service's cluster, narrowed by the service's
worker_selector labels when set) and folds instance states back into the
service-level aggregate.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.server.cache_provider_catalog import asset_providers
from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.cache_services import (
    CacheServiceConfig,
    CacheServiceStateEnum,
)
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server import controllers as cache_service_controller
from gpustack.server.controllers import CacheServiceController


@pytest.fixture(autouse=True)
def catalog_lookup(monkeypatch):
    """The catalog is a table, and these tests hand their code a mock session.
    Default to what this installation carries — the packaged declarations are
    what a cluster serves with no document configured — and let a test install
    a declaration of its own over it."""

    async def lookup(_session, name=None):
        wanted = (name or "").lower()
        return next(
            (
                provider
                for provider in asset_providers()
                if provider.name.lower() == wanted
            ),
            None,
        )

    for target in ("gpustack.server.controllers.get_cache_provider",):
        monkeypatch.setattr(target, lookup)


def _fake_lookup(provider):
    """Stand in for the catalog lookup, which reads a table: a coroutine taking
    the session its caller holds. Whatever a test's service names, it resolves
    to the declaration that test pinned."""

    async def lookup(_session, _name=None):
        return provider

    return lookup


def _provider(topology="replicas") -> CacheProvider:
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
        component="",
        component_addresses=None,
        state=CacheServiceStateEnum.PENDING,
        spec_digest=None,
        delete=AsyncMock(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _worker(id, cluster_id=1, deleted_at=None, labels=None, ip=None):
    return SimpleNamespace(
        id=id,
        cluster_id=cluster_id,
        deleted_at=deleted_at,
        labels=labels or {},
        ip=ip or f"10.0.0.{id}",
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
        "gpustack.server.controllers.get_cache_provider", _fake_lookup(provider)
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
async def test_replicas_pins_one_instance_on_picked_worker(monkeypatch):
    service = _service(worker_id=5)
    created_instance = _instance(worker_id=5)
    create = _patch_reconcile(
        monkeypatch,
        _provider("replicas"),
        workers=[_worker(5)],
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


def _pool_provider() -> CacheProvider:
    from gpustack.schemas.cache_providers import CacheProviderComponent

    return CacheProvider(
        name="Pool",
        supported_modes=["managed"],
        default_image="repo/pool:{{version}}",
        versions={"v1.0": {}},
        components={
            "master": CacheProviderComponent(
                run_command="pool-master --port {{port}}",
                attach_endpoint=True,
                metrics_port="metrics",
                gpu_access=False,
            ),
            "store": CacheProviderComponent(
                topology="per_node",
                depends_on="master",
                run_command="pool-store --port {{port}}",
                gpu_access=False,
            ),
        },
    )


def _store_pool_provider(replicas: int) -> CacheProvider:
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = _pool_provider()
    provider.components["store"] = CacheProviderComponent(
        topology="replicas",
        replicas=replicas,
        depends_on="master",
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    return provider


@pytest.mark.asyncio
async def test_replicas_run_what_fits_when_the_cluster_is_smaller(monkeypatch):
    """A component's instances share what the node holds for them — a
    data directory, a device — so a cluster smaller than the replica
    count runs one per worker and stops, rather than stacking two that
    would collide."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    store = _instance(
        id=22,
        worker_id=5,
        component="store",
        state=CacheServiceStateEnum.RUNNING,
        component_addresses={"master": "10.0.0.5:50051"},
    )
    create = _patch_reconcile(
        monkeypatch,
        _store_pool_provider(3),
        workers=[_worker(5, ip="10.0.0.5")],
        worker=_worker(5, ip="10.0.0.5"),
        instance_lists=[[master, store], [master, store]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    created = [call.args[1] for call in create.await_args_list]
    stores = [row for row in created if row.component == "store"]
    # the one worker already holds a store; the other two replicas have
    # nowhere of their own to go
    assert stores == []
    assert store.delete.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,expected",
    [({}, 1), ({"enable_ha": True, "master_replicas": 3}, 3)],
)
async def test_replica_sizing_follows_the_fields_gate(monkeypatch, fields, expected):
    """A count offered only with a feature resolves to its gated default
    while the feature is off, so the master runs alone until HA is on."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderField,
    )

    provider = _pool_provider()
    provider.fields = [
        CacheProviderField(name="enable_ha", type="boolean", default=False),
        CacheProviderField(
            name="master_replicas",
            type="number",
            default=3,
            gated_default=1,
            visible_by="enable_ha",
            visible_when=True,
        ),
    ]
    provider.components["master"] = CacheProviderComponent(
        replicas_by="master_replicas",
        attach_endpoint=True,
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    del provider.components["store"]
    service = _service(worker_id=None, config=CacheServiceConfig(fields=fields))
    create = _patch_reconcile(
        monkeypatch,
        provider,
        workers=[_worker(5), _worker(6), _worker(7)],
        worker=_worker(5),
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    assert create.await_count == expected


@pytest.mark.asyncio
async def test_dependents_address_a_pool_through_its_declared_template(monkeypatch):
    """A dependency that runs several replicas has no single endpoint, so
    the dependent is stamped with the component's rendered address
    instead of whichever replica was found running."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderField,
    )

    provider = _pool_provider()
    provider.fields = [
        CacheProviderField(name="backend", default=""),
    ]
    provider.components["master"] = CacheProviderComponent(
        replicas=3,
        attach_endpoint=True,
        address_template="etcd://{{backend}}",
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    service = _service(
        worker_id=None, config=CacheServiceConfig(fields={"backend": "10.0.0.3:2379"})
    )
    masters = [
        _instance(
            id=21 + offset,
            worker_id=5 + offset,
            component="master",
            state=CacheServiceStateEnum.RUNNING,
            port=50051,
        )
        for offset in range(3)
    ]
    create = _patch_reconcile(
        monkeypatch,
        provider,
        workers=[_worker(5), _worker(6), _worker(7)],
        worker=_worker(5),
        instance_lists=[masters, masters],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    stores = [
        row
        for row in (call.args[1] for call in create.await_args_list)
        if row.component == "store"
    ]
    assert stores
    assert all(
        row.component_addresses == {"master": "etcd://10.0.0.3:2379"} for row in stores
    )


@pytest.mark.asyncio
async def test_a_disabled_dependency_does_not_hold_a_dependent_back(monkeypatch):
    """A dependency a field turned off will never run, so waiting for its
    address would strand the dependent forever."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderField,
    )

    provider = _pool_provider()
    provider.fields = [
        CacheProviderField(name="enable_extra", type="boolean", default=False),
    ]
    provider.components["master"] = CacheProviderComponent(
        enabled_by="enable_extra",
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    provider.components["store"] = CacheProviderComponent(
        topology="per_node",
        depends_on="master",
        attach_endpoint=True,
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    provider.attach_locality = "node_local"
    service = _service(worker_id=None, config=CacheServiceConfig(fields={}))
    create = _patch_reconcile(
        monkeypatch,
        provider,
        workers=[_worker(5)],
        worker=_worker(5),
        instance_lists=[[], []],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    created = [call.args[1] for call in create.await_args_list]
    assert [row.component for row in created] == ["store"]
    # Nothing to stamp: the dependency does not exist to be addressed.
    assert created[0].component_addresses is None


@pytest.mark.asyncio
async def test_replicas_spread_before_stacking(monkeypatch):
    """Two workers take one replica each before either takes a second."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    create = _patch_reconcile(
        monkeypatch,
        _store_pool_provider(3),
        workers=[_worker(5, ip="10.0.0.5"), _worker(6, ip="10.0.0.6")],
        worker=_worker(5, ip="10.0.0.5"),
        instance_lists=[[master], [master]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    placements = sorted(
        row.worker_id
        for row in (call.args[1] for call in create.await_args_list)
        if row.component == "store"
    )
    # two workers take one store each; the third replica waits for a
    # worker of its own rather than doubling up on one of them
    assert placements == [5, 6]


@pytest.mark.asyncio
async def test_lowered_replica_count_deletes_the_surplus(monkeypatch):
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    addresses = {"master": "10.0.0.5:50051"}
    stores = [
        _instance(
            id=22 + offset,
            worker_id=5,
            component="store",
            state=CacheServiceStateEnum.RUNNING,
            component_addresses=addresses,
        )
        for offset in range(3)
    ]
    _patch_reconcile(
        monkeypatch,
        _store_pool_provider(1),
        workers=[_worker(5, ip="10.0.0.5")],
        worker=_worker(5, ip="10.0.0.5"),
        instance_lists=[[master, *stores], [master, stores[0]]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    assert [store.delete.await_count for store in stores] == [0, 1, 1]


@pytest.mark.asyncio
async def test_dependent_component_gets_stamped_dependency_address(monkeypatch):
    """Stores are created only once the master runs with a known port,
    and carry its resolved host:port — the running process bakes the
    address into its config, so the controller stamps it at creation."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    create = _patch_reconcile(
        monkeypatch,
        _pool_provider(),
        workers=[_worker(5), _worker(6)],
        worker=_worker(5),
        instance_lists=[[master], [master]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    created = [call.args[1] for call in create.await_args_list]
    stores = [row for row in created if row.component == "store"]
    assert {row.worker_id for row in stores} == {5, 6}
    assert all(
        row.component_addresses == {"master": "10.0.0.5:50051"} for row in stores
    )


@pytest.mark.asyncio
async def test_dependent_instance_recreates_when_dependency_address_moves(
    monkeypatch,
):
    """A store whose stamped master address no longer matches the
    current one is deleted; the replacement stamps the fresh address."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    stale_store = _instance(
        id=22,
        worker_id=5,
        component="store",
        component_addresses={"master": "10.0.0.9:40000"},
        state=CacheServiceStateEnum.RUNNING,
        port=8080,
    )
    _patch_reconcile(
        monkeypatch,
        _pool_provider(),
        workers=[_worker(5)],
        worker=_worker(5),
        instance_lists=[[master, stale_store], [master]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    stale_store.delete.assert_awaited_once()
    master.delete.assert_not_called()


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
async def test_replicas_missing_pinned_worker_parks_service_in_error(monkeypatch):
    service = _service(worker_id=5)
    orphan = _instance(worker_id=5)
    _patch_reconcile(
        monkeypatch,
        _provider("replicas"),
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
async def test_replicas_rejects_worker_from_other_cluster(monkeypatch):
    service = _service(worker_id=5, cluster_id=1)
    _patch_reconcile(
        monkeypatch,
        _provider("replicas"),
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
    # These cover folding a single component's states; pin the provider so
    # they read that path rather than whatever the catalog declares.
    monkeypatch.setattr(
        "gpustack.server.controllers.get_cache_provider",
        _fake_lookup(_provider("per_node")),
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
async def test_the_dependency_address_does_not_depend_on_row_order(monkeypatch):
    """The address is stamped on every dependent, so a different pick between
    two passes reads as the dependency having moved — and deletes and recreates
    them all while the pool is still up. Row order is not something to pick
    by."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = _pool_provider()
    provider.components["master"] = CacheProviderComponent(
        topology="per_node",
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    provider.components["store"] = CacheProviderComponent(
        topology="per_node",
        depends_on="master",
        attach_endpoint=True,
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    masters = [
        _instance(
            id=61, worker_id=7, state=CacheServiceStateEnum.RUNNING, component="master"
        ),
        _instance(
            id=62, worker_id=5, state=CacheServiceStateEnum.RUNNING, component="master"
        ),
    ]
    for master in masters:
        master.port = 41000

    controller = CacheServiceController(MagicMock())
    monkeypatch.setattr(
        "gpustack.server.controllers.Worker.one_by_id",
        AsyncMock(side_effect=lambda _s, worker_id: _worker(worker_id)),
    )

    first = await controller._component_addresses(MagicMock(), provider, masters, {})
    second = await controller._component_addresses(
        MagicMock(), provider, list(reversed(masters)), {}
    )

    assert first == second
    # The lowest worker id, whichever order the rows came back in.
    assert "10.0.0.5" in first["master"]


@pytest.mark.asyncio
async def test_a_dependent_waits_out_a_dependency_that_is_not_up_yet(monkeypatch):
    """The other reading of "no address to hand down": the dependency is
    enabled and still coming up. Deleting the dependent there would restart it
    on every pass until the master lands."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = _pool_provider()
    provider.components["master"] = CacheProviderComponent(
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    provider.components["store"] = CacheProviderComponent(
        topology="per_node",
        depends_on="master",
        attach_endpoint=True,
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    # The master exists but has not reached RUNNING, so it contributes no
    # address this pass.
    master = _instance(
        id=50,
        worker_id=5,
        state=CacheServiceStateEnum.STARTING,
        component="master",
    )
    store = _instance(
        id=51,
        worker_id=5,
        state=CacheServiceStateEnum.RUNNING,
        component="store",
    )
    # Stamped when the master was up before: what must not be read as stale
    # while the master is on its way back.
    store.component_addresses = {"master": "10.0.0.9:9000"}
    service = _service(worker_id=None, config=CacheServiceConfig(fields={}))
    _patch_reconcile(
        monkeypatch,
        provider,
        workers=[_worker(5)],
        instance_lists=[[master, store], [master, store]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_a_dependent_drops_the_address_of_a_dependency_turned_off(monkeypatch):
    """Turning the dependency off leaves nothing to hand down, which reads the
    same as one that has not come up yet. A dependent already carrying its
    address is running against something that will never answer, so it is
    recreated rather than left pointing at it."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderField,
    )

    provider = _pool_provider()
    provider.fields = [
        CacheProviderField(name="enable_extra", type="boolean", default=False),
    ]
    provider.components["master"] = CacheProviderComponent(
        enabled_by="enable_extra",
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    provider.components["store"] = CacheProviderComponent(
        topology="per_node",
        depends_on="master",
        attach_endpoint=True,
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    stale = _instance(
        id=41,
        worker_id=5,
        state=CacheServiceStateEnum.RUNNING,
        component="store",
    )
    stale.component_addresses = {"master": "10.0.0.9:9000"}
    service = _service(
        worker_id=None, config=CacheServiceConfig(fields={"enable_extra": False})
    )
    _patch_reconcile(
        monkeypatch,
        provider,
        workers=[_worker(5)],
        instance_lists=[[stale], [stale]],
    )

    controller = CacheServiceController(MagicMock())
    await controller._reconcile_service(MagicMock(), service)

    stale.delete.assert_awaited()


@pytest.mark.asyncio
async def test_aggregate_ignores_a_disabled_component_still_holding_rows(monkeypatch):
    """A component turned off keeps its rows until the next reconcile deletes
    them, and this aggregate also runs straight off an instance event. Counting
    them would let something the user switched off park the service in ERROR."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderField,
    )

    provider = _pool_provider()
    provider.fields = [
        CacheProviderField(name="enable_extra", type="boolean", default=False),
    ]
    provider.components["master"] = CacheProviderComponent(
        enabled_by="enable_extra",
        run_command="pool-master --port {{port}}",
        gpu_access=False,
    )
    provider.components["store"] = CacheProviderComponent(
        topology="per_node",
        attach_endpoint=True,
        run_command="pool-store --port {{port}}",
        gpu_access=False,
    )
    service = _service(
        state=CacheServiceStateEnum.UNREACHABLE,
        config=CacheServiceConfig(fields={"enable_extra": False}),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
        AsyncMock(
            return_value=[
                _instance(
                    id=31,
                    worker_id=5,
                    state=CacheServiceStateEnum.RUNNING,
                    component="store",
                ),
                # The master is off; its leftover row failed on the way out.
                _instance(
                    id=32,
                    worker_id=5,
                    state=CacheServiceStateEnum.ERROR,
                    component="master",
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.get_cache_provider", _fake_lookup(provider)
    )

    controller = CacheServiceController(MagicMock())
    await controller._sync_service_aggregate(MagicMock(), service)

    written = service.update.await_args.args[1]
    assert written["state"] == CacheServiceStateEnum.RUNNING
    assert written["healthy"] is True
    assert "master" not in (written.get("state_message") or "")


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
    with (
        patch(
            "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
            AsyncMock(return_value=[stale]),
        ),
        patch(
            "gpustack.server.controllers.get_cache_provider",
            _fake_lookup(_provider("per_node")),
        ),
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
    with (
        patch(
            "gpustack.server.controllers.CacheServiceInstance.all_by_fields",
            AsyncMock(return_value=[current, legacy]),
        ),
        patch(
            "gpustack.server.controllers.get_cache_provider",
            _fake_lookup(_provider("per_node")),
        ),
    ):
        controller = CacheServiceController(MagicMock())
        await controller._sync_service_aggregate(MagicMock(), service2)

    args = service2.update.await_args.args[1]
    assert args["state_message"] is None


@pytest.mark.asyncio
async def test_concurrent_reconciles_of_one_service_take_turns(monkeypatch):
    """Four drivers reconcile a service — its own events, worker events,
    instance events, the periodic pass — and each reads the instance rows
    before deciding what is missing. Two of them observing the same gap
    at once would both fill it, so a service is reconciled one pass at a
    time."""
    controller = CacheServiceController(MagicMock())
    overlapped = False
    running = 0
    reconciles = 0

    async def reconcile(session, service):
        nonlocal overlapped, running, reconciles
        reconciles += 1
        running += 1
        overlapped = overlapped or running > 1
        await asyncio.sleep(0)
        running -= 1

    @asynccontextmanager
    async def session():
        yield MagicMock()

    monkeypatch.setattr(cache_service_controller, "async_session", session)
    monkeypatch.setattr(
        cache_service_controller.CacheService,
        "one_by_id",
        AsyncMock(return_value=SimpleNamespace(id=5, deleted_at=None)),
    )
    monkeypatch.setattr(controller, "_reconcile_service", reconcile)

    await asyncio.gather(*(controller._reconcile_service_by_id(5) for _ in range(4)))

    assert reconciles == 4, "every pass must run, just not at the same time"
    assert overlapped is False


@pytest.mark.asyncio
async def test_a_pool_smaller_than_asked_for_says_so(monkeypatch):
    """A component takes one worker per replica, so a cluster with fewer
    matching workers runs a smaller pool. It serves — and the service
    says how much smaller, where a bare count of what exists would read
    as the pool being at size."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.RUNNING,
        port=50051,
    )
    store = _instance(
        id=22,
        worker_id=5,
        component="store",
        state=CacheServiceStateEnum.RUNNING,
        component_addresses={"master": "10.0.0.5:50051"},
    )
    _patch_reconcile(
        monkeypatch,
        _store_pool_provider(3),
        workers=[_worker(5, ip="10.0.0.5")],
        worker=_worker(5, ip="10.0.0.5"),
        instance_lists=[[master, store], [master, store]],
    )
    updates = {}

    async def set_state(session, svc, **kwargs):
        updates.update(kwargs)

    controller = CacheServiceController(MagicMock())
    monkeypatch.setattr(controller, "_set_service_state", set_state)
    await controller._reconcile_service(MagicMock(), service)

    assert updates["state"] == CacheServiceStateEnum.RUNNING
    assert "store 1/3" in updates["state_message"]


@pytest.mark.asyncio
async def test_a_failed_dependency_is_reported_not_kept_pending(monkeypatch):
    """A component without rows is waiting on the one it depends on, and
    reads as pending. Not when that one failed: the stores will never be
    created, and a service parked in pending says nothing about why."""
    service = _service(worker_id=None)
    master = _instance(
        id=21,
        worker_id=5,
        component="master",
        state=CacheServiceStateEnum.ERROR,
    )
    _patch_reconcile(
        monkeypatch,
        _store_pool_provider(1),
        workers=[_worker(5, ip="10.0.0.5")],
        worker=_worker(5, ip="10.0.0.5"),
        instance_lists=[[master], [master]],
    )
    updates = {}

    async def set_state(session, svc, **kwargs):
        updates.update(kwargs)

    controller = CacheServiceController(MagicMock())
    monkeypatch.setattr(controller, "_set_service_state", set_state)
    await controller._reconcile_service(MagicMock(), service)

    assert updates["state"] == CacheServiceStateEnum.ERROR
