from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from gpustack.api.tenant import TenantContext
from gpustack.routes import clusters, workers
from gpustack.schemas.clusters import (
    Cluster,
    ClusterProvider,
    ClusterPublic,
    ClusterUpdate,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.workers import (
    Maintenance,
    Worker,
    WorkerCreate,
    WorkerStateEnum,
    WorkerStatusStored,
    WorkerUpdate,
)
from gpustack.server.services import WorkerService


def _worker(**fields):
    now = datetime.now(timezone.utc)
    return Worker.model_validate(
        {
            "id": 1,
            "name": "node",
            "hostname": "node",
            "ip": "192.0.2.10",
            "ifname": "eth0",
            "port": 10150,
            "worker_uuid": "worker-1",
            "status": {},
            "created_at": now,
            "updated_at": now,
            "cluster_id": 7,
            "owner_principal_id": 10,
            **fields,
        }
    )


def _cluster_config(value):
    return ClusterUpdate.model_validate(
        {"name": "cluster", "worker_config": {"kv_transfer_ifname": value}}
    ).worker_config


@pytest.mark.parametrize(
    "provider", [ClusterProvider.Docker, ClusterProvider.Kubernetes]
)
def test_cluster_yaml_keeps_default_out_of_worker_startup(provider, monkeypatch):
    request = SimpleNamespace(url=SimpleNamespace(scheme="http", netloc="gpustack"))
    config = SimpleNamespace(
        server_external_url=None,
        system_default_container_registry=None,
        container_namespace=None,
        effective_trace_otlp_endpoint=None,
    )
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(workers, "get_global_config", lambda: config)
    now = datetime.now(timezone.utc)
    cluster = Cluster.model_validate(
        {
            "id": 7,
            "name": "cluster",
            "provider": provider,
            "registration_token": "test-registration-token",
            "worker_config": {
                "kv_transfer_ifname": " mlx5_2:1, mlx5_5:1 ",
                "huggingface_token": "test-hf-token",
                "worker_port": 10155,
            },
            "created_at": now,
            "updated_at": now,
        }
    )

    column_type = Cluster.__table__.c.worker_config.type
    dialect = postgresql.dialect()
    encoded = column_type.bind_processor(dialect)(cluster.worker_config)
    cluster.worker_config = column_type.result_processor(dialect, None)(encoded)
    public = ClusterPublic.model_validate(cluster, from_attributes=True)
    assert public.worker_config.kv_transfer_ifname == " mlx5_2:1, mlx5_5:1 "
    assert "test-hf-token" not in public.model_dump_json()
    registration = clusters.get_registration_from_cluster(request, cluster)
    assert "GPUSTACK_KV_TRANSFER_IFNAME" not in registration.env
    worker_config = workers._build_worker_config_dict(cluster)
    assert "kv_transfer_ifname" not in worker_config
    assert worker_config["worker_port"] == 10155


@pytest.mark.parametrize(
    "value",
    ["all", "eth0,eth1", " ens5f0, mlx5_2:1 "],
)
def test_server_kv_settings_preserve_the_configured_value(value):
    assert _cluster_config(value).kv_transfer_ifname == value
    assert (
        WorkerUpdate(name="node", kv_transfer_ifname=value).kv_transfer_ifname == value
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, {}),
        (
            {"kv_transfer_ifname": " mlx5_2:1, mlx5_5:1 "},
            {"kv_transfer_ifname": " mlx5_2:1, mlx5_5:1 "},
        ),
        ({"kv_transfer_ifname": None}, {"kv_transfer_ifname": None}),
    ],
)
async def test_worker_edit_distinguishes_omission_from_clearing(
    fields, expected, monkeypatch
):
    maintenance = Maintenance(enabled=True, message="Replacing a network cable")
    latest_labels = {"rack": "rack-a", "pool": "production"}
    worker = _worker(
        kv_transfer_ifname="ens5f0",
        labels=latest_labels,
        maintenance=maintenance,
        heartbeat_time=datetime.now(timezone.utc),
    )
    worker.compute_state()
    monkeypatch.setattr(Worker, "one_by_id", AsyncMock(return_value=worker))

    async def apply_worker_patch(row, patch):
        for key, value in patch.items():
            setattr(row, key, value)

    persist = AsyncMock(side_effect=apply_worker_patch)
    monkeypatch.setattr(WorkerService, "update", persist)
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=10,
        org_role=OrgRole.OWNER,
    )

    await workers.update_worker(
        ctx, None, worker.id, WorkerUpdate(name="node", **fields)
    )

    patch = persist.await_args.args[1]
    assert patch == {"name": worker.name, **expected}
    assert worker.kv_transfer_ifname == expected.get("kv_transfer_ifname", "ens5f0")
    assert worker.labels == latest_labels
    worker.compute_state()
    assert worker.maintenance == maintenance
    assert worker.state == WorkerStateEnum.MAINTENANCE
    assert worker.state_message == maintenance.message


def test_worker_reregistration_preserves_the_admin_override():
    worker = _worker(kv_transfer_ifname="ens5f0")
    reported = WorkerCreate.model_validate(
        {**worker.model_dump(), "kv_transfer_ifname": "ens6f0"}
    )

    updated = workers.update_worker_data(reported, existing=worker)

    assert "kv_transfer_ifname" not in reported.model_fields_set
    assert updated.kv_transfer_ifname == "ens5f0"


def test_worker_status_keeps_the_startup_value_separate_from_the_admin_override():
    report = WorkerStatusStored.model_validate(
        {
            **_worker().model_dump(),
            "status": {"kv_transfer_ifname": "ib0"},
            "kv_transfer_ifname": "ens5f0",
        }
    )

    assert "kv_transfer_ifname" not in report.model_fields_set
    assert report.status.kv_transfer_ifname == "ib0"


@pytest.mark.asyncio
async def test_worker_get_reads_cluster_default_for_its_own_identity(monkeypatch):
    worker = _worker(kv_transfer_ifname="ens5f0")
    cluster = Cluster(name="cluster", id=7, worker_config=_cluster_config("mlx5_2:1"))
    monkeypatch.setattr(Worker, "one_by_id", AsyncMock(return_value=worker))
    lookup_cluster = AsyncMock(return_value=cluster)
    monkeypatch.setattr(Cluster, "one_by_id", lookup_cluster)
    monkeypatch.setattr(workers, "_lookup_allocated", AsyncMock(return_value=(0, {})))
    user = SimpleNamespace(kind=PrincipalType.SYSTEM, worker=worker)
    ctx = TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=10,
        org_role=None,
        scoped_cluster_id=7,
        scoped_cluster_owner_id=10,
    )

    public = await workers.get_worker(user, ctx, None, worker.id)

    assert public.kv_transfer_ifname == "ens5f0"
    assert public.cluster_kv_transfer_ifname == "mlx5_2:1"
    lookup_cluster.assert_awaited_once_with(None, worker.cluster_id)

    cluster.worker_config = _cluster_config("ib0")
    public = await workers.get_worker(user, ctx, None, worker.id)
    assert public.cluster_kv_transfer_ifname == "ib0"

    user.worker = _worker(id=2)
    public = await workers.get_worker(user, ctx, None, worker.id)
    assert public.cluster_kv_transfer_ifname is None
    assert lookup_cluster.await_count == 2
