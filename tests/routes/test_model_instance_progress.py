from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from sqlalchemy.dialects import mysql, postgresql

from gpustack.api.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
    register_handlers,
)
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_instances as routes
from gpustack.routes.model_instances import update_model_instance
from gpustack.client.model_instance_client import ModelInstanceClient
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.schemas.models import (
    DistributedServers,
    ModelInstance,
    ModelInstancePublic,
    ModelInstanceRankUpdate,
    ModelInstanceRankReport,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    ModelInstanceUpdate,
    SourceEnum,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.server.deps import get_session, get_tenant_context
from gpustack.server.controllers import (
    sync_instance_files_state,
    sync_main_worker_model_file_state,
)
from tests.server.test_model_file_state_sync import _distributed_instance, _patched
from tests.utils.model import new_model, new_model_instance, new_model_instance_workload


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_managed_pd_delete_checks_configuration_before_cleanup(
    report_api, monkeypatch, stale
):
    api = report_api
    api.instance.group_id = "pd-group"
    api.service.delete = AsyncMock()
    monkeypatch.setattr(
        routes.Model,
        "one_by_id",
        AsyncMock(return_value=new_model(1, "m", huggingface_repo_id="org/repo")),
    )
    monkeypatch.setattr(
        routes.ModelInstance, "all_by_field", AsyncMock(return_value=[api.instance])
    )
    monkeypatch.setattr(routes, "model_members_stale", AsyncMock(return_value=stale))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/model-instances/{api.instance.id}")
    assert response.status_code == (400 if stale else 200)
    assert api.service.delete.await_count == int(not stale)
    if stale:
        assert "restart the model" in response.text


@pytest.mark.asyncio
async def test_managed_delete_keeps_tenant_visibility(report_api):
    api = report_api
    api.ctx.scoped_cluster_id = 2
    api.service.delete = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/model-instances/{api.instance.id}")
    assert response.status_code == 404
    api.service.delete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('deleted', [False, True])
async def test_instance_delete_distinguishes_pending_cleanup(report_api, deleted):
    api = report_api
    api.service.delete = AsyncMock(return_value=deleted)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.delete(f'/model-instances/{api.instance.id}')
    assert response.status_code == (200 if deleted else 202)
    assert response.json() is None
    api.service.delete.assert_awaited_once_with(api.instance)


async def _apply_report(instance, report):
    session = MagicMock(info={})

    async def persist(current, source):
        await current.update(session, source)

    service = MagicMock()
    service.update = AsyncMock(side_effect=persist)
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    with (
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch.object(ModelInstance, "save", AsyncMock()),
        patch(
            "gpustack.routes.model_instances.ModelInstanceService", return_value=service
        ),
    ):
        await update_model_instance(session, ctx, instance.id, report)


@pytest.mark.asyncio
@pytest.mark.parametrize("progress", [0, 99, 100])
async def test_state_report_preserves_controller_download_progress(progress):
    instance = new_model_instance(
        13, "qwen3-8b", 2, worker_id=2, state=ModelInstanceStateEnum.DOWNLOADING
    )
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-8B"
    instance.download_progress = 97.95
    instance.draft_model_download_progress = 79
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=1, download_progress=87.03),
            ModelInstanceSubordinateWorker(worker_id=3, download_progress=50),
        ]
    )
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.distributed_servers.subordinate_workers.reverse()
    subordinate = report.distributed_servers.subordinate_workers[1]
    subordinate.state = ModelInstanceStateEnum.ERROR
    subordinate.state_message = "Inference server exited or unhealthy."
    instance.download_progress = progress
    instance.draft_model_download_progress = progress
    instance.distributed_servers.subordinate_workers[0].download_progress = progress
    instance.distributed_servers.subordinate_workers[1].download_progress = 60

    session = MagicMock()
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    service = MagicMock()
    service.update = AsyncMock()
    with (
        patch.object(
            ModelInstance, "one_by_id", AsyncMock(return_value=instance)
        ) as get,
        patch(
            "gpustack.routes.model_instances.ModelInstanceService", return_value=service
        ),
    ):
        await update_model_instance(session, ctx, instance.id, report)

    get.assert_awaited_once_with(session, instance.id, for_update=True)
    service.update.assert_awaited_once()
    updated = service.update.call_args.args[1]
    assert updated.download_progress == progress
    assert updated.draft_model_download_progress == progress
    assert updated.distributed_servers.subordinate_workers[0].download_progress == 60
    assert (
        updated.distributed_servers.subordinate_workers[1].download_progress == progress
    )
    assert updated.distributed_servers.subordinate_workers[1].state == (
        ModelInstanceStateEnum.ERROR
    )


@pytest.mark.asyncio
async def test_instance_update_rejects_another_tenants_instance():
    instance = new_model_instance(13, "qwen3-8b", 2, worker_id=2)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-8B"
    instance.owner_principal_id = 1
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    ctx = TenantContext(
        user=MagicMock(),
        is_platform_admin=False,
        current_principal_id=2,
        org_role=None,
    )
    with (
        patch.object(ModelInstance, "one_by_id", AsyncMock(return_value=instance)),
        patch("gpustack.routes.model_instances.ModelInstanceService") as service,
        pytest.raises(NotFoundException),
    ):
        await update_model_instance(MagicMock(), ctx, instance.id, report)
    service.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot_state",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.DOWNLOADING],
)
@pytest.mark.parametrize(
    "current_state", [ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.RUNNING]
)
async def test_stale_put_preserves_download_completion(snapshot_state, current_state):
    instance = _distributed_instance(sub_download_progress=100)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    instance.state = snapshot_state
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.pid = 42
    report.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.RUNNING
    )

    model = new_model(1, "test", huggingface_repo_id="Qwen/Qwen3-0.6B")
    session = MagicMock(info={})
    with _patched(instance, model):
        await sync_main_worker_model_file_state(
            session, instance.model_files[0], instance
        )
    assert instance.state == ModelInstanceStateEnum.STARTING
    instance.state = current_state
    instance.state_message = "Current status"
    await _apply_report(instance, report)

    assert instance.state == current_state
    assert instance.state_message == "Current status"
    assert instance.download_progress == 100
    assert instance.resolved_path == "/cache/main"
    assert instance.mounted_loras == []
    assert instance.pid == 42
    assert instance.distributed_servers.subordinate_workers[0].state == (
        ModelInstanceStateEnum.RUNNING
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot_state",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.DOWNLOADING],
)
@pytest.mark.parametrize("failed_file_index", [0, 1], ids=["main", "subordinate"])
async def test_stale_put_preserves_download_failure(snapshot_state, failed_file_index):
    instance = _distributed_instance(sub_download_progress=79)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    instance.state = snapshot_state
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.pid = 42

    failed_file = instance.model_files[failed_file_index]
    failed_file.state = ModelFileStateEnum.ERROR
    failed_file.state_message = "Download failed: insufficient disk space"
    with _patched(instance):
        await sync_instance_files_state(MagicMock(), instance, [failed_file])
    assert instance.state == ModelInstanceStateEnum.ERROR

    await _apply_report(instance, report)

    assert instance.state == ModelInstanceStateEnum.ERROR
    assert instance.state_message == failed_file.state_message
    assert instance.pid == 42


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_state, reported_state",
    [
        (ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.RUNNING),
        (ModelInstanceStateEnum.STARTING, ModelInstanceStateEnum.ERROR),
        (ModelInstanceStateEnum.RUNNING, ModelInstanceStateEnum.UNREACHABLE),
        (ModelInstanceStateEnum.ERROR, ModelInstanceStateEnum.SCHEDULED),
        (ModelInstanceStateEnum.SCHEDULED, ModelInstanceStateEnum.INITIALIZING),
    ],
)
async def test_worker_runtime_and_restart_transitions_are_allowed(
    current_state, reported_state
):
    instance = new_model_instance(1, "test", 1, state=current_state)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    report = ModelInstanceUpdate.model_validate(instance.model_dump())
    report.state = reported_state
    report.state_message = "Worker status"

    await _apply_report(instance, report)

    assert instance.state == reported_state
    assert instance.state_message == "Worker status"


@pytest.mark.parametrize("workloads", [None, [], [new_model_instance_workload()]])
def test_manifest_survives_public_json_round_trip(workloads):
    instance = new_model_instance(1, "instance", 1)
    instance.source = SourceEnum.LOCAL_PATH
    instance.local_path = "/model"
    instance.distributed_servers = DistributedServers(workloads=workloads)
    instance.created_at = instance.updated_at = datetime.now(timezone.utc)
    public = ModelInstancePublic.model_validate(instance)
    restored = ModelInstancePublic.model_validate_json(public.model_dump_json())
    assert (
        restored.distributed_servers.workloads == public.distributed_servers.workloads
    )
    assert "rank_report" not in public.model_dump()


def test_missing_workloads_keeps_legacy_contract():
    assert DistributedServers.model_validate({}).workloads is None


@pytest.mark.parametrize("dialect", [mysql.dialect(), postgresql.dialect()])
@pytest.mark.parametrize(
    "workloads",
    [
        None,
        [],
        [new_model_instance_workload()],
        [
            {
                "name": "smg",
                "worker_id": 1,
                "state": "pending",
                "component": {
                    "kind": "sglang_smg",
                    "api_port": 40001,
                    "metrics_port": 40002,
                    "attempt": 2,
                    "last_restart_time": "2026-10-28T00:00:00Z",
                },
            }
        ],
    ],
)
def test_manifest_round_trips_through_existing_json_column(dialect, workloads):
    servers = DistributedServers(workloads=workloads)
    column_type = ModelInstance.__table__.c.distributed_servers.type
    stored = column_type.bind_processor(dialect)(servers)
    restored = column_type.result_processor(dialect, None)(stored)
    assert restored == servers


@pytest.mark.parametrize(
    "changes",
    [
        {"dp_rank": -1},
        {"node_rank": -1},
        {"gpu_indexes": [-1]},
        {"gpu_indexes": [0, 0]},
        {"gpu_indexes": []},
        {"api_port": 0},
        {"api_port": 65536},
        {"api_port": None},
        {"node_rank": 1},
    ],
)
def test_invalid_placement_is_rejected(changes):
    item = new_model_instance_workload()
    item["dp_rank_placements"][0].update(changes)
    with pytest.raises(ValidationError):
        DistributedServers(workloads=[item])


@pytest.mark.parametrize("duplicate", ["placement", "name", "device", "port"])
def test_manifest_rejects_ambiguous_identity_or_resources(duplicate):
    first = new_model_instance_workload()
    second = new_model_instance_workload(worker_id=2, dp_rank=1)
    if duplicate == "placement":
        second["dp_rank_placements"][0]["dp_rank"] = 0
    elif duplicate == "name":
        second["name"] = first["name"]
    else:
        second["worker_id"] = 1
        placement = second["dp_rank_placements"][0]
        placement["api_port"] = 40001 if duplicate == "device" else 40000
        placement["gpu_indexes"] = [0] if duplicate == "device" else [1]
    with pytest.raises(ValidationError):
        DistributedServers(workloads=[first, second])


def test_headless_report_can_arrive_before_main():
    manifest = DistributedServers(workloads=[new_model_instance_workload(node_rank=1)])
    placement = manifest.workloads[0].dp_rank_placements[0]
    assert placement.api_port is None
    assert placement.ready is None


def test_rank_update_only_requires_request_fields():
    update = ModelInstanceRankUpdate.model_validate(
        {
            "rank_report": {
                "restart_count": 0,
                "workloads": [new_model_instance_workload()],
            }
        }
    )
    assert update.rank_report.restart_count == 0
    assert "rank_report" not in ModelInstancePublic.model_fields


@pytest.mark.parametrize("field", ["worker_id", "cluster_id", "state"])
def test_rank_report_rejects_extra_identity_and_state(field):
    with pytest.raises(ValidationError):
        ModelInstanceRankUpdate.model_validate(
            {"rank_report": {"restart_count": 0, "workloads": [], field: 1}}
        )


def test_full_update_cannot_silently_discard_rank_report():
    instance = new_model_instance(1, "instance", 1)
    instance.source = SourceEnum.LOCAL_PATH
    instance.local_path = "/model"
    with pytest.raises(ValidationError):
        ModelInstanceUpdate.model_validate(
            {
                **instance.model_dump(),
                "rank_report": {"restart_count": 0, "workloads": []},
            }
        )


@pytest.fixture
def report_api(monkeypatch):
    instance = new_model_instance(7, "instance", 1, worker_id=1, gpu_indexes=[0])
    instance.source = SourceEnum.LOCAL_PATH
    instance.created_at = instance.updated_at = datetime.now(timezone.utc)
    instance.local_path = "/model"
    instance.cluster_id = 1
    instance.distributed_servers = DistributedServers(
        workloads=[],
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
        ],
    )
    caller = SimpleNamespace(
        id=10, kind=PrincipalType.SYSTEM, worker=SimpleNamespace(id=1, cluster_id=1)
    )
    ctx = TenantContext(
        user=caller,
        is_platform_admin=False,
        current_principal_id=None,
        org_role=None,
        scoped_cluster_id=1,
        scoped_cluster_owner_id=instance.owner_principal_id,
    )
    lookup = AsyncMock(return_value=instance)
    monkeypatch.setattr(routes.ModelInstance, "one_by_id", lookup)
    monkeypatch.setattr(routes.ModelInstance, "one_by_id_with_refresh", lookup)
    service = MagicMock()

    async def persist(current, source):
        if not isinstance(source, dict):
            source = {key: getattr(source, key) for key in source.model_fields_set}
        assert "rank_report" not in source
        for key, value in source.items():
            setattr(current, key, value)

    service.update = AsyncMock(side_effect=persist)
    monkeypatch.setattr(routes, "ModelInstanceService", MagicMock(return_value=service))
    monkeypatch.setattr(
        routes.Model,
        "one_by_id",
        AsyncMock(
            return_value=new_model(
                1, "test", huggingface_repo_id="org/repo", cluster_id=1
            )
        ),
    )
    monkeypatch.setattr(routes.Model, "one_by_id_with_refresh", routes.Model.one_by_id)
    app = FastAPI()
    register_handlers(app)
    app.include_router(routes.router, prefix="/model-instances")
    session = MagicMock()
    app.dependency_overrides[get_session] = lambda: session
    session.get = AsyncMock(
        side_effect=lambda cls, key, **kwargs: SimpleNamespace(
            id=key,
            cluster_id=ctx.user.worker.cluster_id,
            system_principal_id=ctx.user.id,
            deleted_at=None,
        )
    )
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    return SimpleNamespace(
        app=app,
        instance=instance,
        ctx=ctx,
        lookup=lookup,
        service=service,
        session=session,
    )


async def put(api, items=None, generation=0, **fields):
    payload = {
        "rank_report": {
            "restart_count": generation,
            "workloads": (
                items if items is not None else [new_model_instance_workload()]
            ),
        },
        **fields,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        return await client.put("/model-instances/7", json=payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'worker_id,generation,code', [(1, 0, 200), (2, 0, 403), (1, 1, 409)]
)
async def test_managed_snapshot_is_primary_only_fenced_and_private(
    report_api, worker_id, generation, code
):
    from gpustack.schemas.models import ModelConfigSnapshot

    api = report_api
    old = ModelConfigSnapshot(backend_parameters=['--old'])
    api.instance.model_config_snapshot = old
    api.ctx.user.worker.id = worker_id
    snapshot = {'backend_parameters': ['--current'], 'env': {'KEY': 'private-value'}}
    response = await put(api, [], generation=generation, model_config_snapshot=snapshot)
    assert response.status_code == code, response.text
    if code != 200:
        assert api.instance.model_config_snapshot == old
        return
    assert api.instance.model_config_snapshot == ModelConfigSnapshot(**snapshot)
    assert 'model_config_snapshot' not in response.json()
    assert 'private-value' not in response.text
    assert (await put(api, [], model_config_snapshot=None)).status_code == 200
    assert api.instance.model_config_snapshot == ModelConfigSnapshot(**snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_id', [1, 2])
async def test_managed_configuration_snapshot_requires_generation_fenced_report(
    report_api, worker_id
):
    api = report_api
    api.ctx.user.worker.id = worker_id
    payload = api.instance.model_dump(mode='json')
    payload['model_config_snapshot'] = {'backend_parameters': ['--current']}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.put('/model-instances/7', json=payload)
    assert response.status_code == 409, response.text
    assert 'model_config_snapshot' not in response.json()
    assert api.instance.model_config_snapshot is None


@pytest.mark.asyncio
async def test_workers_merge_under_row_lock_and_get_round_trips(report_api):
    api = report_api
    assert (await put(api)).status_code == 200
    api.lookup.assert_awaited_with(api.session, 7, for_update=True)
    api.ctx.user.worker.id = 2
    response = await put(api, [new_model_instance_workload(worker_id=2, dp_rank=1)])
    assert response.status_code == 200, response.text
    assert [
        w["worker_id"] for w in response.json()["distributed_servers"]["workloads"]
    ] == [1, 2]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        fetched = await client.get("/model-instances/7")
    assert fetched.json() == response.json()


@pytest.mark.asyncio
async def test_duplicate_worker_reports_preserve_manifest_order_without_writes(
    report_api,
):
    api = report_api
    for worker_id in (1, 2):
        api.ctx.user.worker.id = worker_id
        assert (
            await put(
                api,
                [
                    new_model_instance_workload(
                        worker_id=worker_id, dp_rank=worker_id - 1
                    )
                ],
            )
        ).status_code == 200
    before = api.instance.model_dump()
    api.service.update.reset_mock()
    for worker_id in (1, 2, 1, 2):
        api.ctx.user.worker.id = worker_id
        response = await put(
            api,
            [new_model_instance_workload(worker_id=worker_id, dp_rank=worker_id - 1)],
        )
        assert response.status_code == 200, response.text
        assert api.instance.model_dump() == before
    api.service.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,status",
    [
        ("legacy_token", 403),
        ("user", 403),
        ("nonmember", 403),
        ("wrong_worker", 403),
        ("wrong_cluster", 403),
        ("other_tenant", 404),
        ("old_generation", 409),
        ("future_generation", 409),
        ("legacy_instance", 409),
        ("wrong_gpu", 400),
    ],
)
async def test_report_rejection_never_mutates_instance(report_api, case, status):
    api = report_api
    items = [new_model_instance_workload()]
    generation = 0
    if case == "legacy_token":
        api.ctx.user.worker = None
    elif case == "user":
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.is_platform_admin = True
    elif case == "nonmember":
        api.ctx.user.worker.id = 3
    elif case == "wrong_worker":
        items = [new_model_instance_workload(worker_id=2)]
    elif case == "wrong_cluster":
        api.ctx.user.worker.cluster_id = 2
    elif case == "other_tenant":
        api.ctx.scoped_cluster_id = 2
    elif case == "old_generation":
        api.instance.restart_count = 1
    elif case == "future_generation":
        generation = 1
    elif case == "legacy_instance":
        api.instance.distributed_servers.workloads = None
    elif case == "wrong_gpu":
        items[0]["dp_rank_placements"][0]["gpu_indexes"] = [3]
    before = api.instance.model_dump()
    response = await put(api, items, generation)
    assert response.status_code == status, response.text
    assert api.instance.model_dump() == before
    api.service.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["name", "port", "rank", "drop", "add"])
async def test_registered_topology_is_immutable(report_api, change):
    api = report_api
    assert (await put(api)).status_code == 200
    items = [new_model_instance_workload()]
    if change == "name":
        items[0]["name"] = "different"
    elif change == "port":
        items[0]["dp_rank_placements"][0]["api_port"] += 1
    elif change == "rank":
        items[0]["dp_rank_placements"][0]["dp_rank"] = 1
    elif change == "drop":
        items = []
    else:
        items.append(new_model_instance_workload(worker_id=1, dp_rank=1, name="extra"))
        items[1]["dp_rank_placements"][0].update(gpu_indexes=[1], api_port=40001)
    before = api.instance.model_dump()
    assert (await put(api, items)).status_code == 409
    assert api.instance.model_dump() == before


@pytest.mark.asyncio
async def test_only_runtime_state_and_main_readiness_change(report_api):
    assert (await put(report_api)).status_code == 200
    item = new_model_instance_workload(state="running")
    item["dp_rank_placements"][0]["ready"] = True
    response = await put(report_api, [item])
    assert response.status_code == 200, response.text
    assert response.json()["distributed_servers"]["workloads"][0] == {
        **item,
        "component": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("omitted", ["workloads", "servers", "field", "empty_list"])
async def test_old_snapshot_cannot_reset_manifest(report_api, omitted):
    api = report_api
    assert (await put(api)).status_code == 200
    payload = api.instance.model_dump(mode="json")
    if omitted == "servers":
        payload["distributed_servers"] = None
    elif omitted == "field":
        payload.pop("distributed_servers")
    else:
        payload["distributed_servers"]["workloads"] = (
            [] if omitted == "empty_list" else None
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.put("/model-instances/7", json=payload)
    assert response.status_code == (
        409 if omitted == "empty_list" else 200
    ), response.text
    assert len(api.instance.distributed_servers.workloads) == 1


@pytest.mark.asyncio
async def test_legacy_update_cannot_enable_managed_path(report_api):
    api = report_api
    api.instance.distributed_servers.workloads = None
    payload = api.instance.model_dump(mode="json")
    payload["distributed_servers"]["workloads"] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.put("/model-instances/7", json=payload)
    assert response.status_code == 409
    assert api.instance.distributed_servers.workloads is None


@pytest.mark.asyncio
async def test_create_cannot_bypass_manifest_registration(report_api):
    api = report_api
    payload = api.instance.model_dump(mode="json")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post("/model-instances", json=payload)
    assert response.status_code == 400
    assert api.instance.distributed_servers.workloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("restart_count", 1),
        ("worker_id", 2),
        ("cluster_id", 2),
        ("gpu_indexes", [3]),
        ("state", "running"),
        ("owner_principal_id", 99),
    ],
)
async def test_snapshot_cannot_bypass_managed_ownership(report_api, field, value):
    api = report_api
    payload = api.instance.model_dump(mode="json")
    payload[field] = value
    before = api.instance.model_dump()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.put("/model-instances/7", json=payload)
    assert response.status_code == 409, response.text
    assert api.instance.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["duplicate", "missing_main", "rank_gap"])
async def test_combined_manifest_validation_is_atomic(report_api, kind):
    api = report_api
    assert (await put(api)).status_code == 200
    api.ctx.user.worker.id = 2
    item = new_model_instance_workload(
        worker_id=2,
        dp_rank=0 if kind == "duplicate" else 1,
        node_rank=1 if kind == "missing_main" else 0,
    )
    if kind == "rank_gap":
        item["dp_rank_placements"][0]["dp_rank"] = 2
    before = api.instance.model_dump()
    response = await put(api, [item])
    assert response.status_code == 400, response.text
    assert api.instance.model_dump() == before


@pytest.mark.asyncio
async def test_cross_worker_workload_name_collision_preserves_the_manifest(report_api):
    api = report_api
    assert (await put(api)).status_code == 200
    before = api.instance.model_dump()
    api.ctx.user.worker.id = 2
    item = new_model_instance_workload(worker_id=2, dp_rank=1)
    item['name'] = api.instance.distributed_servers.workloads[0].name
    response = await put(api, [item])
    assert response.status_code == 400, response.text
    assert api.instance.model_dump() == before


@pytest.mark.asyncio
async def test_headless_worker_can_register_first(report_api):
    api = report_api
    api.ctx.user.worker.id = 2
    response = await put(api, [new_model_instance_workload(worker_id=2, node_rank=1)])
    assert response.status_code == 200, response.text
    api.ctx.user.worker.id = 1
    assert (await put(api)).status_code == 200


@pytest.mark.asyncio
async def test_stopped_task_cannot_be_revived_in_same_generation(report_api):
    api = report_api
    assert (await put(api)).status_code == 200
    assert (
        await put(api, [new_model_instance_workload(state="stopped")])
    ).status_code == 200
    before = api.instance.model_dump()
    response = await put(api, [new_model_instance_workload(state="running")])
    assert response.status_code == 409, response.text
    assert api.instance.model_dump() == before


@pytest.mark.asyncio
async def test_failed_task_can_report_confirmed_stop(report_api):
    assert (await put(report_api)).status_code == 200
    assert (
        await put(report_api, [new_model_instance_workload(state="failed")])
    ).status_code == 200
    response = await put(report_api, [new_model_instance_workload(state="stopped")])
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("workloads", [None, []])
def test_instance_client_preserves_manifest_and_sends_only_report(
    report_api, workloads
):
    api = report_api
    api.instance.distributed_servers.workloads = workloads
    public = ModelInstancePublic.model_validate(api.instance)
    transport = MagicMock()
    response = httpx.Response(200, json=public.model_dump(mode="json"))
    transport.get_httpx_client.return_value.put.return_value = response
    transport.get_httpx_client.return_value.get.return_value = response
    client = ModelInstanceClient(transport, enable_cache=False)
    update = ModelInstanceRankUpdate.model_validate(
        {
            "rank_report": {
                "restart_count": 0,
                "workloads": [new_model_instance_workload()],
            }
        }
    )
    result = client.update(7, update)
    assert result.distributed_servers.workloads == workloads
    assert client.get(7).distributed_servers.workloads == workloads
    transport.get_httpx_client.return_value.put.assert_called_once_with(
        "/model-instances/7",
        content=update.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["deleted", "soft_deleted", "principal_changed", "cluster_changed"]
)
async def test_rank_report_rechecks_worker_identity_after_authentication(
    report_api, change
):
    api = report_api
    worker = SimpleNamespace(
        id=1, cluster_id=1, system_principal_id=10, deleted_at=None
    )
    if change == "deleted":
        worker = None
    elif change == "soft_deleted":
        worker.deleted_at = datetime.now(timezone.utc)
    elif change == "principal_changed":
        worker.system_principal_id = 99
    else:
        worker.cluster_id = 2
    api.session.get.side_effect = None
    api.session.get.return_value = worker
    response = await put(api)
    assert response.status_code == 403
    api.service.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_rank_health_fault_is_fenced_without_falsifying_runtime(report_api):
    api = report_api
    api.instance.state = ModelInstanceStateEnum.RUNNING
    items = [new_model_instance_workload(state='running')]
    response = await put(
        api, items, state='error', state_message='rank 0 HTTP 503 (3/3)'
    )
    assert response.status_code == 200
    assert api.instance.state == ModelInstanceStateEnum.ERROR
    assert api.instance.distributed_servers.workloads[0].state.value == 'running'
    assert (
        await put(api, items, generation=99, state='error', state_message='late')
    ).status_code == 409
    assert api.instance.state_message == 'rank 0 HTTP 503 (3/3)'
    assert (await put(api, items, state='running')).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'state', [ModelInstanceStateEnum.ERROR, ModelInstanceStateEnum.RUNNING]
)
async def test_late_rank_health_report_preserves_stop_intent(report_api, state):
    api = report_api
    api.instance.state = state
    api.instance.state_message = 'first failure'
    if state == ModelInstanceStateEnum.RUNNING:
        api.instance.draining_since = datetime.now(timezone.utc)
    item = new_model_instance_workload(state='running')
    item['dp_rank_placements'][0]['ready'] = True
    response = await put(api, [item], state_message='late success')
    assert response.status_code == 200
    assert api.instance.state_message == 'first failure'
    assert not api.instance.distributed_servers.workloads[0].dp_rank_placements[0].ready


@pytest.mark.asyncio
async def test_managed_preparation_publishes_ports_without_legacy_snapshot(report_api):
    api = report_api
    api.instance.state = ModelInstanceStateEnum.SCHEDULED
    response = await put(
        api,
        [],
        preparation={
            'port': 41000,
            'ports': [41000, 41010],
            'named_ports': {'_vllm_external_rpc': {'base': 41010, 'count': 1}},
            'pid': 123,
        },
    )
    assert response.status_code == 200
    assert api.instance.state == ModelInstanceStateEnum.INITIALIZING
    assert api.instance.port == 41000
    assert api.instance.distributed_servers.workloads == []
    api.instance.state = ModelInstanceStateEnum.STARTING
    assert (await put(api, [], preparation={'pid': 124})).status_code == 200
    assert api.instance.state == ModelInstanceStateEnum.STARTING
    assert api.instance.pid == 124
    assert (
        await put(api, [], generation=99, preparation={'pid': 125})
    ).status_code == 409
    assert (
        await put(api, [], preparation={'port': 42000, 'ports': [42000]})
    ).status_code == 409
    assert api.instance.port == 41000


@pytest.mark.asyncio
async def test_managed_preparation_failure_before_manifest_is_reportable(report_api):
    api = report_api
    api.instance.state = ModelInstanceStateEnum.SCHEDULED
    response = await put(api, [], state='error', state_message='cannot prepare')
    assert response.status_code == 200
    assert api.instance.state == ModelInstanceStateEnum.ERROR
    assert api.instance.distributed_servers.workloads == []


@pytest.mark.asyncio
async def test_external_rank_registration_matches_allocation(report_api, monkeypatch):
    from gpustack.schemas.models import BackendEnum, PortBand

    api = report_api
    api.instance.gpu_indexes = [0, 1, 2, 3]
    api.instance.distributed_servers.subordinate_workers[0].gpu_indexes = [0, 1, 2, 3]
    api.instance.named_ports = {'_vllm_external_api': PortBand(base=41000, count=2)}
    monkeypatch.setattr(
        routes.Model,
        'one_by_id_with_refresh',
        AsyncMock(
            return_value=new_model(
                1,
                'moe',
                huggingface_repo_id='org/moe',
                backend=BackendEnum.VLLM,
                backend_parameters=[
                    '--data-parallel-external-lb',
                    '--dp=2',
                    '--tp=4',
                    '--data-parallel-size-local=2',
                ],
            )
        ),
    )
    for worker_id in (1, 2):
        api.ctx.user.worker.id = worker_id
        placements = [
            dict(
                dp_rank=r,
                node_rank=worker_id - 1,
                gpu_indexes=[2 * r, 2 * r + 1],
                **(dict(api_port=41000 + r, ready=False) if worker_id == 1 else {}),
            )
            for r in (0, 1)
        ]
        item = new_model_instance_workload(
            worker_id=worker_id,
            name=f'worker-{worker_id}',
            dp_rank_placements=placements,
        )
        if worker_id == 2:
            bad = dict(
                item, dp_rank_placements=[dict(p, node_rank=2) for p in placements]
            )
            assert (await put(api, [bad])).status_code == 400
        assert (await put(api, [item])).status_code == 200
    assert len(api.instance.distributed_servers.workloads) == 2


@pytest.mark.asyncio
async def test_manager_and_backend_preparation_use_real_report_api(report_api):
    import asyncio
    from tests.worker.test_serve_manager import _build_serve_manager
    from gpustack.worker.backends.base import InferenceServer

    api = report_api
    api.instance.state = ModelInstanceStateEnum.SCHEDULED
    manager, client = _build_serve_manager(worker_id=1)
    manager._hold_managed_instance(api.instance)
    client.model_instances.get.side_effect = lambda **kw: api.instance.model_copy(
        deep=True
    )
    loop = asyncio.get_running_loop()

    async def send(update):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url='http://test'
        ) as http:
            response = await http.put(
                '/model-instances/7', json=update.model_dump(mode='json')
            )
            assert response.status_code == 200, response.text
            return ModelInstance.model_validate_json(response.text)

    client.model_instances.update.side_effect = (
        lambda *, id, model_update: asyncio.run_coroutine_threadsafe(
            send(model_update), loop
        ).result(timeout=5)
    )
    await asyncio.to_thread(
        manager._update_model_instance,
        7,
        state=ModelInstanceStateEnum.INITIALIZING,
        port=41000,
        ports=[41000, 41010],
    )
    assert api.instance.state == ModelInstanceStateEnum.INITIALIZING
    assert api.instance.port == 41000
    backend = SimpleNamespace(
        _clientset=client,
        _model_instance=api.instance.model_copy(deep=True),
        _worker=SimpleNamespace(id=1),
    )
    await asyncio.to_thread(
        InferenceServer._update_model_instance, backend, 7, backend_version='0.32.0'
    )
    assert api.instance.backend_version == '0.32.0'
    assert api.instance.distributed_servers.workloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'preparation',
    [
        {'port': 41000, 'ports': [41000, 41000]},
        {'port': 41000, 'ports': [41000, 70000]},
        {
            'port': 41000,
            'ports': [41000],
            'named_ports': {'bad': {'base': 65535, 'count': 2}},
        },
    ],
)
async def test_preparation_rejects_invalid_port_metadata(report_api, preparation):
    api = report_api
    api.instance.state = ModelInstanceStateEnum.SCHEDULED
    assert (await put(api, [], preparation=preparation)).status_code == 422
    assert api.instance.port is None


@pytest.mark.asyncio
async def test_subordinate_preparation_cannot_change_shared_ports(report_api):
    api = report_api
    api.instance.state = ModelInstanceStateEnum.STARTING
    api.instance.ports = [41000]
    api.ctx.user.worker.id = 2
    assert (
        await put(api, [], preparation={'port': 42000, 'ports': [42000]})
    ).status_code == 403
    assert (await put(api, [], preparation={'pid': 333})).status_code == 200
    assert api.instance.distributed_servers.subordinate_workers[0].pid == 333
    assert api.instance.state == ModelInstanceStateEnum.STARTING


@pytest.mark.asyncio
async def test_native_rank_registration_verifies_contiguous_worker_apis(
    report_api, monkeypatch
):
    from gpustack.schemas.models import BackendEnum, PortBand

    api = report_api
    api.instance.gpu_indexes = [0, 1, 2, 3]
    api.instance.distributed_servers.subordinate_workers[0].gpu_indexes = [0, 1, 2, 3]
    api.instance.named_ports = {'_vllm_external_api': PortBand(base=41000, count=4)}
    monkeypatch.setattr(
        routes.Model,
        'one_by_id_with_refresh',
        AsyncMock(
            return_value=new_model(
                1,
                'moe',
                huggingface_repo_id='org/moe',
                backend=BackendEnum.VLLM,
                backend_parameters=['-dpm', '--dp=4', '--tp=2'],
            )
        ),
    )
    for worker_id in (1, 2):
        api.ctx.user.worker.id = worker_id
        start = 2 * (worker_id - 1)
        placements = [
            dict(
                dp_rank=start + i,
                node_rank=0,
                gpu_indexes=[2 * i, 2 * i + 1],
                api_port=41000 + start + i,
                ready=False,
            )
            for i in (0, 1)
        ]
        item = new_model_instance_workload(
            worker_id=worker_id,
            name=f'worker-{worker_id}',
            dp_rank_placements=placements,
        )
        bad = dict(
            item,
            dp_rank_placements=[
                dict(p, api_port=41990 + i) for i, p in enumerate(placements)
            ],
        )
        assert (await put(api, [bad])).status_code == 400
        assert (await put(api, [item])).status_code == 200
    assert len(api.instance.distributed_servers.workloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'failure',
    [
        None,
        'maintenance',
        'token',
        'epoch',
        'principal',
        'rebind_new',
        'unreachable',
        'not_ready',
    ],
)
async def test_guarded_legacy_stop_report_is_narrow_and_fenced(report_api, failure):
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop

    api = report_api
    token = uuid4()
    api.instance.distributed_servers.workloads = None
    api.instance.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=token,
        restart_count=0,
        worker_ids=[1, 2],
        worker_principal_ids={1: 10, 2: 20},
    )
    api.instance.draining_since = datetime.now(timezone.utc)
    worker = SimpleNamespace(
        id=1,
        cluster_id=1,
        system_principal_id=10,
        deleted_at=None,
        state='ready',
        unreachable=False,
    )
    api.session.get.side_effect = None
    api.session.get.return_value = worker
    if failure in ('maintenance', 'not_ready'):
        worker.state = failure
    elif failure == 'principal':
        worker.system_principal_id = 11
    elif failure == 'rebind_new':
        worker.system_principal_id = 11
        api.ctx.user.id = 11
    elif failure == 'unreachable':
        worker.unreachable = True
    payload = {
        'stop_report': {
            'token': str(uuid4() if failure == 'token' else token),
            'restart_count': 1 if failure == 'epoch' else 0,
        }
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.put('/model-instances/7', json=payload)
    assert response.status_code == (
        200
        if failure in (None, 'maintenance')
        else (
            403
            if failure in ('principal', 'rebind_new', 'unreachable', 'not_ready')
            else 409
        )
    )
    if failure in (None, 'maintenance'):
        assert api.instance.distributed_servers.workloads is None
        assert api.instance.distributed_servers.legacy_stop.confirmed_worker_ids == [1]
    else:
        api.service.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change', ['omit_guard', 'token', 'owner', 'epoch', 'draining', 'name', 'ports']
)
async def test_legacy_snapshot_cannot_erase_a_stop_barrier(report_api, change):
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop

    api = report_api
    api.instance.distributed_servers.workloads = None
    api.instance.distributed_servers.legacy_stop = ModelInstanceLegacyStop(
        token=uuid4(),
        restart_count=0,
        worker_ids=[1, 2],
        worker_principal_ids={1: 10, 2: 20},
    )
    api.instance.draining_since = datetime.now(timezone.utc)
    payload = api.instance.model_dump(mode='json')
    if change == 'omit_guard':
        payload['distributed_servers'] = None
    elif change == 'token':
        payload['distributed_servers']['legacy_stop']['token'] = str(uuid4())
    elif change == 'owner':
        payload['worker_id'] = 2
    elif change == 'epoch':
        payload['restart_count'] = 1
    elif change == 'name':
        payload['name'] = 'wrong-task'
    elif change == 'ports':
        payload['ports'] = [49999]
    else:
        payload['draining_since'] = None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.put('/model-instances/7', json=payload)
    assert response.status_code == (200 if change == 'omit_guard' else 409)
    assert api.instance.distributed_servers.legacy_stop is not None
    assert api.instance.draining_since is not None


@pytest.mark.parametrize('dialect', [mysql.dialect(), postgresql.dialect()])
def test_legacy_stop_guard_round_trips_in_existing_json_column(dialect):
    from uuid import uuid4
    from gpustack.schemas.models import ModelInstanceLegacyStop

    servers = DistributedServers(
        legacy_stop=ModelInstanceLegacyStop(
            token=uuid4(),
            restart_count=3,
            worker_ids=[1],
            worker_principal_ids={1: 10},
            confirmed_worker_ids=[1],
        )
    )
    column = ModelInstance.__table__.c.distributed_servers.type
    restored = column.result_processor(dialect, None)(
        column.bind_processor(dialect)(servers)
    )
    assert restored == servers
    assert restored.workloads is None


def _sg_component(kind='sglang_engine', worker_id=1, **fields):
    from gpustack.schemas.models import ModelInstanceWorkload

    component = {'kind': kind, **fields}
    return ModelInstanceWorkload(
        name=f'{kind}-{worker_id}',
        worker_id=worker_id,
        state='pending',
        component=component,
    )


@pytest.mark.parametrize(
    'fields',
    [
        {'api_port': 40000, 'dp_size': 4},
        {},
    ],
)
def test_sglang_component_has_no_fake_gpu_placements(fields):
    item = _sg_component(**fields)
    assert item.dp_rank_placements == []
    assert item.component.attempt == 0
    assert DistributedServers(workloads=[item]).workloads == [item]


@pytest.mark.parametrize(
    'kind,fields',
    [
        ('sglang_engine', {'api_port': 40000}),
        ('sglang_engine', {'dp_size': 4}),
        ('sglang_engine', {'attempt': 1}),
        ('sglang_engine', {'stop_requested': True}),
        ('sglang_engine', {'metrics_port': 40001}),
        ('sglang_smg', {'api_port': 40000}),
        ('sglang_smg', {'api_port': 40000, 'metrics_port': 40000}),
        ('sglang_smg', {'api_port': 40000, 'metrics_port': 40001, 'dp_size': 2}),
    ],
)
def test_sglang_component_rejects_invalid_shape(kind, fields):
    with pytest.raises(ValidationError):
        _sg_component(kind, **fields)


def test_sglang_component_cannot_mix_rank_placements_or_ports():
    item = _sg_component(api_port=40000, dp_size=4)
    with pytest.raises(ValidationError):
        type(item).model_validate(
            {
                **item.model_dump(),
                'dp_rank_placements': new_model_instance_workload()[
                    'dp_rank_placements'
                ],
            }
        )
    with pytest.raises(ValidationError):
        DistributedServers(
            workloads=[
                item,
                _sg_component('sglang_smg', api_port=40000, metrics_port=40001),
            ]
        )
    with pytest.raises(ValidationError):
        type(item)(name='empty', worker_id=1, state='pending')


def test_component_report_preserves_server_recovery_state(report_api):
    item = _sg_component(api_port=40000, dp_size=4)
    smg = _sg_component('sglang_smg', api_port=40001, metrics_port=40002)
    smg.component.attempt = 2
    smg.component.stop_requested = True
    smg.component.last_restart_time = datetime.now(timezone.utc)
    report_api.instance.distributed_servers.workloads = [item, smg]
    payload = [item.model_copy(deep=True), smg.model_copy(deep=True)]
    payload[1].component.stop_requested = False
    payload[1].component.last_restart_time = None
    payload[1].state = 'stopped'
    result = routes._merge_rank_report(
        report_api.instance,
        report_api.ctx,
        ModelInstanceRankReport(restart_count=0, workloads=payload),
    )
    assert result.workloads[-1].component == smg.component
    assert result.workloads[-1].state == 'stopped'
    payload[1].component.attempt = 3
    with pytest.raises(ConflictException):
        routes._merge_rank_report(
            report_api.instance,
            report_api.ctx,
            ModelInstanceRankReport(restart_count=0, workloads=payload),
        )


@pytest.mark.parametrize(
    'change', ['attempt', 'stop', 'missing_engine', 'secondary_smg', 'primary_headless']
)
def test_first_component_report_rejects_invalid_local_contract(report_api, change):
    items = [
        _sg_component(api_port=40000, dp_size=4),
        _sg_component('sglang_smg', api_port=40001, metrics_port=40002),
    ]
    if change == 'attempt':
        items[1].component.attempt = 1
    elif change == 'stop':
        items[1].component.stop_requested = True
    elif change == 'missing_engine':
        items.pop(0)
    elif change == 'secondary_smg':
        report_api.ctx.user.worker.id = 2
        for item in items:
            item.worker_id = 2
    else:
        items[0] = _sg_component()
    with pytest.raises((BadRequestException, ConflictException)):
        routes._merge_rank_report(
            report_api.instance,
            report_api.ctx,
            ModelInstanceRankReport(restart_count=0, workloads=items),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'backend,parameters,expected',
    [
        ('SGLang', ['--dp=4'], 200),
        ('vLLM', ['--dp=4'], 400),
        ('SGLang', ['--tp=4'], 400),
        ('SGLang', ['--dp=2'], 400),
        ('SGLang', ['--dp=bad'], 400),
    ],
)
async def test_first_component_registration_checks_model_and_reserved_ports(
    report_api, monkeypatch, backend, parameters, expected
):
    model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend=backend,
        backend_parameters=parameters,
    )
    monkeypatch.setattr(
        routes.Model, 'one_by_id_with_refresh', AsyncMock(return_value=model)
    )
    report_api.instance.port = 40000
    report_api.instance.ports = [40000, 40001, 40002]
    items = [
        _sg_component(api_port=40000, dp_size=4).model_dump(mode='json'),
        _sg_component('sglang_smg', api_port=40001, metrics_port=40002).model_dump(
            mode='json'
        ),
    ]
    response = await put(report_api, items)
    assert response.status_code == expected, response.text


@pytest.mark.parametrize(
    'change',
    ['name', 'owner', 'kind', 'api', 'metrics', 'dp', 'terminal', 'drop_engine'],
)
def test_component_registered_identity_is_immutable(report_api, change):
    items = [
        _sg_component(api_port=40000, dp_size=4),
        _sg_component('sglang_smg', api_port=40001, metrics_port=40002),
    ]
    report_api.instance.distributed_servers.workloads = items
    payload = [item.model_copy(deep=True) for item in items]
    if change == 'name':
        payload[1].name = 'replacement'
    elif change == 'owner':
        payload[1].worker_id = 2
    elif change == 'kind':
        payload[1] = _sg_component(worker_id=1)
    elif change == 'api':
        payload[1].component.api_port = 41000
    elif change == 'metrics':
        payload[1].component.metrics_port = 41001
    elif change == 'dp':
        payload[0].component.dp_size = 8
    elif change == 'terminal':
        items[1].state = 'failed'
        payload[1].state = 'running'
    else:
        payload.pop(0)
    with pytest.raises((ConflictException, routes.ForbiddenException, ValidationError)):
        routes._merge_rank_report(
            report_api.instance,
            report_api.ctx,
            ModelInstanceRankReport(restart_count=0, workloads=payload),
        )


def test_component_json_keeps_legacy_stop_exclusive_and_readiness_strict():
    from gpustack.schemas.models import ModelInstanceWorkload

    item = _sg_component(api_port=40000, dp_size=4)
    with pytest.raises(ValidationError):
        ModelInstanceWorkload.model_validate(
            {
                **item.model_dump(),
                'component': {**item.component.model_dump(), 'ready': True},
            }
        )
    guard = {
        'token': '00000000-0000-0000-0000-000000000001',
        'restart_count': 0,
        'worker_ids': [1],
        'worker_principal_ids': {'1': 10},
    }
    with pytest.raises(ValidationError, match="legacy stop guard"):
        DistributedServers(workloads=[item], legacy_stop=guard)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['foreign_org', 'foreign_cluster', 'missing'])
async def test_dashboard_rejects_invisible_instances_before_grafana_lookup(
    report_api, monkeypatch, case
):
    api = report_api
    if case == 'foreign_org':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.current_principal_id = 10
        api.ctx.scoped_cluster_id = None
        api.instance.owner_principal_id = 99
    elif case == 'foreign_cluster':
        api.ctx.scoped_cluster_id = 2
    else:
        api.lookup.return_value = None
    config = MagicMock(
        side_effect=AssertionError('Grafana lookup before authorization')
    )
    cluster = AsyncMock()
    monkeypatch.setattr(routes, 'get_global_config', config)
    monkeypatch.setattr(routes.Cluster, 'one_by_id', cluster)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.get(f'/model-instances/{api.instance.id}/dashboard')
    assert response.status_code == 404
    assert response.json()['message'] == 'Model instance not found'
    assert 'location' not in response.headers
    config.assert_not_called()
    cluster.assert_not_awaited()
    routes.Model.one_by_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['own_org', 'platform_admin', 'scoped_system'])
async def test_dashboard_preserves_visible_instance_redirect(
    report_api, monkeypatch, case
):
    from urllib.parse import parse_qs, urlsplit

    api = report_api
    api.instance.owner_principal_id = 99
    api.ctx.scoped_cluster_owner_id = 99
    api.instance.model_name = 'model /a'
    if case == 'own_org':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.current_principal_id = 99
        api.ctx.scoped_cluster_id = None
    elif case == 'platform_admin':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.is_platform_admin = True
        api.ctx.scoped_cluster_id = None
    cfg = SimpleNamespace(
        grafana_url='http://grafana:3000',
        grafana_model_dashboard_uid='gpustack-model',
        get_grafana_url=lambda: 'http://grafana:3000',
    )
    monkeypatch.setattr(routes, 'get_global_config', lambda: cfg)
    monkeypatch.setattr(
        routes.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(name='cluster /a')),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        response = await client.get(f'/model-instances/{api.instance.id}/dashboard')
    assert response.status_code == 302
    location = urlsplit(response.headers['location'])
    assert location.netloc == 'grafana:3000'
    assert location.path == '/d/gpustack-model/gpustack-model'
    from gpustack.utils.grafana import model_dashboard_scope_params

    expected = model_dashboard_scope_params(
        cluster_id=1,
        model_id=1,
        model_instance_id=api.instance.id,
        restart_count=api.instance.restart_count or 0,
        cluster_name='cluster /a',
        model_name='model /a',
        model_instance_name='instance',
    )
    expected['var-gpustack_base_url'] = 'http://test'
    assert parse_qs(location.query, keep_blank_values=True) == {
        key: [value] for key, value in expected.items()
    }


@pytest.mark.asyncio
async def test_visible_dashboard_keeps_missing_grafana_configuration_error(
    report_api, monkeypatch
):
    monkeypatch.setattr(
        routes,
        'get_global_config',
        lambda: SimpleNamespace(get_grafana_url=lambda: None),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=report_api.app), base_url='http://test'
    ) as client:
        response = await client.get(
            f'/model-instances/{report_api.instance.id}/dashboard'
        )
    assert response.status_code == 500
    assert 'Grafana dashboard settings are not configured' in response.text


@pytest.mark.parametrize(
    'fields',
    [
        {'image': 'runner-image'},
        {'router_version': '0.3.2'},
        {'image': '', 'router_version': '0.3.2'},
        {'image': '   ', 'router_version': '0.3.2'},
        {'image': 'runner-image', 'router_version': 'unknown'},
    ],
)
def test_engine_component_image_evidence_requires_a_known_pair(fields):
    with pytest.raises(ValidationError):
        _sg_component(api_port=40000, dp_size=4, **fields)


def test_engine_component_image_evidence_is_optional_and_never_on_cpu():
    assert _sg_component().component.image is None
    item = _sg_component(
        api_port=40000, dp_size=4, image='selected-image', router_version='0.3.2'
    )
    assert item.component.image == 'selected-image'
    with pytest.raises(ValidationError):
        _sg_component(
            'sglang_smg',
            api_port=40001,
            metrics_port=40002,
            image='selected-image',
            router_version='0.3.2',
        )


@pytest.mark.parametrize('change', ['image', 'version', 'erase'])
def test_registered_engine_image_evidence_is_immutable(report_api, change):
    items = [
        _sg_component(
            api_port=40000, dp_size=4, image='selected-image', router_version='0.3.2'
        ),
        _sg_component('sglang_smg', api_port=40001, metrics_port=40002),
    ]
    report_api.instance.distributed_servers.workloads = items
    payload = [item.model_copy(deep=True) for item in items]
    if change == 'image':
        payload[0].component.image = 'different-image'
    elif change == 'version':
        payload[0].component.router_version = 'unknown'
    else:
        payload[0].component.image = payload[0].component.router_version = None
    with pytest.raises((ConflictException, ValidationError)):
        routes._merge_rank_report(
            report_api.instance,
            report_api.ctx,
            ModelInstanceRankReport(restart_count=0, workloads=payload),
        )
    assert items[0].component.image == 'selected-image'


@pytest.fixture
def dashboard_api(report_api, monkeypatch):
    api = report_api
    api.cfg = SimpleNamespace(
        grafana_url='https://grafana.example/base',
        grafana_model_dashboard_uid='gpustack-model',
        server_external_url='https://gpu.example/stack',
        get_grafana_url=lambda: 'https://grafana.example/base',
    )
    monkeypatch.setattr(routes, 'get_global_config', lambda: api.cfg)
    monkeypatch.setattr(
        routes.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(name='cluster /a')),
    )
    return api


async def _dashboard_get(api, params=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url='http://test'
    ) as client:
        return await client.get(
            f'/model-instances/{api.instance.id}/dashboard', params=params
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params', [{'dp_rank': 'bad'}, {'component': 'bad'}, {'attempt': 'bad'}]
)
@pytest.mark.parametrize('invisible', ['org', 'cluster', 'missing'])
async def test_dashboard_visibility_precedes_raw_target_syntax(
    dashboard_api, monkeypatch, params, invisible
):
    api = dashboard_api
    if invisible == 'org':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.current_principal_id = 99
        api.ctx.scoped_cluster_id = None
        api.instance.owner_principal_id = 77
    elif invisible == 'missing':
        api.lookup.return_value = None
    else:
        api.ctx.scoped_cluster_id = 2
    config = MagicMock(
        side_effect=AssertionError('configuration read before visibility')
    )
    monkeypatch.setattr(routes, 'get_global_config', config)
    response = await _dashboard_get(api, params)
    assert response.status_code == 404
    config.assert_not_called()
    api.lookup.assert_awaited_once_with(api.session, api.instance.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [
        {'dp_rank': 'bad'},
        {'dp_rank': '-1'},
        {'worker_id': '0'},
        {'restart_count': ''},
        {'attempt': 'true'},
        {'component': 'other'},
        {'component': 'sglang_engine', 'attempt': '0'},
        {'component': 'sglang_smg', 'dp_rank': '0'},
        {'attempt': '0'},
    ],
)
async def test_dashboard_bad_target_syntax_precedes_config(
    dashboard_api, monkeypatch, params
):
    config = MagicMock(
        side_effect=AssertionError('configuration read before target syntax')
    )
    monkeypatch.setattr(routes, 'get_global_config', config)
    response = await _dashboard_get(dashboard_api, params)
    assert response.status_code == 400
    config.assert_not_called()


@pytest.mark.asyncio
async def test_dashboard_empty_manifest_uses_current_identity_and_fixed_server_base(
    dashboard_api,
):
    from urllib.parse import parse_qs, urlsplit
    from gpustack.utils.grafana import model_dashboard_scope_params

    api = dashboard_api
    api.instance.restart_count = 3
    response = await _dashboard_get(api)
    assert response.status_code == 302
    actual = parse_qs(
        urlsplit(response.headers['location']).query, keep_blank_values=True
    )
    expected = model_dashboard_scope_params(
        cluster_id=1,
        model_id=1,
        model_instance_id=7,
        restart_count=3,
        cluster_name='cluster /a',
        model_name=api.instance.model_name,
        model_instance_name=api.instance.name,
    )
    expected['var-gpustack_base_url'] = 'https://gpu.example/stack'
    assert actual == {key: [value] for key, value in expected.items()}
    assert actual['var-scope_legacy_enabled'] == ['0']
    assert actual['var-scope_source_kind'] == ['".*"']
    api.lookup.assert_awaited_once_with(api.session, 7)


def _dashboard_sg(api):
    from gpustack.schemas.models import ModelInstanceWorkload

    api.instance.restart_count = 3
    api.instance.state = ModelInstanceStateEnum.STARTING
    api.instance.distributed_servers.workloads = [
        ModelInstanceWorkload(
            name='engine /a',
            worker_id=1,
            state='failed',
            component=dict(kind='sglang_engine', api_port=41000, dp_size=4),
        ),
        ModelInstanceWorkload(
            name='smg /b',
            worker_id=1,
            state='pending',
            component=dict(
                kind='sglang_smg', api_port=41002, metrics_port=41003, attempt=2
            ),
        ),
        ModelInstanceWorkload(
            name='headless',
            worker_id=2,
            state='pending',
            component=dict(kind='sglang_engine'),
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params,kind,name,rank,attempt',
    [
        ({'dp_rank': '3'}, 'sglang_engine', 'engine /a', 3, None),
        ({'component': 'sglang_engine'}, 'sglang_engine', 'engine /a', None, None),
        ({'component': 'sglang_smg'}, 'sglang_smg', 'smg /b', None, 2),
        ({'component': 'sglang_smg', 'attempt': '2'}, 'sglang_smg', 'smg /b', None, 2),
    ],
)
async def test_dashboard_selects_registered_sg_sources_without_ready_gate(
    dashboard_api, params, kind, name, rank, attempt
):
    from urllib.parse import parse_qs, urlsplit
    from gpustack.utils.grafana import model_dashboard_scope_params

    api = dashboard_api
    _dashboard_sg(api)
    response = await _dashboard_get(api, params)
    assert response.status_code == 302
    actual = parse_qs(
        urlsplit(response.headers['location']).query, keep_blank_values=True
    )
    expected = model_dashboard_scope_params(
        cluster_id=1,
        model_id=1,
        model_instance_id=7,
        restart_count=3,
        source_kind=kind,
        source_name=name,
        worker_id=1,
        dp_rank=rank,
        component_attempt=attempt,
        cluster_name='cluster /a',
        model_name=api.instance.model_name,
        model_instance_name=api.instance.name,
    )
    expected['var-gpustack_base_url'] = 'https://gpu.example/stack'
    assert actual == {key: [value] for key, value in expected.items()}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [
        {'restart_count': '2'},
        {'dp_rank': '4'},
        {'component': 'sglang_smg', 'attempt': '1'},
        {'worker_id': '2'},
        {'worker_id': '99'},
        {'component': 'sglang_engine', 'worker_id': '2'},
    ],
)
async def test_dashboard_rejects_stale_rank_attempt_and_headless_owner_before_config(
    dashboard_api, monkeypatch, params
):
    api = dashboard_api
    _dashboard_sg(api)
    config = MagicMock(
        side_effect=AssertionError('configuration before target binding')
    )
    monkeypatch.setattr(routes, 'get_global_config', config)
    response = await _dashboard_get(api, params)
    assert response.status_code == 404
    config.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['missing', 'deleted', 'cluster', 'id', 'unreachable'])
async def test_dashboard_owner_binding_is_fresh_and_not_worker_health(
    dashboard_api, monkeypatch, case
):
    api = dashboard_api
    _dashboard_sg(api)
    worker = SimpleNamespace(
        id=1, cluster_id=1, deleted_at=None, state='unreachable', unreachable=True
    )
    if case == 'missing':
        worker = None
    elif case == 'deleted':
        worker.deleted_at = datetime.now(timezone.utc)
    elif case == 'cluster':
        worker.cluster_id = 2
    elif case == 'id':
        worker.id = 2
    fresh = AsyncMock(return_value=worker)
    monkeypatch.setattr(routes.Worker, 'one_by_id_with_refresh', fresh)
    if case != 'unreachable':
        monkeypatch.setattr(
            routes,
            'get_global_config',
            MagicMock(side_effect=AssertionError('configuration before owner binding')),
        )
    response = await _dashboard_get(api, {'dp_rank': '0'})
    assert response.status_code == (302 if case == 'unreachable' else 404)
    fresh.assert_awaited_once_with(api.session, 1)


@pytest.mark.asyncio
async def test_dashboard_vllm_rank_binds_global_main_owner(dashboard_api, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    api = dashboard_api
    api.instance.restart_count = 3
    api.instance.distributed_servers.workloads = [
        routes.ModelInstanceWorkload.model_validate(
            new_model_instance_workload(
                worker_id=2,
                name='global-worker2',
                state='failed',
                dp_rank_placements=[
                    dict(
                        dp_rank=5,
                        node_rank=0,
                        gpu_indexes=[0],
                        api_port=41005,
                        ready=False,
                    )
                ],
            )
        ),
    ]
    worker = SimpleNamespace(id=2, cluster_id=1, deleted_at=None)
    fresh = AsyncMock(return_value=worker)
    monkeypatch.setattr(routes.Worker, 'one_by_id_with_refresh', fresh)
    response = await _dashboard_get(api, {'dp_rank': '5', 'worker_id': '2'})
    assert response.status_code == 302
    scope = parse_qs(
        urlsplit(response.headers['location']).query, keep_blank_values=True
    )
    assert scope['var-scope_source_kind'] == ['"^vllm_engine$"']
    assert scope['var-scope_source_name'] == ['"^global-worker2/main-5$"']
    assert scope['var-scope_worker_id'] == ['"2"']
    assert scope['var-scope_endpoint_dp_rank'] == ['"5"']
    assert scope['var-scope_scheduler_dp_rank'] == ['".*"']
    assert (
        await _dashboard_get(api, {'dp_rank': '5', 'worker_id': '1'})
    ).status_code == 404
    assert (await _dashboard_get(api, {'dp_rank': '0'})).status_code == 404
    assert (
        await _dashboard_get(api, {'component': 'sglang_engine'})
    ).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [
        None,
        {'restart_count': '0'},
        {'component': 'sglang_smg'},
        {'dp_rank': '0'},
        {'worker_id': '1'},
    ],
)
async def test_dashboard_empty_manifest_does_not_invent_source(dashboard_api, params):
    api = dashboard_api
    response = await _dashboard_get(api, params)
    assert response.status_code == (
        302 if params in (None, {'restart_count': '0'}) else 404
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [
        None,
        {'restart_count': '0'},
        {'dp_rank': '0'},
        {'worker_id': '1'},
        {'component': 'sglang_smg'},
    ],
)
async def test_dashboard_legacy_only_preserves_no_target_query(dashboard_api, params):
    from urllib.parse import parse_qs, urlsplit

    api = dashboard_api
    api.instance.distributed_servers = None
    response = await _dashboard_get(api, params)
    assert response.status_code == (302 if params is None else 404)
    if params is None:
        scope = parse_qs(
            urlsplit(response.headers['location']).query, keep_blank_values=True
        )
        assert scope['var-scope_legacy_enabled'] == ['1']
        assert scope['var-scope_restart_count'] == ['".*"']
        assert scope['var-scope_source_kind'] == ['".*"']
        assert scope['var-scope_worker_id'] == ['".*"']


@pytest.mark.asyncio
@pytest.mark.parametrize('params', [None, {'worker_id': '1'}])
async def test_dashboard_whole_instance_and_owner_views_do_not_pick_arbitrary_source(
    dashboard_api, params
):
    from urllib.parse import parse_qs, urlsplit

    api = dashboard_api
    _dashboard_sg(api)
    response = await _dashboard_get(api, params)
    assert response.status_code == 302
    scope = parse_qs(
        urlsplit(response.headers['location']).query, keep_blank_values=True
    )
    for name in (
        'source_kind',
        'source_name',
        'endpoint_dp_rank',
        'scheduler_dp_rank',
        'component_attempt',
    ):
        assert scope[f'var-scope_{name}'] == ['".*"']
    assert scope['var-scope_worker_id'] == (['"1"'] if params else ['".*"'])
    assert scope['var-scope_restart_count'] == ['"3"']


@pytest.mark.asyncio
@pytest.mark.parametrize('cached', [True, False])
async def test_dashboard_current_identity_never_uses_cached_row(
    dashboard_api, monkeypatch, cached
):
    api = dashboard_api
    _dashboard_sg(api)
    old = api.instance.model_copy(deep=True)
    old.restart_count = 2
    stale = AsyncMock(return_value=old if cached else None)
    monkeypatch.setattr(routes.ModelInstance, 'one_by_id', stale)
    assert (await _dashboard_get(api, {'restart_count': '2'})).status_code == 404
    stale.assert_not_awaited()
    api.lookup.assert_awaited_once_with(api.session, api.instance.id)


@pytest.mark.asyncio
async def test_dashboard_special_names_and_server_origin_are_encoded_once(
    dashboard_api,
):
    from urllib.parse import parse_qs, urlsplit
    from gpustack.utils.grafana import model_dashboard_scope_params

    api = dashboard_api
    _dashboard_sg(api)
    api.instance.name = '实例 "a"/x\\y&z%$.*'
    api.instance.model_name = 'model +(a)[b]'
    api.instance.distributed_servers.workloads[0].name = '引擎\\"/a&%$.*'
    api.cfg.server_external_url = None
    response = await _dashboard_get(
        api,
        {'dp_rank': '1', 'host': 'evil.example', 'port': '1', 'query': 'or vector(0)'},
    )
    assert response.status_code == 302
    location = urlsplit(response.headers['location'])
    assert location.netloc == 'grafana.example'
    expected = model_dashboard_scope_params(
        cluster_id=1,
        model_id=1,
        model_instance_id=7,
        restart_count=3,
        source_kind='sglang_engine',
        source_name=api.instance.distributed_servers.workloads[0].name,
        worker_id=1,
        dp_rank=1,
        cluster_name='cluster /a',
        model_name=api.instance.model_name,
        model_instance_name=api.instance.name,
    )
    expected['var-gpustack_base_url'] = 'http://test'
    assert parse_qs(location.query, keep_blank_values=True) == {
        key: [value] for key, value in expected.items()
    }


@pytest.mark.asyncio
async def test_dashboard_valid_registered_source_keeps_missing_config_error(
    dashboard_api, monkeypatch
):
    api = dashboard_api
    _dashboard_sg(api)
    monkeypatch.setattr(
        routes,
        'get_global_config',
        lambda: SimpleNamespace(get_grafana_url=lambda: None),
    )
    response = await _dashboard_get(api, {'component': 'sglang_smg', 'attempt': '2'})
    assert response.status_code == 500
    assert 'Grafana dashboard settings are not configured' in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize('caller', ['own_org', 'platform_admin', 'scoped_system'])
async def test_dashboard_source_owner_binding_preserves_tenant_visibility(
    dashboard_api, monkeypatch, caller
):
    api = dashboard_api
    _dashboard_sg(api)
    api.instance.owner_principal_id = 99
    api.ctx.scoped_cluster_owner_id = 99
    if caller == 'own_org':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.current_principal_id = 99
        api.ctx.scoped_cluster_id = None
    elif caller == 'platform_admin':
        api.ctx.user.kind = PrincipalType.USER
        api.ctx.is_platform_admin = True
        api.ctx.scoped_cluster_id = None
    worker = SimpleNamespace(
        id=1, cluster_id=1, deleted_at=None, system_principal_id=77
    )
    monkeypatch.setattr(
        routes.Worker, 'one_by_id_with_refresh', AsyncMock(return_value=worker)
    )
    assert (
        await _dashboard_get(api, {'component': 'sglang_engine'})
    ).status_code == 302


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_requested", [False, True])
async def test_snapshot_cannot_forge_or_clear_server_stop_release(
    report_api, stop_requested
):
    api = report_api
    assert (await put(api)).status_code == 200
    payload = api.instance.model_dump(mode="json")
    api.instance.distributed_servers.stop_requested = stop_requested
    payload["distributed_servers"]["stop_requested"] = not stop_requested
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.put("/model-instances/7", json=payload)
    assert response.status_code == (200 if stop_requested else 409), response.text
    assert api.instance.distributed_servers.stop_requested == stop_requested


@pytest.mark.asyncio
async def test_create_cannot_supply_a_server_stop_release(report_api):
    api = report_api
    payload = api.instance.model_dump(mode="json")
    payload["distributed_servers"]["workloads"] = None
    payload["distributed_servers"]["stop_requested"] = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post("/model-instances", json=payload)
    assert response.status_code == 400, response.text
