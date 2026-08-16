"""Tests for worker proxy in-flight tracking and draining rejection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.routes.worker import proxy as proxy_mod
from gpustack.schemas.models import ModelInstanceStateEnum, SourceEnum
from gpustack.server.bus import Event, EventType
from tests.utils.model import new_model_instance
from tests.worker.test_serve_manager import _build_serve_manager


def _mi_with_source(**kwargs):
    mi = new_model_instance(**kwargs)
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "org/model"
    return mi


def test_begin_end_proxy_request_reports_idle_when_draining():
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=7, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    manager._draining_instance_ids.add(mi.id)

    with patch.object(manager, "_report_drain_idle") as report:
        manager.begin_proxy_request(mi.id)
        assert manager._inflight_requests[mi.id] == 1
        report.assert_not_called()

        manager.end_proxy_request(mi.id)
        assert mi.id not in manager._inflight_requests
        report.assert_called_once_with(mi.id)


def test_end_proxy_request_no_idle_report_when_not_draining():
    manager, _ = _build_serve_manager()
    with patch.object(manager, "_report_drain_idle") as report:
        manager.begin_proxy_request(3)
        manager.end_proxy_request(3)
        report.assert_not_called()


def test_report_drain_idle_is_idempotent():
    """Once idle is reported, further calls must not PATCH again."""
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=1, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    mi.drain_idle = False
    manager._model_instance_by_instance_id[mi.id] = mi
    manager._draining_instance_ids.add(mi.id)

    with patch.object(manager, "_update_model_instance") as update:
        manager._report_drain_idle(mi.id)
        manager._report_drain_idle(mi.id)
        manager._report_drain_idle(mi.id)

    update.assert_called_once_with(mi.id, drain_idle=True)
    assert mi.id in manager._drain_idle_reported
    assert mi.drain_idle is True


def test_draining_sync_skips_idle_report_when_already_idle():
    manager, clientset = _build_serve_manager()
    mi = _mi_with_source(
        id=1, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    mi.drain_idle = True
    manager._model_instance_by_instance_id[mi.id] = mi
    clientset.model_instances.list.return_value = SimpleNamespace(items=[mi])

    with (
        patch.object(manager, "_update_model_instance") as update,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=None,
        ),
    ):
        manager.sync_model_instances_state()
        manager.sync_model_instances_state()

    update.assert_not_called()
    assert mi.id in manager._drain_idle_reported


def test_updated_draining_marks_and_does_not_stop():
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=1, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    event = Event(type=EventType.UPDATED, data=mi)

    with (
        patch.object(manager, "_stop_model_instance") as stop,
        patch.object(manager, "_report_drain_idle") as report,
        patch.object(manager, "_restart_model_instance") as restart,
    ):
        manager._dispatch_model_instance_event(event)

    assert 1 in manager._draining_instance_ids
    stop.assert_not_called()
    restart.assert_not_called()
    report.assert_called_once_with(1)


def test_draining_sticky_skips_running_sync():
    manager, clientset = _build_serve_manager()
    mi = _mi_with_source(
        id=1, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    manager._model_instance_by_instance_id[mi.id] = mi
    clientset.model_instances.list.return_value = SimpleNamespace(items=[mi])

    with (
        patch.object(manager, "_update_model_instance") as update,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="RUNNING"),
        ),
    ):
        manager.sync_model_instances_state()

    update.assert_not_called()


@pytest.mark.asyncio
async def test_set_port_rejects_draining_with_503():
    request = MagicMock()
    request.headers = {proxy_mod.router_header_key: "model-1-42.openai"}
    request.app.state.is_instance_draining = lambda iid: iid == 42
    request.app.state.get_instance_port_by_model_instance_id = lambda _: 8000
    call_next = AsyncMock(return_value=MagicMock())

    with patch(
        "gpustack.routes.worker.proxy.get_instance_id_from_header",
        return_value=42,
    ):
        resp = await proxy_mod.set_port_from_model_name(request, call_next)

    assert resp.status_code == 503
    assert b"draining" in resp.body
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_inflight_dec_on_error():
    begin = MagicMock()
    end = MagicMock()

    request = MagicMock()
    request.state.x_target_port = "8000"
    request.state.x_target_instance_id = 9
    request.method = "POST"
    request.url.query = ""
    request.headers = {}
    request.body = AsyncMock(return_value=b"{}")
    request.app.state.worker_ip_getter = lambda: "127.0.0.1"
    request.app.state.begin_proxy_request = begin
    request.app.state.end_proxy_request = end
    request.app.state.http_client = SimpleNamespace(
        request=AsyncMock(side_effect=RuntimeError("boom"))
    )
    request.app.state.http_client_no_proxy = SimpleNamespace(
        request=AsyncMock(side_effect=RuntimeError("boom"))
    )

    with (
        patch("gpustack.routes.worker.proxy.use_proxy_env_for_url", return_value=False),
        pytest.raises(ServiceUnavailableException),
    ):
        await proxy_mod.proxy("v1/chat/completions", request)

    begin.assert_called_once_with(9)
    end.assert_called_once_with(9)
