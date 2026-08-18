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
    """Once idle is reported, further calls must not spawn another thread."""
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=1, name="m-a", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    mi.drain_idle = False
    manager._model_instance_by_instance_id[mi.id] = mi
    manager._draining_instance_ids.add(mi.id)

    with (
        patch.object(manager, "_update_model_instance") as update,
        patch("gpustack.worker.serve_manager.threading.Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        manager._report_drain_idle(mi.id)
        # Execute the thread target to simulate completion
        mock_thread_cls.call_args.kwargs["target"]()

        # Second call should be a no-op (already reported)
        manager._report_drain_idle(mi.id)
        # Third call should also be a no-op
        manager._report_drain_idle(mi.id)

    update.assert_called_once_with(mi.id, drain_idle=True)
    assert mi.id in manager._drain_idle_reported
    assert mi.drain_idle is True
    # Only one thread was spawned
    mock_thread_cls.assert_called_once()
    assert mock_thread_cls.call_args.kwargs["daemon"] is True


def test_report_drain_idle_offloads_to_thread():
    """The blocking HTTP call must not execute on the calling thread."""
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=2, name="m-b", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    mi.drain_idle = False
    manager._model_instance_by_instance_id[mi.id] = mi
    manager._draining_instance_ids.add(mi.id)

    with (
        patch.object(manager, "_update_model_instance") as update,
        patch("gpustack.worker.serve_manager.threading.Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        manager._report_drain_idle(mi.id)

        # The update must NOT have been called synchronously
        update.assert_not_called()
        # But a thread was spawned
        mock_thread_cls.assert_called_once()
        assert mock_thread_cls.call_args.kwargs["daemon"] is True

        # Simulate thread execution
        mock_thread_cls.call_args.kwargs["target"]()
        update.assert_called_once_with(mi.id, drain_idle=True)


def test_report_drain_idle_thread_exception_caught():
    """If the HTTP call fails in the thread, the exception is swallowed and
    the instance is NOT marked as reported (retry possible)."""
    manager, _ = _build_serve_manager()
    mi = _mi_with_source(
        id=3, name="m-c", model_id=1, worker_id=1, state=ModelInstanceStateEnum.DRAINING
    )
    mi.drain_idle = False
    manager._model_instance_by_instance_id[mi.id] = mi
    manager._draining_instance_ids.add(mi.id)

    with (
        patch.object(
            manager, "_update_model_instance", side_effect=RuntimeError("boom")
        ) as update,
        patch("gpustack.worker.serve_manager.threading.Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        manager._report_drain_idle(mi.id)
        # Execute the thread target — should not raise
        mock_thread_cls.call_args.kwargs["target"]()

    update.assert_called_once_with(mi.id, drain_idle=True)
    # Instance NOT marked as reported (retry possible)
    assert mi.id not in manager._drain_idle_reported
    assert mi.drain_idle is False


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


@pytest.mark.asyncio
async def test_set_port_malformed_header_no_500():
    """A malformed router header must not cause a 500; the request should
    fall through to the normal port-resolution path (which will 404)."""
    from gpustack.api.exceptions import NotFoundException

    request = MagicMock()
    request.headers = {proxy_mod.router_header_key: "garbage-not-a-model-id"}
    request.app.state.is_instance_draining = MagicMock(return_value=False)
    request.app.state.get_instance_port_by_model_instance_id = MagicMock(
        side_effect=NotFoundException(message="not found")
    )
    call_next = AsyncMock(return_value=MagicMock())

    resp = await proxy_mod.set_port_from_model_name(request, call_next)

    # Should return 404 from get_model_instance_info_from_model_name, not 500
    assert resp.status_code == 404
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_port_missing_header_no_500():
    """When the router header is absent, the request passes through
    immediately without any draining check."""
    request = MagicMock()
    request.headers = {}
    request.app.state.is_instance_draining = MagicMock(return_value=False)
    call_next = AsyncMock(return_value=MagicMock())

    await proxy_mod.set_port_from_model_name(request, call_next)

    # Early return: call_next is invoked, no draining check
    call_next.assert_awaited_once_with(request)
    request.app.state.is_instance_draining.assert_not_called()


@pytest.mark.asyncio
async def test_set_port_draining_check_uses_int_directly():
    """The draining check receives the int instance ID directly from
    get_instance_id_from_header (no redundant int() cast)."""
    request = MagicMock()
    request.headers = {proxy_mod.router_header_key: "model-1-42.openai"}
    is_draining = MagicMock(return_value=True)
    request.app.state.is_instance_draining = is_draining
    request.app.state.get_instance_port_by_model_instance_id = lambda _: 8000
    call_next = AsyncMock(return_value=MagicMock())

    with patch(
        "gpustack.routes.worker.proxy.get_instance_id_from_header",
        return_value=42,
    ):
        resp = await proxy_mod.set_port_from_model_name(request, call_next)

    assert resp.status_code == 503
    # is_instance_draining received the int 42 directly
    is_draining.assert_called_once_with(42)
