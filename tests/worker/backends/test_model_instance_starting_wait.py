from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.worker.backends.base import ModelInstanceStateError
from gpustack.worker.backends.vllm import VLLMServer


def _server(worker_id):
    server = VLLMServer.__new__(VLLMServer)
    server._model_instance = SimpleNamespace(
        name="instance", worker_id=1, distributed_servers=None
    )
    server._worker = SimpleNamespace(id=worker_id)
    return server


def _event(state):
    return Event(type=EventType.UPDATED, data={"state": state})


@pytest.mark.parametrize(
    "state",
    [
        ModelInstanceStateEnum.ERROR,
        ModelInstanceStateEnum.RUNNING,
        ModelInstanceStateEnum.UNREACHABLE,
    ],
)
def test_main_worker_stops_waiting_when_starting_cannot_be_reached(state):
    with pytest.raises(ModelInstanceStateError):
        _server(worker_id=1)._stop_when_starting(_event(state))


@pytest.mark.parametrize(
    "state",
    [ModelInstanceStateEnum.RUNNING, ModelInstanceStateEnum.UNREACHABLE],
)
def test_follower_keeps_waiting_under_a_live_main_worker(state):
    assert _server(worker_id=2)._stop_when_starting(_event(state)) is False


@pytest.mark.parametrize(
    "state",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.DOWNLOADING],
)
def test_main_worker_keeps_waiting_before_starting(state):
    assert _server(worker_id=1)._stop_when_starting(_event(state)) is False
