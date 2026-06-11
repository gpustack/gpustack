import pytest
from unittest.mock import patch

from gpustack.policies.base import Allocated
from gpustack.routes.gpu_devices import (
    _inject_allocated_into_event,
    to_gpu_device_public,
)
from gpustack.schemas.gpu_devices import GPUDevice, GPUDevicePublic
from gpustack.server.bus import Event, EventType
from gpustack.server.worker_allocated_cache import vram_allocated_for_index


_DEFAULT_MEMORY = "default"


def make_gpu_device(memory=_DEFAULT_MEMORY, index=0):
    if memory is _DEFAULT_MEMORY:
        memory = {"total": 100, "used": 10}
    device = GPUDevice(
        id="w1:cuda:0",
        worker_id=1,
        worker_name="w1",
        worker_ip="10.0.0.1",
        worker_ifname="eth0",
        cluster_id=1,
        index=index,
        name="A100",
    )
    # Simulate a row loaded from gpu_devices_view: JSON columns arrive as
    # raw dicts on table models (no validation on load).
    object.__setattr__(device, "memory", memory)
    return device


@pytest.mark.parametrize(
    "vram, index, expected",
    [
        ({0: 123, 1: 456}, 0, 123),
        ({0: 123}, 1, 0),  # device has no instance assigned
        ({0: 123}, None, 0),  # device index unknown
        ({}, 0, 0),
    ],
)
def test_vram_allocated_for_index(vram, index, expected):
    assert vram_allocated_for_index(vram, index) == expected


def test_to_gpu_device_public_injects_allocated():
    public = to_gpu_device_public(make_gpu_device(), {0: 123})
    assert isinstance(public, GPUDevicePublic)
    assert public.memory.allocated == 123
    assert public.memory.total == 100


def test_to_gpu_device_public_no_assignment_defaults_zero():
    public = to_gpu_device_public(make_gpu_device(), {1: 999})
    assert public.memory.allocated == 0


def test_to_gpu_device_public_index_none_defaults_zero():
    public = to_gpu_device_public(make_gpu_device(index=None), {0: 123})
    assert public.memory.allocated == 0


def test_to_gpu_device_public_memory_none():
    public = to_gpu_device_public(make_gpu_device(memory=None), {0: 123})
    assert public.memory is None


def test_convert_to_public_class():
    """Covers the streaming conversion path: GPUDevicePublic needs
    from_attributes to validate the GPUDevice table model, otherwise the
    watch stream dies on the first replayed event."""
    public = GPUDevice._convert_to_public_class(make_gpu_device())
    assert isinstance(public, GPUDevicePublic)


def _patched_allocated(vram):
    async def fake_get_worker_allocated(worker_id):
        return Allocated(ram=0, vram=vram)

    return patch(
        "gpustack.routes.gpu_devices.get_worker_allocated",
        fake_get_worker_allocated,
    )


@pytest.mark.asyncio
async def test_inject_allocated_into_event():
    device = GPUDevice._convert_to_public_class(make_gpu_device())
    event = Event(type=EventType.UPDATED, data=device, id=1)
    with _patched_allocated({0: 777}):
        await _inject_allocated_into_event(event)
    assert event.data.memory.allocated == 777


@pytest.mark.asyncio
async def test_inject_allocated_into_event_skips_deleted():
    event = Event(type=EventType.DELETED, data={"id": 1}, id=1)
    with _patched_allocated({0: 777}):
        await _inject_allocated_into_event(event)
    assert event.data == {"id": 1}


@pytest.mark.asyncio
async def test_inject_allocated_into_event_skips_non_public_data():
    event = Event(type=EventType.UPDATED, data={"id": 1}, id=1)
    with _patched_allocated({0: 777}):
        await _inject_allocated_into_event(event)
    assert event.data == {"id": 1}


@pytest.mark.asyncio
async def test_inject_allocated_into_event_skips_memory_none():
    device = GPUDevice._convert_to_public_class(make_gpu_device(memory=None))
    event = Event(type=EventType.UPDATED, data=device, id=1)
    with _patched_allocated({0: 777}):
        await _inject_allocated_into_event(event)
    assert event.data.memory is None
