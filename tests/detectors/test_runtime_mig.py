"""
Tests for how the runtime detector reports a MIG-partitioned card:

- Docker clusters address the MIG devices themselves (no operator-side
  partitioner exists there), so the inventory lists them instead of the card.
- Kubernetes clusters under the device plugin keep reporting the card, which is
  what an InstanceType slices on demand.
"""

import pytest

from gpustack_runtime.detector import Device, ManufacturerEnum

import gpustack.detectors.runtime.runtime as runtime_module
from gpustack.detectors.runtime.runtime import Runtime


def _mig_device(index, uuid, memory=4864):
    return {
        "index": index,
        "name": "1g.5gb",
        "uuid": uuid,
        "driver_version": "570.86.15",
        "runtime_version": "12.8",
        "compute_capability": "8.0",
        "cores": 14,
        "cores_utilization": 0,
        "memory": memory,
        "memory_used": 0,
        "memory_utilization": 0,
        "appendix": {
            "arch_family": "ampere",
            "vgpu": True,
            "sliced": True,
            "mig": True,
            "bdf": "0000:00:04.0",
        },
    }


def _card(index=0, uuid="GPU-0", mig_devices=None):
    appendix = {
        "arch_family": "ampere",
        "vgpu": False,
        "mig": mig_devices is not None,
        "bdf": "0000:00:04.0",
    }
    if mig_devices is not None:
        appendix["mig_devices"] = mig_devices
    return Device(
        manufacturer=ManufacturerEnum.NVIDIA,
        index=index,
        name="NVIDIA A100-SXM4-40GB",
        uuid=uuid,
        driver_version="570.86.15",
        runtime_version="12.8",
        compute_capability="8.0",
        cores=108,
        memory=40960,
        appendix=appendix,
    )


@pytest.fixture
def detected(monkeypatch):
    """Feed the detector a fixed device list; returns a setter."""

    def _set(devices):
        monkeypatch.setattr(
            runtime_module, "detect_devices", lambda fast=False: devices
        )

    return _set


def _set_mig_addressable(monkeypatch, addressable):
    monkeypatch.setattr(runtime_module, "_mig_devices_addressable", lambda: addressable)


def test_mig_devices_replace_the_card_where_addressable(detected, monkeypatch):
    detected(
        [
            _card(0, "GPU-0", [_mig_device(2, "MIG-0-0"), _mig_device(3, "MIG-0-1")]),
            _card(1, "GPU-1"),
        ]
    )
    _set_mig_addressable(monkeypatch, True)

    gpus = Runtime().gather_gpu_info()

    assert [gpu.uuid for gpu in gpus] == ["MIG-0-0", "MIG-0-1", "GPU-1"]
    assert [gpu.index for gpu in gpus] == [2, 3, 1]
    # The index is what the deploy path requests, so device_index follows it.
    assert [gpu.device_index for gpu in gpus] == [2, 3, 1]
    assert [gpu.name for gpu in gpus] == ["1g.5gb", "1g.5gb", "NVIDIA A100-SXM4-40GB"]
    assert gpus[0].memory.total == 4864 * (1 << 20)
    assert gpus[0].arch_family == "ampere"
    assert gpus[0].type == "cuda"


def test_card_is_reported_where_mig_devices_are_not_addressable(detected, monkeypatch):
    detected(
        [_card(0, "GPU-0", [_mig_device(2, "MIG-0-0"), _mig_device(3, "MIG-0-1")])]
    )
    _set_mig_addressable(monkeypatch, False)

    gpus = Runtime().gather_gpu_info()

    assert [gpu.uuid for gpu in gpus] == ["GPU-0"]
    assert gpus[0].memory.total == 40960 * (1 << 20)


def test_mig_enabled_card_without_instances_is_still_reported(detected, monkeypatch):
    detected([_card(0, "GPU-0", [])])
    _set_mig_addressable(monkeypatch, True)

    gpus = Runtime().gather_gpu_info()

    assert [gpu.uuid for gpu in gpus] == ["GPU-0"]


def test_mig_devices_are_addressable_by_the_deploying_deployer(monkeypatch):
    class _Deployer:
        def __init__(self, allowed):
            self.allowed_mig_devices = allowed

    # The first supported deployer is the one that would run the workload.
    monkeypatch.setattr(
        runtime_module,
        "supported_deployers",
        lambda: [_Deployer(True), _Deployer(False)],
    )
    assert runtime_module._mig_devices_addressable() is True

    monkeypatch.setattr(
        runtime_module,
        "supported_deployers",
        lambda: [_Deployer(False), _Deployer(True)],
    )
    assert runtime_module._mig_devices_addressable() is False

    monkeypatch.setattr(runtime_module, "supported_deployers", list)
    assert runtime_module._mig_devices_addressable() is False
