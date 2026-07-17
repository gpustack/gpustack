"""E2: PV and PVT gain a ``status`` (phase + phase_message + finalizing).

The status is a nullable JSON column (mirrors GPUInstance.status) so existing
rows default to ``None`` (= active). Camel-case round-trip is verified since the
status is what the finalizer controllers read/write.
"""

from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeSpec,
    GPUInstancePersistentVolumeStatus,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeSpec,
    GPUInstancePersistentVolumeTypeStatus,
)


def test_pv_status_fields_and_camel_roundtrip():
    st = GPUInstancePersistentVolumeStatus(
        phase="Deleting", phase_message="waiting", finalizing=[1, 2]
    )
    assert st.phase == "Deleting"
    assert st.finalizing == [1, 2]
    dumped = st.model_dump(by_alias=True, exclude_none=True)
    assert dumped["phaseMessage"] == "waiting"
    assert GPUInstancePersistentVolumeStatus.model_validate(dumped).finalizing == [1, 2]


def test_pvt_status_fields():
    st = GPUInstancePersistentVolumeTypeStatus(phase="Deleting", finalizing=[])
    assert st.phase == "Deleting"
    assert st.finalizing == []
    assert st.phase_message is None


def test_pv_status_defaults_none():
    pv = GPUInstancePersistentVolume(
        id=1,
        name="pv-1",
        owner_principal_id=1,
        persistent_volume_type_id=2,
        spec=GPUInstancePersistentVolumeSpec(type_="t"),
    )
    assert pv.status is None


def test_pvt_status_defaults_none():
    pvt = GPUInstancePersistentVolumeType(
        id=1,
        name="pvt-1",
        owner_principal_id=1,
        spec=GPUInstancePersistentVolumeTypeSpec(),
    )
    assert pvt.status is None


def test_pv_carries_status_instance():
    pv = GPUInstancePersistentVolume(
        id=1,
        name="pv-1",
        owner_principal_id=1,
        persistent_volume_type_id=2,
        spec=GPUInstancePersistentVolumeSpec(type_="t"),
        status=GPUInstancePersistentVolumeStatus(phase="Deleting", finalizing=[3]),
    )
    assert pv.status.phase == "Deleting"
    assert pv.status.finalizing == [3]


def test_pv_is_deleting():
    def _pv(status):
        return GPUInstancePersistentVolume(
            id=1,
            name="pv-1",
            owner_principal_id=1,
            persistent_volume_type_id=2,
            spec=GPUInstancePersistentVolumeSpec(type_="t"),
            status=status,
        )

    assert _pv(None).is_deleting() is False  # no status = active
    assert _pv(GPUInstancePersistentVolumeStatus(phase="Ready")).is_deleting() is False
    assert (
        _pv(GPUInstancePersistentVolumeStatus(phase="Deleting")).is_deleting() is True
    )


def test_pvt_is_deleting():
    def _pvt(status):
        return GPUInstancePersistentVolumeType(
            id=1,
            name="pvt-1",
            owner_principal_id=1,
            spec=GPUInstancePersistentVolumeTypeSpec(),
            status=status,
        )

    assert _pvt(None).is_deleting() is False
    assert (
        _pvt(GPUInstancePersistentVolumeTypeStatus(phase="Ready")).is_deleting()
        is False
    )
    assert (
        _pvt(GPUInstancePersistentVolumeTypeStatus(phase="Deleting")).is_deleting()
        is True
    )
