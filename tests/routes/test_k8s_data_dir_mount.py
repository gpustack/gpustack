"""Offline tests for the reserved gpustack data-dir volume mount.

No database and no network: ``enforce_data_dir_mounts`` operates on the request
model alone.

Regression cover for the upgrade deadlock in
https://github.com/gpustack/gpustack/issues/6145 — a Kubernetes cluster created
before the mount became configurable (v2.2.0) has no ``volumeMounts`` at all,
and the check used to reject *any* update of it, including one that never
mentioned ``k8s_options``. The historical rows are repaired by the backfill
migration; what is fixed here is the second half, the check reading the request
body as if it were the whole cluster.
"""

import pytest

from gpustack.api.exceptions import InternalServerErrorException, InvalidException
from gpustack.routes.clusters import enforce_data_dir_mounts
from gpustack.schemas.clusters import (
    ClusterCreate,
    ClusterProvider,
    ClusterUpdate,
    ConfigMapVolumeSource,
    DATA_DIR_MOUNT_NAME,
    DATA_DIR_MOUNT_PATH,
    HostPathVolumeSource,
    K8sOptions,
    K8sVolumeMount,
    PersistentVolumeClaimVolumeSource,
    VolumeSource,
)


def _host_path_mount(path, name=DATA_DIR_MOUNT_NAME, mount_path=DATA_DIR_MOUNT_PATH):
    return K8sVolumeMount(
        name=name,
        mount_path=mount_path,
        volume_source=VolumeSource(
            host_path=HostPathVolumeSource(path=path, type="DirectoryOrCreate")
        ),
    )


def _extra_mount():
    return K8sVolumeMount(
        name="model-cache",
        mount_path="/mnt/models",
        volume_source=VolumeSource(
            host_path=HostPathVolumeSource(path="/data/models", type="Directory")
        ),
    )


def _create(**kwargs):
    return ClusterCreate(name="c1", provider=ClusterProvider.Kubernetes, **kwargs)


def _update(**kwargs):
    return ClusterUpdate(name="c1", **kwargs)


def _data_dir(input):
    mounts = input.k8s_options.volume_mounts
    assert mounts, "the data dir mount must always be present"
    return mounts[0]


def _host_path(input):
    return _data_dir(input).volume_source.host_path.path


# --------------------------------------------------------------------------
# an untouched k8s_options is left alone — the half of the deadlock that was
# rejecting valid partial updates
# --------------------------------------------------------------------------


def test_update_without_k8s_options_is_accepted_and_leaves_the_field_alone():
    """``ActiveRecord.update`` only assigns keys in ``model_fields_set``, so the
    stored value — namespace, gpuInstanceOptions and all — stands. Touching
    ``input.k8s_options`` here would turn "leave alone" into "overwrite"."""
    input = _update(description="renamed")

    enforce_data_dir_mounts(input)

    assert input.k8s_options is None
    assert "k8s_options" not in input.model_fields_set


def test_update_carrying_the_mount_round_trips():
    """What the UI does: GET the cluster, edit a field, PUT it all back."""
    input = _update(
        k8s_options=K8sOptions(
            namespace="gpustack-system-abc123",
            volume_mounts=[_host_path_mount("/mnt/data/gpustack"), _extra_mount()],
        )
    )

    enforce_data_dir_mounts(input)

    assert _host_path(input) == "/mnt/data/gpustack"
    assert [m.name for m in input.k8s_options.volume_mounts] == [
        DATA_DIR_MOUNT_NAME,
        "model-cache",
    ]
    assert input.k8s_options.namespace == "gpustack-system-abc123"


# --------------------------------------------------------------------------
# a submitted k8s_options must carry the mount
# --------------------------------------------------------------------------


def test_create_without_k8s_options_is_rejected():
    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(_create())

    assert DATA_DIR_MOUNT_PATH in str(excinfo.value.message)


def test_a_submitted_k8s_options_without_the_mount_is_rejected():
    """Not completed from the stored value: every write goes through this check
    and the backfill fixed the old rows, so a submission missing the mount is a
    malformed payload — and the column is replaced wholesale, so it is likely
    dropping other keys in the same breath."""
    input = _update(k8s_options=K8sOptions(namespace="gpustack-system-abc123"))

    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(input)

    assert "volumeMounts" in str(excinfo.value.message)


def test_an_explicit_null_k8s_options_is_rejected():
    """Clearing the column would drop namespace / gpuInstanceOptions too."""
    input = _update(k8s_options=None)
    assert "k8s_options" in input.model_fields_set

    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(input)

    assert "required" in str(excinfo.value.message)


def test_an_unrelated_mount_in_the_reserved_slot_is_rejected():
    """The reserved slot is verified, not adopted.

    The workaround for the deadlock was to add *some* mount to get past the old
    check; adopting whatever sits at index 0 would have silently turned that
    mount into the data dir and lost it.
    """
    input = _create(k8s_options=K8sOptions(volume_mounts=[_extra_mount()]))

    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(input)

    assert "reserved" in str(excinfo.value.message)
    # The offending entry is named, so the caller can see what to move.
    assert "model-cache" in str(excinfo.value.message)


def test_a_second_entry_claiming_the_reserved_path_is_rejected():
    """Two entries on /var/lib/gpustack render a DaemonSet k8s rejects."""
    input = _create(
        k8s_options=K8sOptions(
            volume_mounts=[
                _host_path_mount("/mnt/first"),
                _host_path_mount("/mnt/second", name="other-data"),
            ]
        )
    )

    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(input)

    assert "reserved" in str(excinfo.value.message)


@pytest.mark.parametrize(
    "volume_source",
    [
        None,
        VolumeSource(
            persistent_volume_claim=PersistentVolumeClaimVolumeSource(claim_name="pvc")
        ),
        VolumeSource(config_map=ConfigMapVolumeSource(name="cm")),
    ],
    ids=["no-source", "pvc", "config-map"],
)
def test_a_non_host_path_data_dir_is_rejected(volume_source):
    input = _create(
        k8s_options=K8sOptions(
            volume_mounts=[
                K8sVolumeMount(
                    name=DATA_DIR_MOUNT_NAME,
                    mount_path=DATA_DIR_MOUNT_PATH,
                    volume_source=volume_source,
                )
            ]
        )
    )

    with pytest.raises(InvalidException) as excinfo:
        enforce_data_dir_mounts(input)

    assert "hostPath" in str(excinfo.value.message)


# --------------------------------------------------------------------------
# everything but the host path is the server's
# --------------------------------------------------------------------------


def test_the_server_owned_fields_are_forced():
    input = _create(
        k8s_options=K8sOptions(
            volume_mounts=[
                K8sVolumeMount(
                    name="my-data",
                    mount_path=DATA_DIR_MOUNT_PATH,
                    read_only=True,
                    volume_source=VolumeSource(
                        host_path=HostPathVolumeSource(
                            path="/mnt/data", type="Directory"
                        ),
                        config_map=ConfigMapVolumeSource(name="cm"),
                    ),
                )
            ]
        )
    )

    enforce_data_dir_mounts(input)

    data_dir = _data_dir(input)
    assert data_dir.name == DATA_DIR_MOUNT_NAME
    assert data_dir.read_only is False
    assert data_dir.volume_source.host_path.type == "DirectoryOrCreate"
    assert data_dir.volume_source.config_map is None
    # Only the host path is the caller's to choose.
    assert _host_path(input) == "/mnt/data"


def test_the_callers_own_mounts_are_left_exactly_as_submitted():
    input = _create(
        k8s_options=K8sOptions(
            volume_mounts=[_host_path_mount("/mnt/data/gpustack"), _extra_mount()]
        )
    )

    enforce_data_dir_mounts(input)

    mounts = input.k8s_options.volume_mounts
    assert [m.name for m in mounts] == [DATA_DIR_MOUNT_NAME, "model-cache"]
    assert mounts[1].mount_path == "/mnt/models"
    assert mounts[1].volume_source.host_path.path == "/data/models"


def test_the_reserved_slot_may_be_claimed_by_mount_path_alone():
    """The slot check accepts either marker, so a row written before the
    reserved name was enforced still passes — and is renamed on the way
    through, keeping its host path."""
    input = _update(
        k8s_options=K8sOptions(
            volume_mounts=[_host_path_mount("/mnt/data", name="data-dir")]
        )
    )

    enforce_data_dir_mounts(input)

    assert _data_dir(input).name == DATA_DIR_MOUNT_NAME
    assert _host_path(input) == "/mnt/data"


def test_the_post_condition_catches_a_broken_normalization(monkeypatch):
    """The invariant is asserted on the value about to be persisted, so a
    normalization bug surfaces here rather than as a worker DaemonSet rendered
    without a persistent data dir."""
    monkeypatch.setattr(
        "gpustack.routes.clusters.normalize_data_dir_mount",
        lambda k8s_options: None,
    )
    input = _create(
        k8s_options=K8sOptions(volume_mounts=[_host_path_mount("/mnt/data", name="x")])
    )

    with pytest.raises(InternalServerErrorException):
        enforce_data_dir_mounts(input)
