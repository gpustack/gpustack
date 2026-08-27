import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.routes.clusters import create_update_check, ensure_k8s_data_dir_mount
from gpustack.schemas.clusters import (
    ClusterCreate,
    ClusterProvider,
    ClusterUpdate,
    K8sOptions,
    K8sVolumeMount,
    VolumeSource,
    HostPathVolumeSource,
    PersistentVolumeClaimVolumeSource,
)


def _host_path_mount(path="/var/lib/gpustack"):
    return {
        "name": "gpustack-data-dir",
        "mountPath": "/var/lib/gpustack",
        "readOnly": False,
        "volumeSource": {"hostPath": {"path": path, "type": "DirectoryOrCreate"}},
    }


def test_legacy_k8s_volume_mounts_hoisted_into_k8s_options():
    created = ClusterCreate.model_validate(
        {
            "name": "c1",
            "provider": "Kubernetes",
            "k8s_volume_mounts": [_host_path_mount()],
        }
    )
    assert created.k8s_options is not None
    assert created.k8s_options.volume_mounts is not None
    assert created.k8s_options.volume_mounts[0].name == "gpustack-data-dir"
    assert (
        created.k8s_options.volume_mounts[0].volume_source.host_path.path
        == "/var/lib/gpustack"
    )


def test_legacy_k8s_volume_mounts_do_not_override_existing_k8s_options_mounts():
    created = ClusterCreate.model_validate(
        {
            "name": "c1",
            "provider": "Kubernetes",
            "k8s_options": {"volumeMounts": [_host_path_mount("/data")]},
            "k8s_volume_mounts": [_host_path_mount("/ignored")],
        }
    )
    assert (
        created.k8s_options.volume_mounts[0].volume_source.host_path.path == "/data"
    )


def test_create_injects_default_data_dir_mount_when_omitted():
    created = ClusterCreate(name="c1", provider=ClusterProvider.Kubernetes)
    create_update_check(ClusterProvider.Kubernetes, created)
    assert created.k8s_options.volume_mounts[0].name == "gpustack-data-dir"
    assert (
        created.k8s_options.volume_mounts[0].volume_source.host_path.path
        == "/var/lib/gpustack"
    )


def test_update_omitted_k8s_options_keeps_stored_valid_mount():
    stored = K8sOptions(
        namespace="gpustack-system-abc",
        volume_mounts=[
            K8sVolumeMount(
                name="gpustack-data-dir",
                mount_path="/var/lib/gpustack",
                volume_source=VolumeSource(
                    host_path=HostPathVolumeSource(
                        path="/var/lib/gpustack", type="DirectoryOrCreate"
                    )
                ),
            )
        ],
    )

    class _Cluster:
        k8s_options = stored

    update = ClusterUpdate(name="c1")
    ensure_k8s_data_dir_mount(update, existing=_Cluster())
    # Caller omitted k8s_options; stored mount is valid so leave the request
    # field unset and let Cluster.update keep the JSON column as-is.
    assert update.k8s_options is None


def test_update_backfills_when_stored_mount_is_missing():
    stored = K8sOptions(namespace="gpustack-system-abc")

    class _Cluster:
        k8s_options = stored

    update = ClusterUpdate(name="c1")
    ensure_k8s_data_dir_mount(update, existing=_Cluster())
    assert update.k8s_options.namespace == "gpustack-system-abc"
    assert update.k8s_options.volume_mounts[0].name == "gpustack-data-dir"


def test_first_non_hostpath_mount_is_rejected():
    update = ClusterUpdate(
        name="c1",
        k8s_options=K8sOptions(
            volume_mounts=[
                K8sVolumeMount(
                    name="models",
                    mount_path="/models",
                    volume_source=VolumeSource(
                        persistent_volume_claim=PersistentVolumeClaimVolumeSource(
                            claim_name="models-pvc"
                        )
                    ),
                )
            ]
        ),
    )
    with pytest.raises(InvalidException) as exc_info:
        ensure_k8s_data_dir_mount(update)
    assert "hostPath" in exc_info.value.message
