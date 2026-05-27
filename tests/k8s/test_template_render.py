import yaml
from gpustack_runtime.detector import ManufacturerEnum

from gpustack.k8s.manifest_template import (
    CPU_WORKER_NAME,
    WORKER_DS_BASENAME,
    TemplateConfig,
)
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic
from gpustack.schemas.clusters import (
    HostPathVolumeSource,
    ImageCredential,
    K8sOptions,
    K8sOptionsOverride,
    K8sVolumeMount,
    VolumeSource,
)


def _registration():
    return ClusterRegistrationTokenPublic(
        token="t",
        server_url="http://server",
        image="gpustack/gpustack:test",
        env={"GPUSTACK_TOKEN": "t"},
        args=[],
        operator_image="gpustack/gpustack-operator:test",
    )


def _config(**kwargs):
    return TemplateConfig(
        registration=_registration(),
        cluster_owner_principal_name="alice",
        **kwargs,
    )


def _render_docs(**kwargs):
    return [d for d in yaml.safe_load_all(_config(**kwargs).render()) if d]


def _daemonsets(docs):
    return {d["metadata"]["name"]: d for d in docs if d.get("kind") == "DaemonSet"}


def _pod_spec(ds):
    return ds["spec"]["template"]["spec"]


# ---------------------------------------------------------------------------
# Single-vendor / no-runtime mode — must stay byte-compatible with v1
# (one DaemonSet named gpustack-worker, no extra labels, no affinity).
# ---------------------------------------------------------------------------


def test_no_runtime_renders_single_legacy_daemonset():
    cfg = _config()
    assert cfg.multi_vendor_mode is False
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert list(dses.keys()) == [WORKER_DS_BASENAME]
    ds = dses[WORKER_DS_BASENAME]
    assert ds["spec"]["template"]["metadata"]["labels"] == {"app": WORKER_DS_BASENAME}
    assert "affinity" not in _pod_spec(ds)


def test_single_runtime_renders_single_legacy_daemonset_with_vendor_blocks():
    cfg = _config(runtimes=[ManufacturerEnum.NVIDIA])
    assert cfg.multi_vendor_mode is False
    docs = [d for d in yaml.safe_load_all(cfg.render()) if d]
    dses = _daemonsets(docs)
    assert list(dses.keys()) == [WORKER_DS_BASENAME]
    ds = dses[WORKER_DS_BASENAME]
    # Pod labels remain legacy single-label so apply does not trigger an
    # unnecessary rolling update on existing v1 clusters.
    assert ds["spec"]["template"]["metadata"]["labels"] == {"app": WORKER_DS_BASENAME}
    # No podAntiAffinity / nodeAffinity in single-vendor mode.
    assert "affinity" not in _pod_spec(ds)
    # Vendor specifics still apply.
    assert _pod_spec(ds).get("runtimeClassName") == "nvidia"


def test_single_runtime_ascend_keeps_vendor_volume_mounts():
    docs = _render_docs(runtimes=[ManufacturerEnum.ASCEND])
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    mounts = _pod_spec(ds)["containers"][0].get("volumeMounts") or []
    assert {m["name"] for m in mounts} == {
        "gpustack-ascend-driver",
        "gpustack-ascend-toolkit",
    }


def test_single_vendor_service_uses_legacy_selector():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA])
    svc = next(d for d in docs if d.get("kind") == "Service")
    assert svc["spec"]["selector"] == {"app": WORKER_DS_BASENAME}


def test_unknown_runtime_falls_back_to_single_legacy_daemonset():
    """UNKNOWN is treated as no GPU → emit only the legacy CPU DS."""
    cfg = _config(runtimes=[ManufacturerEnum.UNKNOWN])
    assert cfg.multi_vendor_mode is False
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert list(dses.keys()) == [WORKER_DS_BASENAME]


def test_repeated_runtime_collapses_to_single_vendor_mode():
    cfg = _config(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.NVIDIA])
    assert cfg.multi_vendor_mode is False
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert list(dses.keys()) == [WORKER_DS_BASENAME]


def test_single_runtime_merges_base_and_override_node_selector():
    values = K8sOptions(
        node_selector={"env": "prod"},
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            )
        },
    )
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=values)
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    assert _pod_spec(ds)["nodeSelector"] == {
        "env": "prod",
        "nvidia.com/gpu": "true",
    }


# ---------------------------------------------------------------------------
# Multi-vendor mode — per-vendor DS + always-on CPU DS + safety nets.
# ---------------------------------------------------------------------------


def _multi_vendor_values():
    return K8sOptions(
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            ),
        }
    )


def test_multi_vendor_emits_cpu_plus_each_vendor_daemonset():
    cfg = _config(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    assert cfg.multi_vendor_mode is True
    docs = [d for d in yaml.safe_load_all(cfg.render()) if d]
    dses = _daemonsets(docs)
    assert set(dses.keys()) == {
        WORKER_DS_BASENAME,
        f"{WORKER_DS_BASENAME}-nvidia",
        f"{WORKER_DS_BASENAME}-ascend",
    }


def test_multi_vendor_all_daemonsets_have_pod_anti_affinity():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    for ds in _daemonsets(docs).values():
        terms = _pod_spec(ds)["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        assert len(terms) == 1
        term = terms[0]
        assert term["topologyKey"] == "kubernetes.io/hostname"
        assert term["labelSelector"]["matchLabels"] == {
            "app.kubernetes.io/component": "worker"
        }
        assert term["namespaceSelector"] == {}


def test_multi_vendor_cpu_ds_has_node_affinity_does_not_exist():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    cpu_ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    exprs = _pod_spec(cpu_ds)["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert {e["key"] for e in exprs} == {
        "nvidia.com/gpu",
        "huawei.com/Ascend910",
    }
    assert {e["operator"] for e in exprs} == {"DoesNotExist"}


def test_multi_vendor_gpu_ds_has_no_node_affinity():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    nvidia_ds = _daemonsets(docs)[f"{WORKER_DS_BASENAME}-nvidia"]
    assert "nodeAffinity" not in _pod_spec(nvidia_ds).get("affinity", {})


def test_multi_vendor_pod_labels_include_common_component_and_runtime_tag():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    dses = _daemonsets(docs)
    nvidia_labels = dses[f"{WORKER_DS_BASENAME}-nvidia"]["spec"]["template"][
        "metadata"
    ]["labels"]
    assert nvidia_labels["app.kubernetes.io/component"] == "worker"
    assert nvidia_labels["gpustack.io/runtime"] == "nvidia"
    assert nvidia_labels["app"] == f"{WORKER_DS_BASENAME}-nvidia"

    cpu_labels = dses[WORKER_DS_BASENAME]["spec"]["template"]["metadata"]["labels"]
    assert cpu_labels["gpustack.io/runtime"] == CPU_WORKER_NAME
    assert cpu_labels["app"] == WORKER_DS_BASENAME


def test_multi_vendor_service_uses_common_component_label():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=_multi_vendor_values(),
    )
    svc = next(d for d in docs if d.get("kind") == "Service")
    assert svc["spec"]["selector"] == {"app.kubernetes.io/component": "worker"}


def test_image_credentials_materialise_to_dockerconfigjson_secrets():
    """Each ImageCredential renders a kubernetes.io/dockerconfigjson Secret
    in the gpustack-system namespace, with a deterministic index-based name."""
    import base64
    import json

    values = K8sOptions(
        image_credentials=[
            ImageCredential(
                registry="harbor.example.com",
                username="alice",
                password="s3cret",
            ),
            ImageCredential(registry="ghcr.io", username="bob", password="token-xyz"),
        ],
    )
    docs = _render_docs(k8s_options=values)
    secrets = {
        d["metadata"]["name"]: d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"].startswith("gpustack-image-pull-secret-")
        and d["metadata"]["namespace"] == "gpustack-system"
    }
    assert set(secrets.keys()) == {
        "gpustack-image-pull-secret-0",
        "gpustack-image-pull-secret-1",
    }
    s0 = secrets["gpustack-image-pull-secret-0"]
    assert s0["type"] == "kubernetes.io/dockerconfigjson"
    decoded = json.loads(base64.b64decode(s0["data"][".dockerconfigjson"]))
    assert "harbor.example.com" in decoded["auths"]
    assert decoded["auths"]["harbor.example.com"]["username"] == "alice"
    assert decoded["auths"]["harbor.example.com"]["password"] == "s3cret"
    # The `auth` field is the base64 of "username:password"
    expected_auth = base64.b64encode(b"alice:s3cret").decode()
    assert decoded["auths"]["harbor.example.com"]["auth"] == expected_auth


def test_image_credentials_emitted_once_in_shared_namespace():
    """Worker DaemonSets and the operator Job share the gpustack-system
    namespace, so each ImageCredential renders exactly one Secret there."""
    values = K8sOptions(
        image_credentials=[
            ImageCredential(
                registry="harbor.example.com",
                username="alice",
                password="s3cret",
            )
        ]
    )
    docs = _render_docs(k8s_options=values)
    secrets = [
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"] == "gpustack-image-pull-secret-0"
    ]
    namespaces = [s["metadata"]["namespace"] for s in secrets]
    assert namespaces == ["gpustack-system"]


def test_image_credentials_referenced_by_all_worker_daemonsets():
    values = K8sOptions(
        image_credentials=[
            ImageCredential(
                registry="harbor.example.com",
                username="alice",
                password="s3cret",
            ),
            ImageCredential(registry="ghcr.io", username="bob", password="token-xyz"),
        ],
        gpu_vendor_overrides={
            ManufacturerEnum.NVIDIA: K8sOptionsOverride(
                node_selector={"nvidia.com/gpu": "true"}
            ),
            ManufacturerEnum.ASCEND: K8sOptionsOverride(
                node_selector={"huawei.com/Ascend910": "true"}
            ),
        },
    )
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=values,
    )
    expected_refs = [
        {"name": "gpustack-image-pull-secret-0"},
        {"name": "gpustack-image-pull-secret-1"},
    ]
    for ds in _daemonsets(docs).values():
        assert _pod_spec(ds).get("imagePullSecrets") == expected_refs


def test_image_credentials_without_username_password_emit_placeholder_secret():
    """An ImageCredential without username/password renders a placeholder
    Secret with an empty ``auths`` map — useful for pre-creating a Secret
    reference that the user (or another controller) patches in-cluster."""
    import base64
    import json

    values = K8sOptions(
        image_credentials=[ImageCredential(registry="docker.io")],
    )
    docs = _render_docs(k8s_options=values)
    secret = next(
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"] == "gpustack-image-pull-secret-0"
        and d["metadata"]["namespace"] == "gpustack-system"
    )
    decoded = json.loads(base64.b64decode(secret["data"][".dockerconfigjson"]))
    assert decoded == {"auths": {}}
    # The Secret is still referenced by worker DSes so users can patch in
    # credentials post-install without re-applying the manifest.
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    assert _pod_spec(ds)["imagePullSecrets"] == [
        {"name": "gpustack-image-pull-secret-0"}
    ]


def test_image_credentials_with_only_username_falls_back_to_placeholder():
    """Half-configured credentials (username without password, or vice
    versa) fall back to the empty-auths placeholder rather than producing
    a broken auth entry."""
    import base64
    import json

    values = K8sOptions(
        image_credentials=[ImageCredential(registry="docker.io", username="alice")],
    )
    docs = _render_docs(k8s_options=values)
    secret = next(
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"] == "gpustack-image-pull-secret-0"
        and d["metadata"]["namespace"] == "gpustack-system"
    )
    decoded = json.loads(base64.b64decode(secret["data"][".dockerconfigjson"]))
    assert decoded == {"auths": {}}


def test_no_image_credentials_emits_no_secret_or_reference():
    docs = _render_docs(k8s_options=K8sOptions())
    image_secrets = [
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"].startswith("gpustack-image-pull-secret-")
    ]
    assert image_secrets == []
    for ds in _daemonsets(docs).values():
        assert "imagePullSecrets" not in _pod_spec(ds)


def test_user_volume_mounts_flow_through_via_k8s_options():
    """K8sOptions now carries the cluster's volumeMounts (previously the
    top-level k8s_volume_mounts field). Each rendered DS should pick them
    up."""
    values = K8sOptions(
        volume_mounts=[
            K8sVolumeMount(
                name="data-dir",
                mount_path="/var/lib/gpustack",
                volume_source=VolumeSource(
                    host_path=HostPathVolumeSource(
                        path="/var/lib/gpustack", type="DirectoryOrCreate"
                    )
                ),
            )
        ]
    )
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=values)
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    mount_names = {
        m["name"] for m in (_pod_spec(ds)["containers"][0].get("volumeMounts") or [])
    }
    volume_names = {v["name"] for v in (_pod_spec(ds).get("volumes") or [])}
    assert "data-dir" in mount_names
    assert "data-dir" in volume_names


def test_multi_vendor_volume_mounts_applied_only_to_owning_vendor():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.ASCEND, ManufacturerEnum.AMD],
        k8s_options=K8sOptions(
            gpu_vendor_overrides={
                ManufacturerEnum.ASCEND: K8sOptionsOverride(
                    node_selector={"huawei.com/Ascend910": "true"}
                ),
                ManufacturerEnum.AMD: K8sOptionsOverride(
                    node_selector={"amd.com/gpu": "true"}
                ),
            }
        ),
    )
    dses = _daemonsets(docs)

    def mount_names(ds):
        return {
            m["name"]
            for m in (_pod_spec(ds)["containers"][0].get("volumeMounts") or [])
        }

    assert mount_names(dses[f"{WORKER_DS_BASENAME}-ascend"]) == {
        "gpustack-ascend-driver",
        "gpustack-ascend-toolkit",
    }
    assert mount_names(dses[f"{WORKER_DS_BASENAME}-amd"]) == {"gpustack-amd-driver"}
    assert mount_names(dses[WORKER_DS_BASENAME]) == set()
