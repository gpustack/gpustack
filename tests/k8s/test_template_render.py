import yaml
from gpustack_runtime.detector import ManufacturerEnum

from gpustack.k8s.manifest_template import (
    WORKER_DS_BASENAME,
    TemplateConfig,
)
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic
from gpustack.schemas.clusters import (
    HostPathVolumeSource,
    ImageCredential,
    K8sOptions,
    K8sVolumeMount,
    OperatorOptions,
    VolumeSource,
)

# PCI presence labels per vendor (mirrors _MANUFACTURER_PCI_ID).
NVIDIA_PCI_LABEL = "feature.node.kubernetes.io/pci-10de.present"
ASCEND_PCI_LABEL = "feature.node.kubernetes.io/pci-19e5.present"
AMD_PCI_LABEL = "feature.node.kubernetes.io/pci-1002.present"
# CPU node label (mirrors _CPU_NODE_LABEL).
CPU_NODE_LABEL = "feature.gpustack.ai/acceleratable"


def _registration():
    return ClusterRegistrationTokenPublic(
        token="t",
        server_url="http://server",
        image="gpustack/gpustack:test",
        env={"GPUSTACK_TOKEN": "t"},
        args=[],
    )


def _config(**kwargs):
    return TemplateConfig(
        registration=_registration(),
        cluster_owner_principal_identifier="alice",
        **kwargs,
    )


def _render_docs(**kwargs):
    return [d for d in yaml.safe_load_all(_config(**kwargs).render()) if d]


def _daemonsets(docs):
    return {d["metadata"]["name"]: d for d in docs if d.get("kind") == "DaemonSet"}


def _pod_spec(ds):
    return ds["spec"]["template"]["spec"]


# ---------------------------------------------------------------------------
# CPU-only mode — always rendered, even with no GPU runtimes.
# ---------------------------------------------------------------------------


def test_no_runtime_renders_cpu_daemonset_only():
    """When no GPU runtimes are specified, only the CPU DaemonSet is rendered."""
    cfg = _config()
    assert cfg.multi_vendor_mode is False
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert set(dses.keys()) == {WORKER_DS_BASENAME}
    ds = dses[WORKER_DS_BASENAME]
    # CPU DS owns the legacy name.
    assert ds["spec"]["template"]["metadata"]["labels"] == {"app": WORKER_DS_BASENAME}
    # CPU node selector — no PCI labels.
    assert _pod_spec(ds)["nodeSelector"] == {CPU_NODE_LABEL: "false"}
    # No affinity in single-worker mode.
    assert "affinity" not in _pod_spec(ds)


def test_unknown_runtime_renders_cpu_daemonset_only():
    """UNKNOWN is treated as no GPU → only the CPU worker DS is rendered."""
    cfg = _config(runtimes=[ManufacturerEnum.UNKNOWN])
    assert cfg.multi_vendor_mode is False
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert set(dses.keys()) == {WORKER_DS_BASENAME}


def test_cpu_daemonset_no_vendor_volumes():
    """CPU DaemonSet has no vendor-specific volumeMounts or volumes."""
    docs = _render_docs()
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    mounts = _pod_spec(ds)["containers"][0].get("volumeMounts") or []
    assert mounts == []
    volumes = _pod_spec(ds).get("volumes") or []
    assert volumes == []


def test_cpu_daemonset_no_runtime_class():
    """CPU DaemonSet has no runtimeClassName."""
    docs = _render_docs()
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    assert "runtimeClassName" not in _pod_spec(ds)


def test_cpu_daemonset_merges_base_node_selector():
    """Base nodeSelector is merged with the CPU label."""
    values = K8sOptions(node_selector={"env": "prod"})
    docs = _render_docs(k8s_options=values)
    ds = _daemonsets(docs)[WORKER_DS_BASENAME]
    assert _pod_spec(ds)["nodeSelector"] == {
        "env": "prod",
        CPU_NODE_LABEL: "false",
    }


# ---------------------------------------------------------------------------
# Single-GPU-runtime mode — CPU DS + one GPU vendor DS. Label-minimal,
# affinity-free, with the vendor's PCI nodeSelector.
# ---------------------------------------------------------------------------


def test_single_gpu_runtime_renders_cpu_and_gpu_daemonsets():
    cfg = _config(runtimes=[ManufacturerEnum.NVIDIA])
    assert cfg.multi_vendor_mode is True
    docs = [d for d in yaml.safe_load_all(cfg.render()) if d]
    dses = _daemonsets(docs)
    assert set(dses.keys()) == {
        WORKER_DS_BASENAME,
        f"{WORKER_DS_BASENAME}-nvidia",
    }


def test_single_gpu_runtime_gpu_daemonset_has_vendor_blocks():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA])
    dses = _daemonsets(docs)
    gpu_ds = dses[f"{WORKER_DS_BASENAME}-nvidia"]
    # Vendor specifics still apply.
    assert _pod_spec(gpu_ds).get("runtimeClassName") == "nvidia"
    # PCI nodeSelector pins the DS to nvidia nodes.
    assert _pod_spec(gpu_ds)["nodeSelector"] == {NVIDIA_PCI_LABEL: "true"}


def test_single_gpu_runtime_ascend_keeps_vendor_volume_mounts():
    docs = _render_docs(runtimes=[ManufacturerEnum.ASCEND])
    dses = _daemonsets(docs)
    gpu_ds = dses[f"{WORKER_DS_BASENAME}-ascend"]
    mounts = _pod_spec(gpu_ds)["containers"][0].get("volumeMounts") or []
    assert {m["name"] for m in mounts} == {
        "gpustack-ascend-driver",
        "gpustack-ascend-toolkit",
    }


def test_single_gpu_runtime_service_uses_common_selector():
    """With CPU + 1 GPU, multi_vendor_mode is true so the Service uses the
    common component label selector."""
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA])
    svc = next(d for d in docs if d.get("kind") == "Service")
    assert svc["spec"]["selector"] == {"app.kubernetes.io/component": "worker"}


def test_repeated_runtime_collapses_to_two_daemonsets():
    """Duplicate GPU runtime does not produce an extra DS — one CPU + one GPU."""
    cfg = _config(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.NVIDIA])
    assert cfg.multi_vendor_mode is True
    dses = _daemonsets([d for d in yaml.safe_load_all(cfg.render()) if d])
    assert set(dses.keys()) == {
        WORKER_DS_BASENAME,
        f"{WORKER_DS_BASENAME}-nvidia",
    }


def test_single_gpu_runtime_merges_base_node_selector_with_pci_label():
    values = K8sOptions(node_selector={"env": "prod"})
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=values)
    dses = _daemonsets(docs)
    # GPU DS gets base + PCI label.
    assert _pod_spec(dses[f"{WORKER_DS_BASENAME}-nvidia"])["nodeSelector"] == {
        "env": "prod",
        NVIDIA_PCI_LABEL: "true",
    }
    # CPU DS gets base + CPU label.
    assert _pod_spec(dses[WORKER_DS_BASENAME])["nodeSelector"] == {
        "env": "prod",
        CPU_NODE_LABEL: "false",
    }


# ---------------------------------------------------------------------------
# Multi-vendor mode — CPU DS + one DS per GPU runtime, plus the cross-DS
# safety net (component/runtime labels + podAntiAffinity).
# ---------------------------------------------------------------------------


def test_multi_vendor_emits_one_daemonset_per_runtime_plus_cpu():
    cfg = _config(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
    assert cfg.multi_vendor_mode is True
    docs = [d for d in yaml.safe_load_all(cfg.render()) if d]
    dses = _daemonsets(docs)
    # CPU DS + ascend DS + nvidia DS. All GPU vendors get suffixed names.
    assert set(dses.keys()) == {
        WORKER_DS_BASENAME,
        f"{WORKER_DS_BASENAME}-ascend",
        f"{WORKER_DS_BASENAME}-nvidia",
    }


def test_daemonset_names_are_request_order_independent():
    """The rendered DS name set is identical no matter what order the GPU
    runtimes were requested in (avoids selector churn on re-apply)."""
    names_forward = set(
        _daemonsets(
            _render_docs(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
        )
    )
    names_reversed = set(
        _daemonsets(
            _render_docs(runtimes=[ManufacturerEnum.ASCEND, ManufacturerEnum.NVIDIA])
        )
    )
    assert (
        names_forward
        == names_reversed
        == {
            WORKER_DS_BASENAME,
            f"{WORKER_DS_BASENAME}-ascend",
            f"{WORKER_DS_BASENAME}-nvidia",
        }
    )


def test_multi_vendor_all_daemonsets_have_pod_anti_affinity():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
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
        # No nodeAffinity any more (the CPU DoesNotExist block is gone).
        assert "nodeAffinity" not in _pod_spec(ds)["affinity"]


def test_multi_vendor_each_gpu_daemonset_pins_its_vendor_pci_label():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
    dses = _daemonsets(docs)
    # GPU DSes have their PCI labels.
    assert _pod_spec(dses[f"{WORKER_DS_BASENAME}-ascend"])["nodeSelector"] == {
        ASCEND_PCI_LABEL: "true"
    }
    assert _pod_spec(dses[f"{WORKER_DS_BASENAME}-nvidia"])["nodeSelector"] == {
        NVIDIA_PCI_LABEL: "true"
    }
    # CPU DS has the CPU label, not a PCI label.
    assert _pod_spec(dses[WORKER_DS_BASENAME])["nodeSelector"] == {
        CPU_NODE_LABEL: "false"
    }


def test_multi_vendor_pod_labels_include_common_component_and_runtime_tag():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
    dses = _daemonsets(docs)

    # NVIDIA DS labels.
    nvidia_labels = dses[f"{WORKER_DS_BASENAME}-nvidia"]["spec"]["template"][
        "metadata"
    ]["labels"]
    assert nvidia_labels["app.kubernetes.io/component"] == "worker"
    assert nvidia_labels["gpustack.io/runtime"] == "nvidia"
    assert nvidia_labels["app"] == f"{WORKER_DS_BASENAME}-nvidia"

    # ASCEND DS labels.
    ascend_labels = dses[f"{WORKER_DS_BASENAME}-ascend"]["spec"]["template"][
        "metadata"
    ]["labels"]
    assert ascend_labels["gpustack.io/runtime"] == "ascend"
    assert ascend_labels["app"] == f"{WORKER_DS_BASENAME}-ascend"

    # CPU DS labels — uses the legacy DS name but carries the cpu runtime tag.
    cpu_labels = dses[WORKER_DS_BASENAME]["spec"]["template"]["metadata"]["labels"]
    assert cpu_labels["gpustack.io/runtime"] == "cpu"
    assert cpu_labels["app"] == WORKER_DS_BASENAME


def test_multi_vendor_service_uses_common_component_label():
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND])
    svc = next(d for d in docs if d.get("kind") == "Service")
    assert svc["spec"]["selector"] == {"app.kubernetes.io/component": "worker"}


# ---------------------------------------------------------------------------
# Image pull secrets — should be referenced by all DaemonSets (CPU + GPU).
# ---------------------------------------------------------------------------


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
    )
    docs = _render_docs(
        runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.ASCEND],
        k8s_options=values,
    )
    expected_refs = [
        {"name": "gpustack-image-pull-secret-0"},
        {"name": "gpustack-image-pull-secret-1"},
    ]
    dses = _daemonsets(docs)
    # CPU DS + nvidia DS + ascend DS — all must reference the same secrets.
    assert set(dses.keys()) == {
        WORKER_DS_BASENAME,
        f"{WORKER_DS_BASENAME}-nvidia",
        f"{WORKER_DS_BASENAME}-ascend",
    }
    for ds in dses.values():
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
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=values)
    secret = next(
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"] == "gpustack-image-pull-secret-0"
        and d["metadata"]["namespace"] == "gpustack-system"
    )
    decoded = json.loads(base64.b64decode(secret["data"][".dockerconfigjson"]))
    assert decoded == {"auths": {}}
    # The Secret is still referenced by all worker DSes so users can patch in
    # credentials post-install without re-applying the manifest.
    dses = _daemonsets(docs)
    for ds in dses.values():
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
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=K8sOptions())
    image_secrets = [
        d
        for d in docs
        if d.get("kind") == "Secret"
        and d["metadata"]["name"].startswith("gpustack-image-pull-secret-")
    ]
    assert image_secrets == []
    for ds in _daemonsets(docs).values():
        assert "imagePullSecrets" not in _pod_spec(ds)


# ---------------------------------------------------------------------------
# Volume mounts — applied to all DaemonSets (CPU + GPU).
# ---------------------------------------------------------------------------


def test_user_volume_mounts_flow_through_via_k8s_options():
    """K8sOptions volumeMounts are applied to all DaemonSets."""
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
    dses = _daemonsets(docs)
    for ds_name in dses:
        ds = dses[ds_name]
        mount_names = {
            m["name"]
            for m in (_pod_spec(ds)["containers"][0].get("volumeMounts") or [])
        }
        volume_names = {v["name"] for v in (_pod_spec(ds).get("volumes") or [])}
        assert "data-dir" in mount_names
        assert "data-dir" in volume_names


def test_multi_vendor_volume_mounts_applied_only_to_owning_vendor():
    docs = _render_docs(
        runtimes=[ManufacturerEnum.ASCEND, ManufacturerEnum.AMD],
    )
    dses = _daemonsets(docs)

    def mount_names(ds):
        return {
            m["name"]
            for m in (_pod_spec(ds)["containers"][0].get("volumeMounts") or [])
        }

    # CPU DS has no vendor-specific mounts.
    assert mount_names(dses[WORKER_DS_BASENAME]) == set()
    # Each GPU DS has its own vendor mounts.
    assert mount_names(dses[f"{WORKER_DS_BASENAME}-amd"]) == {"gpustack-amd-driver"}
    assert mount_names(dses[f"{WORKER_DS_BASENAME}-ascend"]) == {
        "gpustack-ascend-driver",
        "gpustack-ascend-toolkit",
    }


# ---------------------------------------------------------------------------
# GPUSTACK_CONTAINER_NAMESPACE — derived from the gpustack image, not the
# operator image (the operator image may live in a different namespace).
# ---------------------------------------------------------------------------


def _config_with_image(image, **kwargs):
    registration = ClusterRegistrationTokenPublic(
        token="t",
        server_url="http://server",
        image=image,
        env={"GPUSTACK_TOKEN": "t"},
        args=[],
    )
    return TemplateConfig(
        registration=registration,
        cluster_owner_principal_identifier="alice",
        **kwargs,
    )


def test_container_namespace_default_gpustack_is_suppressed():
    # gpustack/gpustack:test → namespace "gpustack" is the built-in default,
    # so the operator already knows it and the env var is omitted.
    cfg = _config_with_image("gpustack/gpustack:test")
    assert cfg.container_namespace is None


def test_container_namespace_from_custom_gpustack_image():
    cfg = _config_with_image("myorg/gpustack:test")
    assert cfg.container_namespace == "myorg"


def test_container_namespace_strips_registry_and_keeps_deep_namespace():
    cfg = _config_with_image(
        "reg.io/myorg/sub/gpustack:v1",
        system_default_container_registry="reg.io",
    )
    assert cfg.container_namespace == "myorg/sub"


def test_container_namespace_ignores_operator_image_namespace():
    # The operator image lives in a different namespace than the gpustack
    # image; the env var must follow the gpustack image so the operator
    # composes sibling references against the right namespace.
    cfg = _config_with_image(
        "myorg/gpustack:test",
        k8s_options=K8sOptions(operatorImage="otherns/gpustack-operator:test"),
    )
    assert cfg.container_namespace == "myorg"


def test_container_namespace_strips_embedded_registry_from_image_override():
    # image_name_override may carry a full reference with an embedded registry
    # and no system_default_container_registry set. The registry (first segment
    # with a ".") must not leak into the namespace; quay.io/gpustack/gpustack
    # resolves to the default "gpustack" namespace → suppressed.
    cfg = _config_with_image("quay.io/gpustack/gpustack:dev")
    assert cfg.container_namespace is None


def test_container_namespace_strips_embedded_registry_with_port():
    cfg = _config_with_image("myreg:5000/org/gpustack:dev")
    assert cfg.container_namespace == "org"


# ---------------------------------------------------------------------------
# Operator env vars from k8s_options.operator.env
# ---------------------------------------------------------------------------


def _operator_deployment_env(docs):
    """Extract the operator Deployment container env list from the embedded
    ConfigMap's template.yaml data (ytt-processed)."""
    cm = next(
        d
        for d in docs
        if d.get("kind") == "ConfigMap"
        and d["metadata"]["name"] == "gpustack-operator-worker-deployment"
    )
    template_yaml = cm["data"]["template.yaml"]
    # The template.yaml is ytt-templated, but the Deployment portion is plain
    # YAML after the last ytt directive. Parse all YAML docs and find the
    # Deployment.
    inner_docs = [d for d in yaml.safe_load_all(template_yaml) if d]
    deploy = next(d for d in inner_docs if d.get("kind") == "Deployment")
    return deploy["spec"]["template"]["spec"]["containers"][0].get("env") or []


def test_operator_env_vars_rendered_in_deployment():
    """Extra env vars from k8s_options.operator.env appear in the operator
    Deployment container."""
    values = K8sOptions(
        operator=OperatorOptions(env={"MY_VAR": "my_value", "OTHER_VAR": "other"}),
    )
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA], k8s_options=values)
    env = _operator_deployment_env(docs)
    env_map = {e["name"]: e.get("value") for e in env if "value" in e}
    assert env_map.get("MY_VAR") == "my_value"
    assert env_map.get("OTHER_VAR") == "other"


def test_operator_env_vars_absent_when_not_set():
    """No extra env vars when k8s_options.operator is unset."""
    docs = _render_docs(runtimes=[ManufacturerEnum.NVIDIA])
    env = _operator_deployment_env(docs)
    env_names = {e["name"] for e in env}
    assert "MY_VAR" not in env_names
    assert "OTHER_VAR" not in env_names
