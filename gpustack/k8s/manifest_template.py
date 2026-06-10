import jinja2
import base64
import json
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, computed_field

from gpustack import __operator_version__
from gpustack.gpu_instances.cluster_apis_util import get_namespace_name
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic, K8sOptions
from gpustack_runtime.detector import ManufacturerEnum


_DEFAULT_OPERATOR_IMAGE = f"gpustack/gpustack-operator:{__operator_version__}"
_DEFAULT_CONTAINER_NAMESPACE = "gpustack"
_DEFAULT_CLUSTER_NAMESPACE = "gpustack-system"


WORKER_DS_BASENAME = "gpustack-worker"
IMAGE_PULL_SECRET_NAME_PREFIX = "gpustack-image-pull-secret"

# PCI vendor IDs per GPU manufacturer, mirroring the operator's
# ``_ManufacturerPciIDMap``. Each worker DaemonSet derives a nodeSelector
# label ``feature.node.kubernetes.io/pci-<id>.present: "true"`` from this map
# (NFD advertises these labels), so a DS only lands on nodes carrying that
# vendor's device.
_MANUFACTURER_PCI_ID: Dict[ManufacturerEnum, str] = {
    ManufacturerEnum.AMD: "1002",
    ManufacturerEnum.ASCEND: "19e5",
    ManufacturerEnum.CAMBRICON: "cabc",
    ManufacturerEnum.HYGON: "1d94",
    ManufacturerEnum.ILUVATAR: "1e3e",
    ManufacturerEnum.METAX: "9999",
    ManufacturerEnum.MTHREADS: "1ed5",
    ManufacturerEnum.NVIDIA: "10de",
    ManufacturerEnum.THEAD: "1ded",
}
_PCI_NODE_LABEL = "feature.node.kubernetes.io/pci-{pci_id}.present"
# Node label that identifies non-acceleratable (CPU-only) nodes.
_CPU_NODE_LABEL = "feature.gpustack.ai/acceleratable"
# Canonical, request-order-independent runtime ordering. Used for deterministic
# output ordering of GPU vendor DaemonSets regardless of the order they were
# requested in.
_RUNTIME_ORDER: Dict[ManufacturerEnum, int] = {
    runtime: index for index, runtime in enumerate(_MANUFACTURER_PCI_ID)
}


class ImagePullSecretRenderSpec(BaseModel):
    """
    One materialised ``kubernetes.io/dockerconfigjson`` Secret derived from
    a single ``K8sOptions.image_credentials`` entry. The name is index-based
    so the same name is rendered into both the Secret (image_pull_secrets.jinja)
    and each worker DaemonSet's imagePullSecrets list.
    """

    name: str
    registry: str
    dockerconfigjson_b64: str


class WorkerRenderSpec(BaseModel):
    """
    Per-DaemonSet render data. One entry is always produced for the CPU worker
    (named ``gpustack-worker``), plus one entry per requested GPU runtime
    (each named ``gpustack-worker-<runtime>``).

    The grouping mirrors how Helm chart values are typically organized
    (``.Values.worker.*``), so a future chart migration maps 1:1 onto these
    entries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    ds_name: str
    # Runtime tag consumed by per-vendor jinja branches (volume mounts /
    # volumes / runtimeClassName / vendor-specific env).
    runtime: str = ""
    # Base nodeSelector merged with the runtime's PCI-presence label.
    node_selector: Optional[Dict[str, str]] = None


class TemplateConfig(ClusterRegistrationTokenPublic):
    # cluster owner namespace, defaults to
    # "gpustack-{cluster_owner_principal_identifier}", used to place the
    # Kubernetes resources for the cluster owner.
    cluster_owner_namespace: Optional[str] = None
    cluster_owner_principal_identifier: Optional[str] = None
    runtimes: Optional[List[ManufacturerEnum]] = None
    k8s_options: Optional[K8sOptions] = None
    # Cluster-level default container registry (mirrors
    # ``clusters.system_default_container_registry``). Drives the operator
    # image registry prefix and the GPUSTACK_CONTAINER_REGISTRY env var
    # surfaced to the operator at runtime.
    system_default_container_registry: Optional[str] = None
    workers: List[WorkerRenderSpec] = []
    # Pre-computed Secret render data, one per K8sOptions.image_credentials
    # entry. Both image_pull_secrets.jinja (Secret resource) and the
    # daemonset.jinja imagePullSecrets reference iterate this list, so the
    # Secret name is the single source of truth.
    image_pull_secrets: List[ImagePullSecretRenderSpec] = []

    @computed_field
    @property
    def multi_vendor_mode(self) -> bool:
        """
        True when 2+ worker DaemonSets are rendered (CPU + any GPU vendor, or
        2+ GPU vendors). Adds the multi-DS safety net (per-pod
        ``app.kubernetes.io/component``/``gpustack.io/runtime`` labels +
        cross-DS podAntiAffinity so at most one worker pod lands per node).
        Single-worker output stays label-minimal and affinity-free.
        """
        return len(self.workers) >= 2

    @computed_field
    @property
    def namespace(self) -> str:
        """
        Kubernetes namespace this cluster's manifests render into. Reads
        ``k8s_options.namespace`` — which the routes layer pre-resolves from
        the server-wide ``Config.namespace`` when the cluster doesn't override
        it — and falls back to the built-in ``gpustack-system`` default
        otherwise (e.g. in unit tests). Referenced as ``config.namespace``
        across every cluster-level jinja template.
        """
        if self.k8s_options and self.k8s_options.namespace:
            return self.k8s_options.namespace
        return _DEFAULT_CLUSTER_NAMESPACE

    @computed_field
    @property
    def operator_image(self) -> str:
        """
        Fully-qualified operator image reference for ``operator.jinja``.
        Reads ``k8s_options.operator_image`` — which the routes layer
        pre-resolves from the server-wide ``Config.operator_image`` (settable
        via ``GPUSTACK_OPERATOR_IMAGE``) when the cluster doesn't override it —
        and falls back to the built-in default otherwise (e.g. in unit tests);
        prefixes the cluster's container registry when one is configured and
        the image doesn't already carry one.
        """
        image = (
            self.k8s_options.operator_image if self.k8s_options else None
        ) or _DEFAULT_OPERATOR_IMAGE
        registry = (self.system_default_container_registry or "").strip().rstrip("/")
        if registry and not image.startswith(registry + "/"):
            return f"{registry}/{image}"
        return image

    @computed_field
    @property
    def container_namespace(self) -> Optional[str]:
        """
        Namespace segment inferred from the resolved gpustack image — used by
        the operator runtime (``GPUSTACK_CONTAINER_NAMESPACE``) to compose
        sibling image references. The operator image may live elsewhere, so the
        namespace must come from the gpustack image (``self.image``) instead.

        Strip the registry prefix first so the leading segment isn't mistaken
        for a namespace, then take everything up to the final ``/`` (the
        trailing ``<name>:<tag>`` segment is discarded). The registry can be
        either the configured ``system_default_container_registry`` or one
        embedded directly in the reference (e.g. via an ``image_name_override``
        like ``quay.io/gpustack/gpustack:dev``); the latter is detected with
        the same heuristic as ``apply_registry_override_to_image`` — the first
        path segment is a registry when it contains ``.`` or ``:`` or equals
        ``localhost``. Suppressed when the namespace is the built-in
        ``gpustack`` default since the operator already knows that one.
        """
        image = self.image
        registry = (self.system_default_container_registry or "").strip().rstrip("/")
        if registry and image.startswith(registry + "/"):
            image = image[len(registry) + 1 :]
        first, sep, rest = image.partition("/")
        if sep and ("." in first or ":" in first or first == "localhost"):
            image = rest
        if "/" not in image:
            return None
        namespace = image.rsplit("/", 1)[0]
        if namespace == _DEFAULT_CONTAINER_NAMESPACE:
            return None
        return namespace

    @computed_field
    @property
    def operator_instance_access_static_address(self) -> Optional[str]:
        if (
            self.k8s_options
            and self.k8s_options.gpu_instance_options
            and self.k8s_options.gpu_instance_options.gpu_instances_access_static_address
        ):
            return (
                self.k8s_options.gpu_instance_options.gpu_instances_access_static_address
            )
        return None

    @computed_field
    @property
    def operator_env(self) -> Optional[Dict[str, str]]:
        """
        Extra env vars for the operator container, sourced from
        ``k8s_options.operator.env``. Returns None when no extra env vars
        are configured.
        """
        if (
            self.k8s_options
            and self.k8s_options.operator
            and self.k8s_options.operator.env
        ):
            return self.k8s_options.operator.env
        return None

    def render(self) -> str:
        def b64encode(value):
            return base64.b64encode(value.encode("utf-8")).decode("utf-8")

        def to_yaml(value):
            if hasattr(value, "model_dump"):
                value = value.model_dump(by_alias=True, exclude_none=True)
            elif isinstance(value, list):
                value = [
                    (
                        v.model_dump(by_alias=True, exclude_none=True)
                        if hasattr(v, "model_dump")
                        else v
                    )
                    for v in value
                ]

            dumped = yaml.dump(value, default_flow_style=False)
            if dumped.endswith("...\n"):
                dumped = dumped[:-4]

            return dumped.strip()

        with pkg_resources.path("gpustack.k8s", "manifests.jinja") as manifest_path:
            with manifest_path.open(encoding="utf-8") as f:
                template_data = f.read()
        with pkg_resources.path("gpustack.k8s", "image_pull_secrets.jinja") as ips_path:
            with ips_path.open(encoding="utf-8") as f:
                ips_data = f.read()
        with pkg_resources.path("gpustack.k8s", "daemonset.jinja") as daemon_set_path:
            with daemon_set_path.open(encoding="utf-8") as f:
                daemon_set_data = f.read()
        with pkg_resources.path("gpustack.k8s", "operator.jinja") as operator_path:
            with operator_path.open(encoding="utf-8") as f:
                operator_data = f.read()
        env = jinja2.Environment()
        env.filters["b64encode"] = b64encode
        env.filters["to_yaml"] = to_yaml
        rendered = env.from_string(template_data).render(config=self)
        image_pull_secrets = env.from_string(ips_data).render(config=self)
        daemon_set = env.from_string(daemon_set_data).render(config=self)
        operator = env.from_string(operator_data).render(config=self)
        return "\n".join([rendered, image_pull_secrets, daemon_set, operator])

    def __init__(
        self, registration: Optional[ClusterRegistrationTokenPublic] = None, **data
    ):
        if registration is not None:
            base_data = registration.model_dump()
            base_data.update(data)
            super().__init__(**base_data)
        else:
            super().__init__(**data)
        if self.cluster_owner_namespace is None:
            self.cluster_owner_namespace = get_namespace_name(
                self.cluster_owner_principal_identifier
            )
        self.image_pull_secrets = self._build_image_pull_secrets()
        self.workers = self._build_workers()

    def _build_image_pull_secrets(self) -> List[ImagePullSecretRenderSpec]:
        if self.k8s_options is None or not self.k8s_options.image_credentials:
            return []
        specs: List[ImagePullSecretRenderSpec] = []
        for i, cred in enumerate(self.k8s_options.image_credentials):
            # Credentials without both username and password become a
            # placeholder Secret with an empty auths map — same shape that
            # the gpustack Helm chart's canonical pull-secret uses when no
            # credentials are configured. The Secret is still referenced
            # from imagePullSecrets so users can later patch it in-cluster
            # without re-applying the manifest.
            if cred.username and cred.password:
                auth = base64.b64encode(
                    f"{cred.username}:{cred.password}".encode("utf-8")
                ).decode("utf-8")
                dockerconfigjson = json.dumps(
                    {
                        "auths": {
                            cred.registry: {
                                "username": cred.username,
                                "password": cred.password,
                                "auth": auth,
                            }
                        }
                    }
                )
            else:
                dockerconfigjson = json.dumps({"auths": {}})
            specs.append(
                ImagePullSecretRenderSpec(
                    name=f"{IMAGE_PULL_SECRET_NAME_PREFIX}-{i}",
                    registry=cred.registry,
                    dockerconfigjson_b64=base64.b64encode(
                        dockerconfigjson.encode("utf-8")
                    ).decode("utf-8"),
                )
            )
        return specs

    def _build_workers(self) -> List[WorkerRenderSpec]:
        seen: set = set()
        gpu_runtimes: List[ManufacturerEnum] = []
        for r in self.runtimes or []:
            # UNKNOWN is the "no GPU detected" sentinel — never gets its own
            # GPU worker DaemonSet.
            if r == ManufacturerEnum.UNKNOWN:
                continue
            if r in seen:
                continue
            seen.add(r)
            gpu_runtimes.append(r)

        # Order the GPU DaemonSets by a fixed canonical runtime order rather
        # than the request order. The canonical order makes the rendered output
        # deterministic for a given set of runtimes regardless of the order they
        # were requested in.
        gpu_runtimes.sort(key=lambda r: _RUNTIME_ORDER.get(r, len(_RUNTIME_ORDER)))

        base_node_selector = (
            self.k8s_options.node_selector if self.k8s_options is not None else None
        )

        workers: List[WorkerRenderSpec] = []

        # Always render the CPU DaemonSet first. It owns the legacy
        # ``gpustack-worker`` name.
        workers.append(
            WorkerRenderSpec(
                name="cpu",
                ds_name=WORKER_DS_BASENAME,
                runtime="",
                node_selector=_cpu_node_selector_for(base_node_selector),
            )
        )

        # All GPU vendor DaemonSets get a runtime-suffixed name.
        for runtime in gpu_runtimes:
            workers.append(
                WorkerRenderSpec(
                    name=runtime.value,
                    ds_name=f"{WORKER_DS_BASENAME}-{runtime.value}",
                    runtime=runtime.value,
                    node_selector=_node_selector_for(runtime, base_node_selector),
                )
            )

        return workers


def _node_selector_for(
    runtime: ManufacturerEnum, base: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """
    Merge the cluster-level base ``nodeSelector`` with the runtime's PCI
    presence label (``feature.node.kubernetes.io/pci-<id>.present: "true"``)
    so the DaemonSet only schedules onto nodes carrying that vendor's device.
    Base keys lose to the PCI label on collision (the PCI label is the
    vendor-scoping invariant). Returns None when nothing applies.
    """
    selector: Dict[str, str] = dict(base or {})
    pci_id = _MANUFACTURER_PCI_ID.get(runtime)
    if pci_id:
        selector[_PCI_NODE_LABEL.format(pci_id=pci_id)] = "true"
    return selector or None


def _cpu_node_selector_for(
    base: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """
    Merge the cluster-level base ``nodeSelector`` with the CPU node label
    (``feature.gpustack.ai/acceleratable: "false"``) so the DaemonSet only
    schedules onto non-acceleratable (CPU-only) nodes. No NFD PCI labels are
    added. Base keys lose to the CPU label on collision.
    """
    selector: Dict[str, str] = dict(base or {})
    selector[_CPU_NODE_LABEL] = "false"
    return selector
