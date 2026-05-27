import jinja2
import base64
import json
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict

from gpustack.gpu_instances.cluster_apis_util import get_namespace_name
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.clusters import ClusterRegistrationTokenPublic, K8sOptions
from gpustack_runtime.detector import ManufacturerEnum


CPU_WORKER_NAME = "cpu"
WORKER_DS_BASENAME = "gpustack-worker"
IMAGE_PULL_SECRET_NAME_PREFIX = "gpustack-image-pull-secret"


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
    Per-DaemonSet render data, one entry per worker variant. The CPU worker
    is always emitted and keeps the legacy ``gpustack-worker`` name so
    existing clusters can in-place update without a DaemonSet selector
    mutation. Each requested GPU vendor adds a ``gpustack-worker-<vendor>``
    worker alongside it.

    The grouping mirrors how Helm chart values are typically organized
    (``.Values.worker.*``), so a future chart migration maps 1:1 onto these
    entries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    ds_name: str
    # Runtime tag consumed by per-vendor jinja branches (volume mounts /
    # volumes / runtimeClassName / vendor-specific env). Empty string for
    # the CPU worker so none of those branches activate.
    runtime: str = ""
    is_cpu: bool = False
    node_selector: Optional[Dict[str, str]] = None
    # Vendor node labels the CPU worker must avoid (DoesNotExist
    # matchExpression). Populated only for the CPU worker; derived from
    # all configured ``gpu_vendor_overrides[*].node_selector`` keys.
    cpu_exclusion_keys: List[str] = []


class TemplateConfig(ClusterRegistrationTokenPublic):
    # cluster owner namespace, defaults to "gpustack-{cluster_owner_principal_name}",
    # used to placing the Kubernetes resources for the cluster owner.
    cluster_owner_namespace: Optional[str] = None
    # cluster-specific namespace, defaults to "gpustack-system".
    namespace: Optional[str] = None
    cluster_owner_principal_name: Optional[str] = None
    runtimes: Optional[List[ManufacturerEnum]] = None
    k8s_options: Optional[K8sOptions] = None
    workers: List[WorkerRenderSpec] = []
    # Pre-computed Secret render data, one per K8sOptions.image_credentials
    # entry. Both image_pull_secrets.jinja (Secret resource) and the
    # daemonset.jinja imagePullSecrets reference iterate this list, so the
    # Secret name is the single source of truth.
    image_pull_secrets: List[ImagePullSecretRenderSpec] = []
    # True when 2+ distinct GPU runtimes are requested. Switches the jinja
    # templates from the v1-compatible single-DaemonSet output to the
    # multi-DaemonSet output (per-vendor DS + always-on CPU DS + safety nets
    # like podAntiAffinity and CPU nodeAffinity DoesNotExist).
    multi_vendor_mode: bool = False

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
        if self.namespace is None:
            self.namespace = "gpustack-system"
        if self.cluster_owner_namespace is None:
            self.cluster_owner_namespace = get_namespace_name(
                self.cluster_owner_principal_name
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
        gpu_runtimes: List[ManufacturerEnum] = []
        seen: set = set()
        for r in self.runtimes or []:
            # UNKNOWN is the "no GPU detected" sentinel — covered by the
            # legacy worker, so don't emit a separate vendor worker for it.
            if r == ManufacturerEnum.UNKNOWN:
                continue
            if r in seen:
                continue
            seen.add(r)
            gpu_runtimes.append(r)

        self.multi_vendor_mode = len(gpu_runtimes) >= 2

        if not self.multi_vendor_mode:
            # 0 or 1 GPU runtime → emit ONE DaemonSet under the legacy name
            # ``gpustack-worker`` with that runtime's vendor blocks. Output
            # is byte-compatible with v1 single-vendor manifests (no
            # additional labels, no podAntiAffinity, no nodeAffinity). The
            # CPU worker concept does not exist in this mode.
            runtime = gpu_runtimes[0] if gpu_runtimes else None
            resolved = (
                self.k8s_options.resolve_for(runtime)
                if self.k8s_options is not None
                else None
            )
            return [
                WorkerRenderSpec(
                    name=runtime.value if runtime else CPU_WORKER_NAME,
                    ds_name=WORKER_DS_BASENAME,
                    runtime=runtime.value if runtime else "",
                    is_cpu=runtime is None,
                    node_selector=(resolved.node_selector if resolved else None),
                )
            ]

        # 2+ GPU runtimes → multi-vendor mode: a per-vendor DS plus the
        # always-present CPU DS, with the full safety-net set.
        workers: List[WorkerRenderSpec] = []
        for runtime in gpu_runtimes:
            resolved = (
                self.k8s_options.resolve_for(runtime)
                if self.k8s_options is not None
                else None
            )
            workers.append(
                WorkerRenderSpec(
                    name=runtime.value,
                    ds_name=f"{WORKER_DS_BASENAME}-{runtime.value}",
                    runtime=runtime.value,
                    is_cpu=False,
                    node_selector=(resolved.node_selector if resolved else None),
                )
            )

        cpu_resolved = (
            self.k8s_options.resolve_for(None) if self.k8s_options is not None else None
        )
        workers.append(
            WorkerRenderSpec(
                name=CPU_WORKER_NAME,
                ds_name=WORKER_DS_BASENAME,
                runtime="",
                is_cpu=True,
                node_selector=(cpu_resolved.node_selector if cpu_resolved else None),
                cpu_exclusion_keys=_collect_vendor_label_keys(self.k8s_options),
            )
        )
        return workers


def _collect_vendor_label_keys(k8s_options: Optional[K8sOptions]) -> List[str]:
    """
    Union of all label keys declared under any ``gpu_vendor_overrides[*].node_selector``.
    The CPU DS uses these as DoesNotExist matchExpressions so it avoids any
    node the user has marked as belonging to a specific vendor.
    """
    if k8s_options is None or k8s_options.gpu_vendor_overrides is None:
        return []
    keys: set = set()
    for override in k8s_options.gpu_vendor_overrides.values():
        if override.node_selector:
            keys.update(override.node_selector.keys())
    return sorted(keys)
