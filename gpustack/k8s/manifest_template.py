import base64
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, computed_field

from gpustack import __operator_version__
from gpustack.gpu_instances.cluster_apis_util import get_namespace_name
from gpustack.schemas.clusters import (
    ClusterRegistrationTokenPublic,
    K8sOptions,
    K8sVolumeMount,
    ensure_data_dir_mount,
)
from gpustack_runtime.detector import ManufacturerEnum


_DEFAULT_OPERATOR_IMAGE = f"gpustack/gpustack-operator:{__operator_version__}"
_DEFAULT_CONTAINER_NAMESPACE = "gpustack"
_DEFAULT_CLUSTER_NAMESPACE = "gpustack-system"


IMAGE_PULL_SECRET_NAME_PREFIX = "gpustack-image-pull-secret"


def _env_bool(value: Optional[bool]) -> Optional[str]:
    """
    Render one tri-state ``GpuInstanceOptions`` knob as the string its
    ``GPUSTACK_*`` environment variable carries.

    A *string*, not a bool, for two reasons. A Kubernetes env value is a string,
    so the template quotes it and YAML never turns it back into a boolean. And
    the template has to tell "unmanaged" apart from an explicit off: ``None``
    stays ``None`` so no entry is rendered at all and the operator's own default
    applies, while ``False`` must render the literal ``"false"``. A jinja
    truthiness test on a raw bool would collapse those two into the same
    (wrong) answer, which is why the template tests ``is not none`` instead.
    """
    if value is None:
        return None
    return "true" if value else "false"


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
    # Worker listen/metrics ports, sourced from ``cluster.worker_config``
    # (``worker_port``/``worker_metrics_port``). Threaded into the worker
    # container env (so the worker binds these ports), the DaemonSet
    # containerPorts, and the Service ports + prometheus scrape annotation.
    # Fall back to the built-in defaults when the cluster doesn't override them.
    worker_port: int = 10150
    worker_metrics_port: int = 10151
    # Pre-computed Secret render data, one per K8sOptions.image_credentials
    # entry. Both image_pull_secrets.jinja (Secret resource) and the
    # daemonset.jinja imagePullSecrets reference iterate this list, so the
    # Secret name is the single source of truth.
    image_pull_secrets: List[ImagePullSecretRenderSpec] = []

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
    def volume_mounts(self) -> List[K8sVolumeMount]:
        """
        Pod volumes / container volumeMounts applied to every worker DaemonSet,
        with the reserved gpustack data-dir mount guaranteed present and first.

        The templates read this instead of ``k8s_options.volume_mounts`` so the
        data dir can never be rendered away: a cluster created before the mount
        became configurable (v2.2.0) holds no ``volumeMounts`` at all, and the
        DaemonSet would otherwise come out with no persistent
        ``/var/lib/gpustack`` — worker data lost on every pod restart, silently.
        Only injects; the persisted shape is normalized by the routes layer.
        """
        k8s_options = self.k8s_options or K8sOptions()
        return ensure_data_dir_mount(k8s_options.volume_mounts)

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
    def operator_instance_type_derived_from_node(self) -> Optional[str]:
        """
        First-deploy seed for the operator's ``instance-type-derived-from-node``
        setting. The operator seeds each ``Setting`` from ``GPUSTACK_<NAME>``
        on first deploy and never overwrites a stored value afterwards, so this
        is what keeps a cluster from deriving instance types the administrator
        asked it not to derive in the window before anything else can reach it.

        ``None`` when the knob is unmanaged (or the cluster is not a GPU Service
        cluster at all) — the template then renders no entry and the operator's
        own default stands. See :func:`_env_bool`.
        """
        options = self.k8s_options.gpu_instance_options if self.k8s_options else None
        return _env_bool(
            options.gpu_instance_type_derived_from_node if options else None
        )

    @computed_field
    @property
    def operator_instance_type_mixed_on_node(self) -> Optional[str]:
        """
        First-deploy seed for the operator's ``instance-type-mixed-on-node``
        setting, on the same terms as
        :attr:`operator_instance_type_derived_from_node`.
        """
        options = self.k8s_options.gpu_instance_options if self.k8s_options else None
        return _env_bool(options.gpu_instance_type_mixed_on_node if options else None)

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
