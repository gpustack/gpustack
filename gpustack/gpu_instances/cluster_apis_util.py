from typing import Optional, Tuple

from kubernetes_asyncio import client

from gpustack.schemas.principals import PLATFORM_PRINCIPAL_NAME

from gpustack.schemas import (
    GPUInstance,
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeType,
)


def _hoist_meta(spec: dict, owner) -> dict:
    """Hoist row-level ``display_name`` / ``description`` into ``spec``.

    The Python schemas keep these on the SQLModel row (mirroring the
    table columns), but the Go CRDs declare them on the resource's
    ``spec``. Each CRD-bound dict_* function calls this so the on-wire
    payload matches the Go shape.
    """
    if getattr(owner, "display_name") is not None:
        spec["displayName"] = owner.display_name
    if getattr(owner, "description") is not None:
        spec["description"] = owner.description
    return spec


def spec_instance(instance: GPUInstance) -> dict:
    """Convert a :class:`GPUInstance` row into the dict shape expected by
    the ``worker.gpustack.ai/v1`` Instance CRD's ``spec`` field, suitable
    for handing to :meth:`ClusterOps.create_instance`.

    Field names follow the Go CRD's camelCase convention via the
    pydantic ``alias_generator`` already configured on
    :class:`GPUInstanceSpec`, so ``model_dump(by_alias=True)`` produces
    the on-wire keys directly — no manual snake→camel mapping is needed.

    Transforms applied:

    1. ``displayName`` and ``description``, which live on the row in
       Python but on ``InstanceSpec`` in Go, are hoisted into the spec
       dict so the CRD sees them in the expected place.
    2. ``volume.persistentTemplate`` collapses into ``volume.persistent``.
       Only the ``name`` survives; ``spec`` and ``releaseWithInstance``
       drive server-side provisioning and are not part of the CRD.
    3. ``sshPublicKeys`` (list) collapses into ``sshPublicKey`` (singular
       ``LocalObjectReference``) to match the Go field.
       The name of the referenced SSH key is the same as the instance name.
    """
    spec = instance.spec.model_dump(by_alias=True, exclude_none=True)
    _hoist_meta(spec, instance)

    volume = spec.get("volume")
    if volume is not None:
        tmpl = volume.pop("persistentTemplate", None)
        if tmpl is not None:
            volume["persistent"] = {"name": tmpl["name"]}

    spec.pop("sshPublicKeys", None)
    spec["sshPublicKey"] = {"name": instance.name}

    return spec


def spec_persistent_volume(pv: GPUInstancePersistentVolume) -> dict:
    """Convert a :class:`GPUInstancePersistentVolume` row into the dict
    shape expected by the ``worker.gpustack.ai/v1`` InstancePersistentVolume
    CRD's ``spec`` field, suitable for handing to
    :meth:`ClusterOps.create_persistent_volume`.

    Only ``displayName`` / ``description`` need a hoist; the rest of the
    spec (``type``, ``capacity``) already lines up with the Go schema
    once camelCase aliases are applied.
    """
    spec = pv.spec.model_dump(by_alias=True, exclude_none=True)
    _hoist_meta(spec, pv)
    return spec


def spec_persistent_volume_type(
    pvt: GPUInstancePersistentVolumeType, principal_name: str
) -> dict:
    """Convert a :class:`GPUInstancePersistentVolumeType` row into the
    dict shape expected by the ``worker.gpustack.ai/v1``
    InstancePersistentVolumeType CRD's ``spec`` field, suitable for
    handing to :meth:`ClusterOps.create_persistent_volume_type`.

    ``displayName`` / ``description`` are hoisted from the row, and for
    NFS-based specs ``subDirectory`` is always suffixed with
    ``${pvc.metadata.namespace}/${pvc.metadata.name}`` so each PVC mounts
    an isolated path.
    """
    spec = pvt.spec.model_dump(by_alias=True, exclude_none=True)
    _hoist_meta(spec, pvt)
    _inject_nfs_sub_directory(spec, principal_name)
    _inject_s3_bucket_prefix(spec, principal_name)
    return spec


PREFIX = "gpustack"
SUFFIX = "${pvc.metadata.name}"


def _inject_nfs_sub_directory(spec: dict, principal_name: str) -> dict:
    """Ensure ``spec.nfs.subDirectory`` includes a per-PVC path suffix."""
    nfs = spec.get("nfs")
    if not isinstance(nfs, dict):
        return spec

    # Preserve user prefix while ensuring every PVC gets a unique leaf path.
    current = nfs.get("subDirectory")
    if current:
        nfs["subDirectory"] = f"{str(current).rstrip('/')}/{principal_name}/{SUFFIX}"
    else:
        nfs["subDirectory"] = f"{principal_name}/{SUFFIX}"

    return spec


def _inject_s3_bucket_prefix(spec: dict, principal_name: str) -> dict:
    """Ensure ``spec.s3.bucketPrefix`` includes a per-PVC path suffix."""
    s3 = spec.get("s3")
    if not isinstance(s3, dict):
        return spec

    # Preserve user prefix while ensuring every PVC gets a unique leaf path.
    current = s3.get("bucket")
    if current:
        s3["prefix"] = f"{principal_name}/{SUFFIX}"
    else:
        s3["bucket"] = principal_name
        s3["prefix"] = SUFFIX

    return spec


def get_k8s_client_config(
    server_api_port: int,
    cluster_id: int,
    cluster_registration_token: str,
) -> client.Configuration:
    api_config = client.Configuration(
        host=f"http://localhost:{server_api_port}/v2/clusters/{cluster_id}/proxy",
        api_key={
            "BearerToken": cluster_registration_token,
        },
        api_key_prefix={
            "BearerToken": "Bearer",
        },
    )
    api_config.verify_ssl = False
    return api_config


def get_k8s_client(
    server_api_port: int,
    cluster_id: int,
    cluster_registration_token: str,
) -> client.api_client.ApiClient:
    api_config = get_k8s_client_config(
        server_api_port, cluster_id, cluster_registration_token
    )
    api = client.api_client.ApiClient(configuration=api_config)
    api.user_agent = "gpustack/gpustack"
    return api


def get_namespace_name(
    principal_name: Optional[str] = None,
) -> str:
    """
    Get the Kubernetes Namespace name for the given principal name.
    """

    if principal_name is None:
        principal_name = PLATFORM_PRINCIPAL_NAME

    return f"{PREFIX}-{principal_name}"


def parse_namespace_name(
    namespace_name: str,
) -> Optional[str]:
    """
    Parse the principal name from the given Kubernetes Namespace name.

    Returns the principal name if the name is valid, or None if it is not.
    """

    if not namespace_name.startswith(f"{PREFIX}-"):
        return None

    parts = namespace_name.split("-")
    if len(parts) < 2:
        return None

    if parts[0] != PREFIX:
        return None

    # {principal_name}
    return "-".join(parts[1:])


def get_persistent_volume_type_name(
    name: str,
    principal_name: Optional[str] = None,
) -> str:
    """
    Get the GPUStack-Operator InstancePersistentVolumeType name for
    the given GPU instance persistent volume type name and  principal name.
    """

    if principal_name is None:
        principal_name = PLATFORM_PRINCIPAL_NAME

    return f"{PREFIX}-{principal_name}-{name}"


def parse_persistent_volume_type_name(
    persistent_volume_type_name: str,
) -> Optional[Tuple[str, str]]:
    """
    Parse the GPU instance persistent volume type name from the given
    GPUStack-Operator InstancePersistentVolumeType name.

    Returns a tuple of (name, principal_name) if the name is valid,
    or None if it is not.
    """

    if not persistent_volume_type_name.startswith(f"{PREFIX}-"):
        return None

    parts = persistent_volume_type_name.split("-")
    if len(parts) < 3:
        return None

    if parts[0] != PREFIX:
        return None

    # {name}, {principal_name}
    return "-".join(parts[2:]), parts[1]
