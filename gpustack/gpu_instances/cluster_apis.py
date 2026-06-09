from __future__ import annotations

import http
import logging
from dataclasses import dataclass
from typing import Optional

from kubernetes_asyncio import client

from .cluster_apis_util import (
    get_namespace_name,
    get_k8s_client,
)

logger = logging.getLogger(__name__)

_GROUP = "worker.gpustack.ai"
_VERSION = "v1"


@dataclass(frozen=True)
class _CRDSpec:
    plural: str
    kind: str
    namespaced: bool

    @property
    def api_version(self) -> str:
        return f"{_GROUP}/{_VERSION}"


_SSH_PUBLIC_KEY = _CRDSpec(
    plural="instancesshpublickeys",
    kind="InstanceSSHPublicKey",
    namespaced=True,
)
_PV_TYPE = _CRDSpec(
    plural="instancepersistentvolumetypes",
    kind="InstancePersistentVolumeType",
    namespaced=False,
)
_PV = _CRDSpec(
    plural="instancepersistentvolumes",
    kind="InstancePersistentVolume",
    namespaced=True,
)
_INSTANCE_TYPE = _CRDSpec(
    plural="instancetypes",
    kind="InstanceType",
    namespaced=False,
)
_INSTANCE = _CRDSpec(
    plural="instances",
    kind="Instance",
    namespaced=True,
)


class ClusterOps:
    """Raw CRD client for ``worker.gpustack.ai/v1`` resources.

    Owns a :class:`kubernetes_asyncio.client.api_client.ApiClient` which must
    be closed. Use as an async context manager so the client is released on
    exit:

        async with ClusterOps(...) as ops:
            await ops.create_instance(...)

    The :func:`cluster_ops` factory is a thin alias kept for callers that
    prefer the explicit context-manager call style.
    """

    cluster_id: int
    cluster_owner_principal_identifier: str
    api_client: client.api_client.ApiClient
    org_namespace: str

    def __init__(
        self,
        server_api_port: int,
        cluster_id: int,
        cluster_registration_token: str,
        cluster_owner_principal_identifier: str,
    ):
        self.cluster_id = cluster_id
        self.cluster_owner_principal_identifier = cluster_owner_principal_identifier
        self.api_client = get_k8s_client(
            server_api_port=server_api_port,
            cluster_id=cluster_id,
            cluster_registration_token=cluster_registration_token,
        )
        self.org_namespace = get_namespace_name(
            principal_identifier=cluster_owner_principal_identifier,
        )

    async def __aenter__(self) -> "ClusterOps":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.api_client.close()

    #
    # Generic CRD helpers
    #

    def _crd(self) -> client.CustomObjectsApi:
        return client.CustomObjectsApi(self.api_client)

    async def _read(self, spec: _CRDSpec, name: str) -> Optional[dict]:
        crd = self._crd()
        try:
            if spec.namespaced:
                return await crd.get_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    name=name,
                )
            return await crd.get_cluster_custom_object(
                group=_GROUP,
                version=_VERSION,
                plural=spec.plural,
                name=name,
            )
        except client.exceptions.ApiException as e:
            if e.status == http.HTTPStatus.NOT_FOUND:
                return None
            raise

    async def _create(
        self,
        spec: _CRDSpec,
        name: str,
        body_spec: dict,
        ignore_existed: bool,
    ) -> dict:
        """Create a CRD object and return the server-acknowledged dict.

        When ``ignore_existed`` is true and the object already exists (either
        observed up-front or raced against another writer), the current
        server state is read back and returned, so callers always see a
        consistent post-condition.
        """
        if spec.namespaced:
            await self.ensure_org_namespace()

        if ignore_existed:
            existing = await self._read(spec, name)
            if existing is not None:
                return existing

        body = {
            "apiVersion": spec.api_version,
            "kind": spec.kind,
            "metadata": {"name": name},
            "spec": body_spec,
        }
        if spec.namespaced:
            body["metadata"]["namespace"] = self.org_namespace

        crd = self._crd()
        try:
            if spec.namespaced:
                created = await crd.create_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    body=body,
                )
            else:
                created = await crd.create_cluster_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    body=body,
                )
            logger.info(
                "Created %s %s in cluster %s",
                spec.kind,
                name,
                self.cluster_id,
            )
            return created
        except client.exceptions.ApiException as e:
            if ignore_existed and e.status == http.HTTPStatus.CONFLICT:
                existing = await self._read(spec, name)
                if existing is not None:
                    return existing
            raise

    async def _upsert(
        self,
        spec: _CRDSpec,
        name: str,
        body_spec: dict,
    ) -> dict:
        """Patch-then-create-on-404 for a CRD object.

        Returns the server-acknowledged dict (patched or freshly created).

        Race semantics: on a PATCH(404) → CREATE(409) sequence the create's
        409 is swallowed and the current server state is read back, so the
        returned object reflects the concurrent writer's spec. Callers that
        require last-writer-wins must retry on their side.
        """
        if spec.namespaced:
            await self.ensure_org_namespace()

        crd = self._crd()
        patch_body = {"spec": body_spec}

        try:
            if spec.namespaced:
                patched = await crd.patch_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    name=name,
                    body=patch_body,
                    _content_type="application/merge-patch+json",
                )
            else:
                patched = await crd.patch_cluster_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    name=name,
                    body=patch_body,
                    _content_type="application/merge-patch+json",
                )
            logger.info(
                "Patched %s %s in cluster %s",
                spec.kind,
                name,
                self.cluster_id,
            )
            return patched
        except client.exceptions.ApiException as e:
            if e.status != http.HTTPStatus.NOT_FOUND:
                raise

        create_body = {
            "apiVersion": spec.api_version,
            "kind": spec.kind,
            "metadata": {"name": name},
            "spec": body_spec,
        }
        if spec.namespaced:
            create_body["metadata"]["namespace"] = self.org_namespace

        try:
            if spec.namespaced:
                created = await crd.create_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    body=create_body,
                )
            else:
                created = await crd.create_cluster_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    body=create_body,
                )
            logger.info(
                "Created %s %s in cluster %s",
                spec.kind,
                name,
                self.cluster_id,
            )
            return created
        except client.exceptions.ApiException as e:
            if e.status == http.HTTPStatus.CONFLICT:
                existing = await self._read(spec, name)
                if existing is not None:
                    return existing
            raise

    async def _patch_spec(
        self, spec: _CRDSpec, name: str, body_spec: dict
    ) -> Optional[dict]:
        """Merge-patch the spec of a CRD object. Return None when absent.

        Keys set to ``None`` in ``body_spec`` are removed from the live
        spec by merge-patch semantics.
        """
        crd = self._crd()
        body = {"spec": body_spec}
        try:
            if spec.namespaced:
                patched = await crd.patch_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    name=name,
                    body=body,
                    _content_type="application/merge-patch+json",
                )
            else:
                patched = await crd.patch_cluster_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    name=name,
                    body=body,
                    _content_type="application/merge-patch+json",
                )
            logger.info(
                "Patched %s %s spec in cluster %s",
                spec.kind,
                name,
                self.cluster_id,
            )
            return patched
        except client.exceptions.ApiException as e:
            if e.status == http.HTTPStatus.NOT_FOUND:
                return None
            raise

    async def _delete(self, spec: _CRDSpec, name: str) -> None:
        crd = self._crd()
        try:
            if spec.namespaced:
                await crd.delete_namespaced_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    namespace=self.org_namespace,
                    name=name,
                )
            else:
                await crd.delete_cluster_custom_object(
                    group=_GROUP,
                    version=_VERSION,
                    plural=spec.plural,
                    name=name,
                )
            logger.info(
                "Deleted %s %s in cluster %s",
                spec.kind,
                name,
                self.cluster_id,
            )
        except client.exceptions.ApiException as e:
            if e.status == http.HTTPStatus.NOT_FOUND:
                return
            raise

    #
    # Namespace Operations
    #

    async def create_namespace(self, name: str, ignore_existed: bool = True):
        """
        Create the namespace in the cluster if it does not exist.
        If the namespace already exists, do nothing.
        """
        core = client.CoreV1Api(self.api_client)

        if ignore_existed:
            try:
                await core.read_namespace(name=name)
                return
            except client.exceptions.ApiException as e:
                if e.status != http.HTTPStatus.NOT_FOUND:
                    raise

        try:
            await core.create_namespace(
                body=client.V1Namespace(metadata=client.V1ObjectMeta(name=name)),
            )
            logger.info("Created namespace %s in cluster %s", name, self.cluster_id)
        except client.exceptions.ApiException as e:
            if ignore_existed and e.status == http.HTTPStatus.CONFLICT:
                return
            raise

    async def delete_namespace(self, name: str):
        """
        Delete the namespace in the cluster if it exists.
        If the namespace does not exist, do nothing.
        """
        core = client.CoreV1Api(self.api_client)
        try:
            await core.delete_namespace(name=name)
            logger.info("Deleted namespace %s in cluster %s", name, self.cluster_id)
        except client.exceptions.ApiException as e:
            if e.status == http.HTTPStatus.NOT_FOUND:
                return
            raise

    async def ensure_org_namespace(self):
        """
        Ensure the organization namespace exists in the cluster by creating it if it does not exist.
        """
        await self.create_namespace(self.org_namespace)

    #
    # SSH Public Key Operations
    #

    async def read_ssh_public_key(self, name: str) -> Optional[dict]:
        """
        Read the instance ssh public key in the cluster by name.
        If the instance ssh public key does not exist, return None.
        """
        return await self._read(_SSH_PUBLIC_KEY, name)

    async def upsert_ssh_public_key(self, name: str, spec: dict) -> dict:
        """
        Upsert the instance ssh public key in the cluster by patching it if it exists,
        or creating it if it does not exist.

        Returns the server-acknowledged object.
        """
        return await self._upsert(_SSH_PUBLIC_KEY, name, spec)

    async def delete_ssh_public_key(self, name: str):
        """
        Delete the instance ssh public key in the cluster if it exists.
        If the instance ssh public key does not exist, do nothing.
        """
        await self._delete(_SSH_PUBLIC_KEY, name)

    #
    # Persistent Volume Type Operations
    #

    async def read_persistent_volume_type(self, name: str) -> Optional[dict]:
        """
        Read the persistent volume type in the cluster by name.
        If the persistent volume type does not exist, return None.
        """
        return await self._read(_PV_TYPE, name)

    async def create_persistent_volume_type(
        self, name: str, spec: dict, ignore_existed: bool = True
    ) -> dict:
        """
        Create the persistent volume type in the cluster.

        Returns the created object, or the existing one when
        ``ignore_existed`` is true and the resource already exists.
        """
        return await self._create(_PV_TYPE, name, spec, ignore_existed)

    async def delete_persistent_volume_type(self, name: str):
        """
        Delete the persistent volume type in the cluster if it exists.
        If the persistent volume type does not exist, do nothing.
        """
        await self._delete(_PV_TYPE, name)

    #
    # Persistent Volume Operations
    #

    async def read_persistent_volume(self, name: str) -> Optional[dict]:
        """
        Read the persistent volume in the cluster by name.
        If the persistent volume does not exist, return None.
        """
        return await self._read(_PV, name)

    async def create_persistent_volume(
        self, name: str, spec: dict, ignore_existed: bool = True
    ) -> dict:
        """
        Create the persistent volume in the cluster.

        Returns the created object, or the existing one when
        ``ignore_existed`` is true and the resource already exists.
        """
        return await self._create(_PV, name, spec, ignore_existed)

    async def delete_persistent_volume(self, name: str):
        """
        Delete the persistent volume in the cluster if it exists.
        If the persistent volume does not exist, do nothing.
        """
        await self._delete(_PV, name)

    #
    # Instance Types
    #

    async def read_instance_type(self, name: str) -> Optional[dict]:
        """
        Read the instance type in the cluster by name.
        If the instance type does not exist, return None.
        """
        return await self._read(_INSTANCE_TYPE, name)

    #
    # Instance Operations
    #

    async def read_instance(self, name: str) -> Optional[dict]:
        """
        Read the instance in the cluster by name.
        If the instance does not exist, return None.
        """
        return await self._read(_INSTANCE, name)

    async def create_instance(
        self, name: str, spec: dict, ignore_existed: bool = True
    ) -> dict:
        """
        Create the instance in the cluster.

        Returns the created object, or the existing one when
        ``ignore_existed`` is true and the resource already exists.
        """
        return await self._create(_INSTANCE, name, spec, ignore_existed)

    async def stop_instance(self, name: str) -> Optional[dict]:
        """Stop the instance by patching ``spec.stop=true``.

        Returns the server-acknowledged object, or ``None`` if the
        instance is gone.
        """
        return await self._patch_spec(_INSTANCE, name, {"stop": True})

    async def start_instance(self, name: str) -> Optional[dict]:
        """Resume the instance by removing ``spec.stop`` via merge-patch null.

        Returns the server-acknowledged object, or ``None`` if the
        instance is gone.
        """
        return await self._patch_spec(_INSTANCE, name, {"stop": None})

    async def delete_instance(self, name: str):
        """
        Delete the instance in the cluster if it exists.
        If the instance does not exist, do nothing.
        """
        await self._delete(_INSTANCE, name)
