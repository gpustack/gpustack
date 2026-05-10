from __future__ import annotations

import asyncio
import http
import logging
from typing import Optional

from kubernetes_asyncio import client
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas import GPUInstanceSSHPublicKey

from .__util__ import get_k8s_client
from .namespace import ensure_namespace_in_cluster, get_namespace_name

SSH_PUBLIC_KEY_NAME = "gpustack-ssh-public-key"

logger = logging.getLogger(__name__)


async def get_ssh_public_key(
    session: AsyncSession,
    owner_principal_id: int,
) -> str | None:
    """
    Get the instance ssh public key for the organization.
    """

    ret: GPUInstanceSSHPublicKey = await GPUInstanceSSHPublicKey.one_by_fields(
        session=session,
        fields={
            "owner_principal_id": owner_principal_id,
            "name": SSH_PUBLIC_KEY_NAME,
        },
    )
    if ret is None:
        return None

    return ret.spec.data


async def sync_ssh_public_key_to_cluster(
    ssh_public_key: str,
    server_api_port: int,
    cluster_id: int,
    cluster_owner_principal_slug: Optional[str] = None,
):
    """
    Sync the instance ssh public key to a single cluster by creating or patching a custom resource in the cluster.
    """

    group = "worker.gpustack.ai"
    version = "v1"
    plural = "instancesshpublickeys"
    api_version = f"{group}/{version}"
    kind = "InstanceSSHPublicKey"

    namespace = get_namespace_name(cluster_owner_principal_slug)

    async with get_k8s_client(server_api_port, cluster_id) as api:
        # Ensure the namespace exists before creating the custom object.
        await ensure_namespace_in_cluster(
            name=namespace,
            api=api,
        )

        crd = client.CustomObjectsApi(api)

        # Try to patch the resource first, in case it already exist.
        try:
            await crd.patch_namespaced_custom_object(
                group=group,
                version=version,
                plural=plural,
                namespace=namespace,
                name=SSH_PUBLIC_KEY_NAME,
                body={
                    "spec": {
                        "data": ssh_public_key,
                    },
                },
                _content_type="application/merge-patch+json",
            )
            logger.info(f"Patched instance ssh public key in cluster {cluster_id}")
            return
        except client.exceptions.ApiException as e:
            if e.status != http.HTTPStatus.NOT_FOUND:
                logger.exception(
                    f"Failed to patch instance ssh public key in cluster {cluster_id}"
                )
                raise e

        # If the resource doesn't exist, create it.
        try:
            await crd.create_namespaced_custom_object(
                group=group,
                version=version,
                plural=plural,
                namespace=namespace,
                body={
                    "apiVersion": api_version,
                    "kind": kind,
                    "metadata": {
                        "name": SSH_PUBLIC_KEY_NAME,
                    },
                    "spec": {
                        "data": ssh_public_key,
                    },
                },
            )
            logger.info(f"Created instance ssh public key in cluster {cluster_id}")
        except client.exceptions.ApiException as e:
            if e.status != http.HTTPStatus.CONFLICT:
                logger.exception(
                    f"Failed to create instance ssh public key in cluster {cluster_id}"
                )
                raise e


async def sync_ssh_public_key_to_clusters(
    ssh_public_key: str,
    server_api_port: int,
    cluster_ids: int | list[int],
    cluster_owner_principal_slug: Optional[str] = None,
):
    """
    Sync the instance ssh public key to multiple clusters in parallel.
    """

    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]

    tasks = [
        sync_ssh_public_key_to_cluster(
            ssh_public_key=ssh_public_key,
            server_api_port=server_api_port,
            cluster_id=cluster_id,
            cluster_owner_principal_slug=cluster_owner_principal_slug,
        )
        for cluster_id in cluster_ids
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )
    for cluster_id, result in zip(cluster_ids, results):
        if isinstance(result, Exception):
            logger.error(
                f"Failed to sync instance ssh public key to cluster {cluster_id}",
                exc_info=result,
            )
