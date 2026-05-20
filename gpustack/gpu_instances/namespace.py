from __future__ import annotations

import http
import logging
from typing import Optional

from kubernetes_asyncio import client
from kubernetes_asyncio.client.api_client import ApiClient

from .__util__ import get_k8s_client
from ..schemas.principals import PLATFORM_PRINCIPAL_NAME

NAMESPACE_PREFIX = "gpustack"

logger = logging.getLogger(__name__)


def get_namespace_name(
    principal_name: Optional[str] = None,
) -> str:
    """
    Get the namespace name for the given principal ``name``.
    """

    if principal_name is None:
        principal_name = PLATFORM_PRINCIPAL_NAME

    return f"{NAMESPACE_PREFIX}-{principal_name}"


async def sync_namespace_to_cluster(
    server_api_port: int,
    cluster_id: int,
    cluster_owner_principal_name: Optional[str] = None,
):
    """
    Sync the namespace to a single cluster by creating the namespace in the cluster if it does not exist.
    """

    name = get_namespace_name(cluster_owner_principal_name)

    async with get_k8s_client(server_api_port, cluster_id) as api:
        await ensure_namespace_in_cluster(
            name=name,
            api=api,
        )


async def ensure_namespace_in_cluster(
    name: str,
    api: ApiClient,
):
    """
    Ensure the namespace exists in the cluster by creating it if it does not exist.
    If the namespace already exists, do nothing.
    """

    core = client.CoreV1Api(api)

    # Check if the namespace already exists
    try:
        await core.read_namespace(name=name)
        return
    except client.exceptions.ApiException as e:
        if e.status != http.HTTPStatus.NOT_FOUND:
            logger.exception(f"Failed to read namespace {name}")
            raise e

    # Check if the namespace already exists
    try:
        await core.create_namespace(
            body=client.V1Namespace(metadata=client.V1ObjectMeta(name=name)),
        )
    except client.exceptions.ApiException as e:
        if e.status != http.HTTPStatus.CONFLICT:
            logger.exception(f"Failed to create namespace {name}")
            raise e
