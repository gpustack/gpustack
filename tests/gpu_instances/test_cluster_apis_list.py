"""E1: ClusterOps gains list capability for the PV/PVT finalizer controllers.

``list_persistent_volume_types`` is cluster-scoped and ``list_persistent_volumes``
lists across all namespaces — both hit ``list_cluster_custom_object`` (the
all-namespaces endpoint for a namespaced CRD). A missing CRD yields ``[]``.
"""

import http
from unittest.mock import AsyncMock, MagicMock

import pytest

from kubernetes_asyncio import client

from gpustack.gpu_instances.cluster_apis import ClusterOps


def _ops(list_return=None, api_exc=None):
    ops = object.__new__(ClusterOps)  # bypass __init__ (no real k8s client)
    crd = MagicMock()
    if api_exc is not None:
        crd.list_cluster_custom_object = AsyncMock(side_effect=api_exc)
    else:
        crd.list_cluster_custom_object = AsyncMock(return_value=list_return)
    ops._crd = lambda: crd
    ops._list_crd = crd  # exposed for assertions
    return ops


@pytest.mark.asyncio
async def test_list_persistent_volumes_returns_items():
    ops = _ops(
        list_return={
            "items": [{"metadata": {"name": "pv-1"}}, {"metadata": {"name": "pv-2"}}]
        }
    )

    items = await ops.list_persistent_volumes()

    assert [i["metadata"]["name"] for i in items] == ["pv-1", "pv-2"]
    _, kwargs = ops._list_crd.list_cluster_custom_object.call_args
    assert kwargs["plural"] == "instancepersistentvolumes"


@pytest.mark.asyncio
async def test_list_persistent_volume_types_returns_items():
    ops = _ops(list_return={"items": [{"metadata": {"name": "pvt-1"}}]})

    items = await ops.list_persistent_volume_types()

    assert [i["metadata"]["name"] for i in items] == ["pvt-1"]
    _, kwargs = ops._list_crd.list_cluster_custom_object.call_args
    assert kwargs["plural"] == "instancepersistentvolumetypes"


@pytest.mark.asyncio
async def test_list_missing_crd_returns_empty():
    exc = client.exceptions.ApiException(status=http.HTTPStatus.NOT_FOUND)
    ops = _ops(api_exc=exc)

    assert await ops.list_persistent_volumes() == []


@pytest.mark.asyncio
async def test_list_empty_result_returns_empty():
    ops = _ops(list_return={"items": []})
    assert await ops.list_persistent_volume_types() == []
