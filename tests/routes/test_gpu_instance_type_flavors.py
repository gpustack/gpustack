"""Per-cluster GPU instance-type-flavor route tests.

The handler reuses ``gpu_instances_helper.build_cluster_ops`` (which owns the
``ClusterOps`` reference), so ``ClusterOps`` is patched on that module.
"""

from types import SimpleNamespace

import pytest

from gpustack.api.exceptions import NotFoundException
from gpustack.routes import gpu_instance_type_flavors as flavor_routes
from gpustack.routes import gpu_instances_helper as helper
from gpustack.schemas.principals import PrincipalType

# SYSTEM principal → bypasses tenant filters (visible everywhere).
CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
    scoped_cluster_id=None,
)

REQUEST = SimpleNamespace(
    app=SimpleNamespace(
        state=SimpleNamespace(
            server_config=SimpleNamespace(get_api_port=lambda: 80),
        )
    )
)


def _patch_cluster(monkeypatch, cluster):
    async def fake_one_by_id(session, id=None, *args, **kwargs):
        return cluster

    monkeypatch.setattr(flavor_routes.Cluster, "one_by_id", fake_one_by_id)


def _patch_ops(monkeypatch, *, list_result=None):
    class FakeOps:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_instance_type_flavors(self, resource_version=None):
            return list_result

    # build_cluster_ops lives in the shared helper and resolves ClusterOps
    # from that module's globals.
    monkeypatch.setattr(helper, "ClusterOps", FakeOps)


def _cluster(id_=1, owner_principal_id=None):
    return SimpleNamespace(
        id=id_,
        owner_principal_id=owner_principal_id,
        registration_token="tok",
    )


@pytest.mark.asyncio
async def test_list_flavors_maps_metadata_name(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(
        monkeypatch,
        list_result={
            "items": [
                {
                    "metadata": {"name": "flavor-a"},
                    "spec": {"acceleratable": True, "manufacturer": "nvidia"},
                }
            ]
        },
    )

    out = await flavor_routes.get_gpu_instance_type_flavors(REQUEST, None, CTX, 1)

    assert len(out.items) == 1
    assert out.items[0].name == "flavor-a"
    assert out.items[0].spec.acceleratable is True
    assert out.items[0].spec.manufacturer == "nvidia"


@pytest.mark.asyncio
async def test_invisible_cluster_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, None)
    _patch_ops(monkeypatch, list_result={"items": []})

    with pytest.raises(NotFoundException) as exc:
        await flavor_routes.get_gpu_instance_type_flavors(REQUEST, None, CTX, 1)
    assert exc.value.status_code == 404


def _all_paths(router):
    paths = []
    for r in getattr(router, "routes", []):
        path = getattr(r, "path", None)
        if path is not None:
            paths.append(path)
        paths.extend(_all_paths(r))
    return paths


@pytest.mark.asyncio
async def test_aggregated_flavors_empty_clusters_short_circuits(monkeypatch):
    # Zero visible clusters must NOT reach the gateway, whose empty-cluster
    # filter would otherwise return the whole fleet.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(flavor_routes, "async_session", lambda: _FakeSession())

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        return []

    monkeypatch.setattr(flavor_routes.Cluster, "all_by_fields", fake_all)

    async def boom(*a, **kw):
        raise AssertionError("gateway must not be called for zero clusters")

    monkeypatch.setattr(
        flavor_routes.gateway_client, "list_instance_type_flavors", boom
    )

    out = await flavor_routes.get_gpu_aggregated_instance_type_flavors(CTX)
    assert out.items == []


def test_flavor_routes_registered():
    by_path = {}
    for r in flavor_routes.router.routes:
        methods = getattr(r, "methods", None)
        if methods:
            by_path.setdefault(r.path, set()).update(methods)

    assert "GET" in by_path["/aggregated"]
    assert "GET" in by_path[""]


def test_flavor_route_registered_in_app():
    from gpustack.routes.routes import api_router

    assert any("gpu-instance-type-flavors" in p for p in _all_paths(api_router))
