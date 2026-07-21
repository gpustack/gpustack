from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from gpustack.gpu_instances import cluster_apis
from gpustack.gpu_instances.cluster_apis import (
    ClusterOps,
    _GROUP,
    _VERSION,
    _INSTANCE,
    _INSTANCE_TYPE,
    _INSTANCE_TYPE_FLAVOR,
    _SSH_PUBLIC_KEY,
)


@pytest_asyncio.fixture
async def ops():
    o = ClusterOps(
        server_api_port=1,
        cluster_id=42,
        cluster_registration_token="tok",
        cluster_owner_principal_identifier="default",
    )
    yield o
    await o.close()


@pytest.mark.asyncio
async def test_list_cluster_scoped_passes_resource_version(monkeypatch, ops):
    crd = MagicMock()
    crd.list_cluster_custom_object = AsyncMock(return_value={"items": []})
    crd.list_namespaced_custom_object = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(ops, "_crd", lambda: crd)

    out = await ops._list(_INSTANCE_TYPE, resource_version="123")

    assert out == {"items": []}
    crd.list_cluster_custom_object.assert_awaited_once_with(
        group=_GROUP,
        version=_VERSION,
        plural=_INSTANCE_TYPE.plural,
        resource_version="123",
    )
    crd.list_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_cluster_scoped_omits_resource_version_when_absent(monkeypatch, ops):
    crd = MagicMock()
    crd.list_cluster_custom_object = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(ops, "_crd", lambda: crd)

    await ops._list(_INSTANCE_TYPE)

    crd.list_cluster_custom_object.assert_awaited_once_with(
        group=_GROUP,
        version=_VERSION,
        plural=_INSTANCE_TYPE.plural,
    )


@pytest.mark.asyncio
async def test_list_namespaced_uses_org_namespace(monkeypatch, ops):
    crd = MagicMock()
    crd.list_namespaced_custom_object = AsyncMock(return_value={"items": []})
    monkeypatch.setattr(ops, "_crd", lambda: crd)

    await ops._list(_SSH_PUBLIC_KEY, resource_version="7")

    crd.list_namespaced_custom_object.assert_awaited_once_with(
        group=_GROUP,
        version=_VERSION,
        plural=_SSH_PUBLIC_KEY.plural,
        namespace=ops.org_namespace,
        resource_version="7",
    )


@pytest.mark.asyncio
async def test_list_instance_types_delegates(monkeypatch, ops):
    captured = {}

    async def fake_list(spec, resource_version=None):
        captured["spec"] = spec
        captured["rv"] = resource_version
        return {"items": [{"metadata": {"name": "a"}}]}

    monkeypatch.setattr(ops, "_list", fake_list)

    out = await ops.list_instance_types(resource_version="9")

    assert out["items"][0]["metadata"]["name"] == "a"
    assert captured["spec"] is _INSTANCE_TYPE
    assert captured["rv"] == "9"


@pytest.mark.asyncio
async def test_create_instance_type_wraps_body(monkeypatch, ops):
    captured = {}

    async def fake_create(spec, body, ignore_existed):
        captured["spec"] = spec
        captured["body"] = body
        captured["ignore_existed"] = ignore_existed
        return {"metadata": {"name": body["metadata"]["name"]}, "spec": body["spec"]}

    monkeypatch.setattr(ops, "_create", fake_create)

    out = await ops.create_instance_type("it-1", {"acceleratable": True})

    assert captured["spec"] is _INSTANCE_TYPE
    assert captured["body"] == {
        "metadata": {"name": "it-1"},
        "spec": {"acceleratable": True},
    }
    assert captured["ignore_existed"] is True
    assert out["metadata"]["name"] == "it-1"


@pytest.mark.asyncio
async def test_delete_instance_type_delegates(monkeypatch, ops):
    captured = {}

    async def fake_delete(spec, name):
        captured["spec"] = spec
        captured["name"] = name
        return True

    monkeypatch.setattr(ops, "_delete", fake_delete)

    assert await ops.delete_instance_type("it-1") is True
    assert captured["spec"] is _INSTANCE_TYPE
    assert captured["name"] == "it-1"


@pytest.mark.asyncio
async def test_deactivate_instance_type_patches_inactive_true(monkeypatch, ops):
    captured = {}

    async def fake_patch(spec, name, body_spec):
        captured["spec"] = spec
        captured["name"] = name
        captured["body_spec"] = body_spec
        return {"ok": True}

    monkeypatch.setattr(ops, "_patch_spec", fake_patch)

    out = await ops.deactivate_instance_type("it-1")

    assert out == {"ok": True}
    assert captured["spec"] is _INSTANCE_TYPE
    assert captured["name"] == "it-1"
    assert captured["body_spec"] == {"inactive": True}


@pytest.mark.asyncio
async def test_activate_instance_type_patches_inactive_false(monkeypatch, ops):
    captured = {}

    async def fake_patch(spec, name, body_spec):
        captured["spec"] = spec
        captured["name"] = name
        captured["body_spec"] = body_spec
        return {"ok": True}

    monkeypatch.setattr(ops, "_patch_spec", fake_patch)

    await ops.activate_instance_type("it-1")

    assert captured["spec"] is _INSTANCE_TYPE
    assert captured["name"] == "it-1"
    assert captured["body_spec"] == {"inactive": False}


@pytest.mark.asyncio
async def test_deactivate_instance_type_passthrough_none(monkeypatch, ops):
    # _patch_spec returns None on 404; the method passes it through so the
    # route can 404.
    async def fake_patch(spec, name, body_spec):
        return None

    monkeypatch.setattr(ops, "_patch_spec", fake_patch)

    assert await ops.deactivate_instance_type("gone") is None


@pytest.mark.asyncio
async def test_start_instance_patches_stop_false(monkeypatch, ops):
    captured = {}

    async def fake_patch(spec, name, body_spec):
        captured["spec"] = spec
        captured["name"] = name
        captured["body_spec"] = body_spec
        return {"ok": True}

    monkeypatch.setattr(ops, "_patch_spec", fake_patch)

    await ops.start_instance("inst-1")

    assert captured["spec"] is _INSTANCE
    assert captured["name"] == "inst-1"
    assert captured["body_spec"] == {"stop": False}


@pytest.mark.asyncio
async def test_start_instance_merges_spec_with_stop_false(monkeypatch, ops):
    captured = {}

    async def fake_patch(spec, name, body_spec):
        captured["body_spec"] = body_spec
        return {"ok": True}

    monkeypatch.setattr(ops, "_patch_spec", fake_patch)

    await ops.start_instance("inst-1", spec={"foo": "bar"})

    # Spec is re-applied on resume, with stop explicitly false (not removed).
    assert captured["body_spec"] == {"foo": "bar", "stop": False}


def _install_fake_watch(monkeypatch, captured, events=()):
    """Swap kubernetes_asyncio's Watch for a fake that records the streamed
    ``func``/``kwargs`` and replays ``events``, so watch calls can be asserted
    without a live cluster."""

    class FakeWatch:
        def __init__(self):
            captured["watch"] = self
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

        def stream(self, func, **kwargs):
            captured["func"] = func
            captured["kwargs"] = kwargs

            async def _events():
                for evt in events:
                    yield evt

            return _events()

    monkeypatch.setattr(cluster_apis.watch, "Watch", FakeWatch)


@pytest.mark.asyncio
async def test_watch_instance_types_builds_cluster_watch(monkeypatch, ops):
    captured = {}
    _install_fake_watch(
        monkeypatch,
        captured,
        events=[{"type": "ADDED", "raw_object": {"metadata": {"name": "it-a"}}}],
    )

    list_cluster = object()
    crd = MagicMock()
    crd.list_cluster_custom_object = list_cluster
    monkeypatch.setattr(ops, "_crd", lambda: crd)

    events = [evt async for evt in ops.watch_instance_types()]

    # Watches the cluster-scoped list call with the InstanceType GVR ...
    assert captured["func"] is list_cluster
    assert captured["kwargs"] == {
        "group": _GROUP,
        "version": _VERSION,
        "plural": _INSTANCE_TYPE.plural,
    }
    # ... yields native events unchanged, and closes the Watch on exit.
    assert events == [{"type": "ADDED", "raw_object": {"metadata": {"name": "it-a"}}}]
    assert captured["watch"].closed


@pytest.mark.asyncio
async def test_watch_namespaced_uses_org_namespace(monkeypatch, ops):
    captured = {}
    _install_fake_watch(monkeypatch, captured)

    list_namespaced = object()
    crd = MagicMock()
    crd.list_namespaced_custom_object = list_namespaced
    monkeypatch.setattr(ops, "_crd", lambda: crd)

    # A namespaced spec streams the namespaced call scoped to the org namespace,
    # passing resource_version through.
    _ = [evt async for evt in ops._watch(_SSH_PUBLIC_KEY, resource_version="7")]

    assert captured["func"] is list_namespaced
    assert captured["kwargs"] == {
        "group": _GROUP,
        "version": _VERSION,
        "plural": _SSH_PUBLIC_KEY.plural,
        "namespace": ops.org_namespace,
        "resource_version": "7",
    }


@pytest.mark.asyncio
async def test_list_instance_type_flavors_delegates(monkeypatch, ops):
    captured = {}

    async def fake_list(spec, resource_version=None):
        captured["spec"] = spec
        captured["rv"] = resource_version
        return {"items": []}

    monkeypatch.setattr(ops, "_list", fake_list)

    await ops.list_instance_type_flavors(resource_version="5")

    assert captured["spec"] is _INSTANCE_TYPE_FLAVOR
    assert captured["rv"] == "5"
