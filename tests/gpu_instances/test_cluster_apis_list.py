import http
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from kubernetes_asyncio import client

from gpustack.gpu_instances import cluster_apis
from gpustack.gpu_instances.cluster_apis import (
    ClusterOps,
    _CRDSpec,
    _Scope,
    _INSTANCE,
    _INSTANCE_TYPE,
    _INSTANCE_TYPE_FLAVOR,
    _SSH_PUBLIC_KEY,
)
from gpustack.gpu_instances.cluster_apis_util import DEFAULT_SYSTEM_NAMESPACE

# The worker CRD group/version as they must appear on the wire, pinned here
# instead of imported from the module under test: an assertion that reads the
# same constant the code reads would mirror a regression rather than catch it.
_GROUP = "worker.gpustack.ai"
_VERSION = "v1"

# The org namespace the ``ops`` fixture's principal derives.
_ORG_NAMESPACE = "gpustack-default"

# A system-namespaced spec in a *different* group, standing in for the
# operator's ``gpustack.ai/v1 Setting`` (owned by the task that adds it for
# real). It exists so the scope cases below prove group, version and namespace
# are all read off the spec rather than off the worker defaults.
_FOREIGN_SYSTEM = _CRDSpec(
    plural="settings",
    kind="Setting",
    scope=_Scope.SYSTEM_NAMESPACED,
    group="gpustack.ai",
)

# (spec, expected group, expected version, expected namespace) — ``None``
# namespace means the call must go out cluster-scoped.
_SCOPE_CASES = [
    pytest.param(_INSTANCE_TYPE, _GROUP, _VERSION, None, id="cluster-scoped"),
    pytest.param(
        _SSH_PUBLIC_KEY, _GROUP, _VERSION, _ORG_NAMESPACE, id="org-namespaced"
    ),
    pytest.param(
        _FOREIGN_SYSTEM,
        "gpustack.ai",
        "v1",
        DEFAULT_SYSTEM_NAMESPACE,
        id="system-namespaced",
    ),
]


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
async def test_list_devices_delegates(monkeypatch, ops):
    captured = {}

    async def fake_list(spec, resource_version=None):
        captured["spec"] = spec
        return {"items": [{"metadata": {"name": "node-a"}}]}

    monkeypatch.setattr(ops, "_list", fake_list)

    out = await ops.list_devices()

    assert out["items"][0]["metadata"]["name"] == "node-a"
    # Devices is cluster-scoped: one object per node, not per namespace.
    assert captured["spec"].plural == "devices"
    assert captured["spec"].scope is _Scope.CLUSTER


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

    events = [evt async for evt in ops.watch_instance_types("42")]

    # Watches the cluster-scoped list call with the InstanceType GVR ...
    assert captured["func"] is list_cluster
    assert captured["kwargs"] == {
        "group": _GROUP,
        "version": _VERSION,
        "plural": _INSTANCE_TYPE.plural,
        "resource_version": "42",
    }
    # ... yields native events unchanged, and closes the Watch on exit.
    assert events == [{"type": "ADDED", "raw_object": {"metadata": {"name": "it-a"}}}]
    assert captured["watch"].closed


@pytest.mark.asyncio
async def test_watch_without_resource_version_lists_for_one(monkeypatch, ops):
    # A version-less watch reads as a WatchList request to the worker cluster's
    # aggregated apiserver and is rejected (422), so one is listed first.
    captured = {}
    _install_fake_watch(monkeypatch, captured)

    crd = MagicMock()
    crd.list_cluster_custom_object = object()
    monkeypatch.setattr(ops, "_crd", lambda: crd)
    monkeypatch.setattr(
        ops,
        "_list",
        AsyncMock(return_value={"metadata": {"resourceVersion": "913"}, "items": []}),
    )

    _ = [evt async for evt in ops.watch_instance_types()]

    assert captured["kwargs"]["resource_version"] == "913"


@pytest.mark.asyncio
async def test_watch_tolerates_list_without_metadata(monkeypatch, ops):
    # A list answer carrying no usable metadata leaves the watch version-less
    # (the apiserver then rejects it, which the caller already retries) instead
    # of raising on a None metadata.
    captured = {}
    _install_fake_watch(monkeypatch, captured)

    crd = MagicMock()
    crd.list_cluster_custom_object = object()
    monkeypatch.setattr(ops, "_crd", lambda: crd)
    monkeypatch.setattr(ops, "_list", AsyncMock(return_value={"metadata": None}))

    _ = [evt async for evt in ops.watch_instance_types()]

    assert "resource_version" not in captured["kwargs"]


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


#
# Group / version / namespace resolution off the spec
#


def _fake_crd(monkeypatch, ops, **returns):
    """Install a ``CustomObjectsApi`` double on ``ops`` with every generic
    call stubbed, so a helper's choice of call *and* its kwargs can be
    asserted without a live cluster."""
    crd = MagicMock()
    for name in (
        "get_cluster_custom_object",
        "get_namespaced_custom_object",
        "list_cluster_custom_object",
        "list_namespaced_custom_object",
        "create_cluster_custom_object",
        "create_namespaced_custom_object",
        "patch_cluster_custom_object",
        "patch_namespaced_custom_object",
        "delete_cluster_custom_object",
        "delete_namespaced_custom_object",
    ):
        setattr(crd, name, AsyncMock(return_value=returns.get(name, {"ok": True})))
    monkeypatch.setattr(ops, "_crd", lambda: crd)
    return crd


def _api_exception(status: int) -> client.exceptions.ApiException:
    return client.exceptions.ApiException(status=status, reason="boom")


def test_system_namespace_defaults_to_gpustack_system(ops):
    # A cluster whose ``k8s_options.namespace`` is unset renders its operator
    # into ``gpustack-system``, so that is where its system-namespaced
    # resources are read from.
    assert DEFAULT_SYSTEM_NAMESPACE == "gpustack-system"
    assert ops.system_namespace == "gpustack-system"


@pytest.mark.asyncio
async def test_system_namespace_follows_cluster_k8s_options_namespace(monkeypatch):
    o = ClusterOps(
        server_api_port=1,
        cluster_id=42,
        cluster_registration_token="tok",
        cluster_owner_principal_identifier="default",
        system_namespace="ops-ns",
    )
    try:
        assert o.system_namespace == "ops-ns"
        crd = _fake_crd(monkeypatch, o)

        await o._read(_FOREIGN_SYSTEM, "instance-type-derived-from-node")

        crd.get_namespaced_custom_object.assert_awaited_once_with(
            group="gpustack.ai",
            version="v1",
            plural="settings",
            namespace="ops-ns",
            name="instance-type-derived-from-node",
        )
    finally:
        await o.close()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_read_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)

    assert await ops._read(spec, "obj") == {"ok": True}

    if namespace is None:
        crd.get_cluster_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural, name="obj"
        )
        crd.get_namespaced_custom_object.assert_not_awaited()
    else:
        crd.get_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            name="obj",
        )
        crd.get_cluster_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_list_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)

    await ops._list(spec)

    if namespace is None:
        crd.list_cluster_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural
        )
        crd.list_namespaced_custom_object.assert_not_awaited()
    else:
        crd.list_namespaced_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural, namespace=namespace
        )
        crd.list_cluster_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_watch_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    captured = {}
    _install_fake_watch(monkeypatch, captured)
    crd = _fake_crd(monkeypatch, ops)

    _ = [evt async for evt in ops._watch(spec, resource_version="7")]

    expected = {
        "group": group,
        "version": version,
        "plural": spec.plural,
        "resource_version": "7",
    }
    if namespace is None:
        assert captured["func"] is crd.list_cluster_custom_object
    else:
        assert captured["func"] is crd.list_namespaced_custom_object
        expected["namespace"] = namespace
    assert captured["kwargs"] == expected


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_create_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)
    monkeypatch.setattr(ops, "ensure_org_namespace", AsyncMock())

    await ops._create(spec, {"metadata": {"name": "obj"}, "spec": {"a": 1}}, False)

    body = {
        "apiVersion": f"{group}/{version}",
        "kind": spec.kind,
        "metadata": {"name": "obj"},
        "spec": {"a": 1},
    }
    if namespace is None:
        crd.create_cluster_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural, body=body
        )
        crd.create_namespaced_custom_object.assert_not_awaited()
    else:
        body["metadata"] = {"name": "obj", "namespace": namespace}
        crd.create_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            body=body,
        )
        crd.create_cluster_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_upsert_patches_with_resolved_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)
    monkeypatch.setattr(ops, "ensure_org_namespace", AsyncMock())

    await ops._upsert(spec, {"metadata": {"name": "obj"}, "spec": {"a": 1}})

    if namespace is None:
        crd.patch_cluster_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            name="obj",
            body={"spec": {"a": 1}},
            _content_type="application/merge-patch+json",
        )
        crd.patch_namespaced_custom_object.assert_not_awaited()
    else:
        crd.patch_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            name="obj",
            body={"spec": {"a": 1}},
            _content_type="application/merge-patch+json",
        )
        crd.patch_cluster_custom_object.assert_not_awaited()
    # The upsert's patch branch never falls through to a create.
    crd.create_cluster_custom_object.assert_not_awaited()
    crd.create_namespaced_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_patch_spec_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)

    assert await ops._patch_spec(spec, "obj", {"value": "on"}) == {"ok": True}

    if namespace is None:
        crd.patch_cluster_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            name="obj",
            body={"spec": {"value": "on"}},
            _content_type="application/merge-patch+json",
        )
        crd.patch_namespaced_custom_object.assert_not_awaited()
    else:
        crd.patch_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            name="obj",
            body={"spec": {"value": "on"}},
            _content_type="application/merge-patch+json",
        )
        crd.patch_cluster_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_delete_resolves_gvr_and_namespace(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)

    assert await ops._delete(spec, "obj") is True

    if namespace is None:
        crd.delete_cluster_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural, name="obj"
        )
        crd.delete_namespaced_custom_object.assert_not_awaited()
    else:
        crd.delete_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            name="obj",
        )
        crd.delete_cluster_custom_object.assert_not_awaited()


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
def test_envelope_stamps_api_version_and_scope_namespace(
    ops, spec, group, version, namespace
):
    body = ops._envelope(spec, {"metadata": {"name": "obj"}, "spec": {"a": 1}})

    assert body["apiVersion"] == f"{group}/{version}"
    assert body["kind"] == spec.kind
    assert body["spec"] == {"a": 1}
    if namespace is None:
        assert "namespace" not in body["metadata"]
    else:
        assert body["metadata"]["namespace"] == namespace


@pytest.mark.parametrize(
    "spec, ensured",
    [
        pytest.param(_INSTANCE_TYPE, False, id="cluster-scoped"),
        pytest.param(_SSH_PUBLIC_KEY, True, id="org-namespaced"),
        pytest.param(_FOREIGN_SYSTEM, False, id="system-namespaced"),
    ],
)
@pytest.mark.asyncio
async def test_create_ensures_only_the_org_namespace(monkeypatch, ops, spec, ensured):
    # The org namespace is GPUStack's to create; the cluster's system namespace
    # belongs to the operator's own deployment and must never be created here.
    _fake_crd(monkeypatch, ops)
    ensure = AsyncMock()
    monkeypatch.setattr(ops, "ensure_org_namespace", ensure)

    await ops._create(spec, {"metadata": {"name": "obj"}, "spec": {}}, False)

    assert ensure.await_count == (1 if ensured else 0)


@pytest.mark.parametrize(
    "spec, ensured",
    [
        pytest.param(_INSTANCE_TYPE, False, id="cluster-scoped"),
        pytest.param(_SSH_PUBLIC_KEY, True, id="org-namespaced"),
        pytest.param(_FOREIGN_SYSTEM, False, id="system-namespaced"),
    ],
)
@pytest.mark.asyncio
async def test_upsert_ensures_only_the_org_namespace(monkeypatch, ops, spec, ensured):
    _fake_crd(monkeypatch, ops)
    ensure = AsyncMock()
    monkeypatch.setattr(ops, "ensure_org_namespace", ensure)

    await ops._upsert(spec, {"metadata": {"name": "obj"}, "spec": {}})

    assert ensure.await_count == (1 if ensured else 0)


#
# Error paths shared by every scope
#


@pytest.mark.asyncio
async def test_read_returns_none_on_404(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.get_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )

    assert await ops._read(_INSTANCE_TYPE, "gone") is None


@pytest.mark.asyncio
async def test_read_reraises_other_api_errors(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.get_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.FORBIDDEN
    )

    with pytest.raises(client.exceptions.ApiException):
        await ops._read(_INSTANCE_TYPE, "denied")


@pytest.mark.asyncio
async def test_create_reads_back_when_ignoring_an_existing_object(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    monkeypatch.setattr(ops, "_read", AsyncMock(return_value={"existing": True}))

    out = await ops._create(
        _INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}}, True
    )

    assert out == {"existing": True}
    crd.create_cluster_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_reads_back_on_a_raced_409(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.create_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.CONFLICT
    )
    # Absent up-front, then created by a racing writer between the two calls.
    monkeypatch.setattr(ops, "_read", AsyncMock(side_effect=[None, {"raced": True}]))

    out = await ops._create(
        _INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}}, True
    )

    assert out == {"raced": True}


@pytest.mark.asyncio
async def test_create_reraises_409_when_not_ignoring_existed(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.create_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.CONFLICT
    )

    with pytest.raises(client.exceptions.ApiException):
        await ops._create(
            _INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}}, False
        )


@pytest.mark.parametrize("spec, group, version, namespace", _SCOPE_CASES)
@pytest.mark.asyncio
async def test_upsert_creates_with_the_resolved_gvr_when_the_patch_404s(
    monkeypatch, ops, spec, group, version, namespace
):
    crd = _fake_crd(monkeypatch, ops)
    monkeypatch.setattr(ops, "ensure_org_namespace", AsyncMock())
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )
    crd.patch_namespaced_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )

    out = await ops._upsert(spec, {"metadata": {"name": "obj"}, "spec": {}})

    assert out == {"ok": True}
    body = {
        "apiVersion": f"{group}/{version}",
        "kind": spec.kind,
        "metadata": {"name": "obj"},
        "spec": {},
    }
    if namespace is None:
        crd.create_cluster_custom_object.assert_awaited_once_with(
            group=group, version=version, plural=spec.plural, body=body
        )
        crd.create_namespaced_custom_object.assert_not_awaited()
    else:
        body["metadata"] = {"name": "obj", "namespace": namespace}
        crd.create_namespaced_custom_object.assert_awaited_once_with(
            group=group,
            version=version,
            plural=spec.plural,
            namespace=namespace,
            body=body,
        )
        crd.create_cluster_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_reraises_a_non_404_patch_failure(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.FORBIDDEN
    )

    with pytest.raises(client.exceptions.ApiException):
        await ops._upsert(_INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}})

    crd.create_cluster_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_reads_back_when_the_create_races(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )
    crd.create_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.CONFLICT
    )
    monkeypatch.setattr(ops, "_read", AsyncMock(return_value={"raced": True}))

    out = await ops._upsert(_INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}})

    assert out == {"raced": True}


@pytest.mark.asyncio
async def test_upsert_reraises_when_the_raced_object_is_already_gone(monkeypatch, ops):
    # 409 then a read-back that finds nothing: there is no post-condition to
    # return, so the conflict propagates instead of reading as a success.
    crd = _fake_crd(monkeypatch, ops)
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )
    crd.create_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.CONFLICT
    )
    monkeypatch.setattr(ops, "_read", AsyncMock(return_value=None))

    with pytest.raises(client.exceptions.ApiException):
        await ops._upsert(_INSTANCE_TYPE, {"metadata": {"name": "obj"}, "spec": {}})


@pytest.mark.asyncio
async def test_patch_spec_returns_none_on_404(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )

    assert await ops._patch_spec(_INSTANCE_TYPE, "gone", {"inactive": True}) is None


@pytest.mark.asyncio
async def test_patch_spec_reraises_other_api_errors(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.patch_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.FORBIDDEN
    )

    with pytest.raises(client.exceptions.ApiException):
        await ops._patch_spec(_INSTANCE_TYPE, "denied", {"inactive": True})


@pytest.mark.asyncio
async def test_delete_returns_false_on_404(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.delete_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.NOT_FOUND
    )

    assert await ops._delete(_INSTANCE_TYPE, "gone") is False


@pytest.mark.asyncio
async def test_delete_reraises_other_api_errors(monkeypatch, ops):
    crd = _fake_crd(monkeypatch, ops)
    crd.delete_cluster_custom_object.side_effect = _api_exception(
        http.HTTPStatus.FORBIDDEN
    )

    with pytest.raises(client.exceptions.ApiException):
        await ops._delete(_INSTANCE_TYPE, "denied")
