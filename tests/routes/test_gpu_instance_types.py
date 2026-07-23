"""Per-cluster GPU instance-type route tests.

Handlers are called directly with a fake ``ctx`` / ``request`` and a
monkeypatched ``ClusterOps`` / ``Cluster.one_by_id`` — no live cluster or DB.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from kubernetes_asyncio import client
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.routes import gpu_instance_types as it_routes
from gpustack.routes import gpu_instances_helper as helper
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceTypeCreate,
    GPUInstanceTypeSpec,
    GPUInstanceTypeSpecUpdate,
    GPUInstanceTypeUpdate,
)
from gpustack.schemas.principals import OrgRole, PrincipalType

# SYSTEM principal → bypasses tenant filters (visible + writable everywhere).
CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
    scoped_cluster_id=None,
)

# Org MEMBER who can SEE cluster 1 via a grant but does not OWN it (owner is
# principal 999) → read passes, write is forbidden.
CTX_NON_WRITER = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.USER, id=5),
    is_platform_admin=False,
    current_principal_id=10,
    org_role=OrgRole.MEMBER,
    current_is_personal_scope=False,
    scoped_cluster_id=None,
    accessible_cluster_ids={1},
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

    monkeypatch.setattr(it_routes.Cluster, "one_by_id", fake_one_by_id)


def _patch_ops(
    monkeypatch,
    *,
    list_result=None,
    create_result=None,
    delete_existed=True,
    patch_absent=False,
    capture=None,
):
    # Record calls into a throwaway dict when the caller doesn't want to inspect
    # them, so the fakes never guard on ``capture is not None``.
    capture = capture if capture is not None else {}

    class FakeOps:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_instance_types(self, resource_version=None):
            return list_result

        async def create_instance_type(self, name, spec, ignore_existed=True):
            capture["name"] = name
            capture["spec"] = spec
            return (
                create_result
                if create_result is not None
                else {
                    "metadata": {"name": name},
                    "spec": spec,
                }
            )

        async def update_instance_type(self, name, spec):
            capture["name"] = name
            capture["spec"] = spec
            if patch_absent:
                return None
            return {
                "metadata": {"name": name},
                "spec": spec,
                "status": {"phase": "Active"},
            }

        async def delete_instance_type(self, name):
            capture["deleted"] = name
            return delete_existed

        async def deactivate_instance_type(self, name):
            return self._patch(name, inactive=True, phase="Inactive")

        async def activate_instance_type(self, name):
            return self._patch(name, inactive=False, phase="Active")

        def _patch(self, name, *, inactive, phase):
            capture["name"] = name
            capture["inactive"] = inactive
            if patch_absent:
                return None
            return {
                "metadata": {"name": name},
                "spec": {"acceleratable": True},
                "status": {"phase": phase},
            }

    # build_cluster_ops lives in the shared helper and resolves ClusterOps
    # from that module's globals.
    monkeypatch.setattr(helper, "ClusterOps", FakeOps)


def _patch_watch_ops(monkeypatch, watch_events):
    """Patch ClusterOps with a minimal stand-in whose watch replays
    ``watch_events`` — the watch path needs no list/create/delete methods."""

    class FakeWatchOps:
        def __init__(self, **kwargs):
            self.cluster_id = kwargs.get("cluster_id")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def watch_instance_types(self, resource_version=None):
            for evt in watch_events:
                yield evt

    monkeypatch.setattr(helper, "ClusterOps", FakeWatchOps)


def _cluster(id_=1, owner_principal_id=None):
    return SimpleNamespace(
        id=id_,
        owner_principal_id=owner_principal_id,
        registration_token="tok",
    )


@pytest.mark.asyncio
async def test_list_maps_metadata_name(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(
        monkeypatch,
        list_result={
            "items": [
                {
                    "metadata": {"name": "it-a"},
                    "spec": {
                        "acceleratable": True,
                        "manufacturer": "nvidia",
                        "displayName": "A10G Pool",
                    },
                    "status": {"phase": "Active"},
                }
            ]
        },
    )

    out = await it_routes.get_gpu_instance_types(REQUEST, None, CTX, 1)

    assert len(out.items) == 1
    assert out.items[0].name == "it-a"
    assert out.items[0].spec.acceleratable is True
    # displayName from the raw CR spec survives the CR→public read mapping.
    assert out.items[0].spec.display_name == "A10G Pool"
    assert out.items[0].status.phase == "Active"


@pytest.mark.asyncio
async def test_create_sends_spec_and_defaults_missing_status(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, capture=capture)

    body = GPUInstanceTypeCreate(
        name="new-it",
        spec=GPUInstanceTypeSpec(
            acceleratable=True, os="linux", accelerator_group="nvidia-a10g"
        ),
    )
    out = await it_routes.create_gpu_instance_type(REQUEST, None, CTX, body, 1)

    # The CR spec is the create-spec dumped by camelCase alias, none-excluded —
    # multi-word fields must serialize to camelCase (accelerator_group → acceleratorGroup).
    assert capture["name"] == "new-it"
    assert capture["spec"] == {
        "acceleratable": True,
        "os": "linux",
        "acceleratorGroup": "nvidia-a10g",
    }
    # The ack dict carries no status → maps to an all-None status.
    assert out.name == "new-it"
    assert out.status.phase is None


@pytest.mark.asyncio
async def test_update_sends_editable_spec_and_maps(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, capture=capture)

    body = GPUInstanceTypeUpdate(
        name="it-a",
        spec=GPUInstanceTypeSpecUpdate(display_name="A10G Pool"),
    )
    out = await it_routes.update_gpu_instance_type(REQUEST, None, CTX, body, 1)

    # Only the display name is editable; it is merge-patched by camelCase alias.
    assert capture["name"] == "it-a"
    assert capture["spec"] == {"displayName": "A10G Pool"}
    assert out.name == "it-a"
    assert out.status.phase == "Active"


@pytest.mark.asyncio
async def test_update_absent_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(monkeypatch, patch_absent=True)

    body = GPUInstanceTypeUpdate(name="gone", spec=GPUInstanceTypeSpecUpdate())
    with pytest.raises(NotFoundException) as exc:
        await it_routes.update_gpu_instance_type(REQUEST, None, CTX, body, 1)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_visible_but_not_writable_raises_403(monkeypatch):
    _patch_cluster(monkeypatch, _cluster(owner_principal_id=999))
    _patch_ops(monkeypatch)

    body = GPUInstanceTypeUpdate(name="it-a", spec=GPUInstanceTypeSpecUpdate())
    with pytest.raises(ForbiddenException) as exc:
        await it_routes.update_gpu_instance_type(REQUEST, None, CTX_NON_WRITER, body, 1)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_write_to_invisible_cluster_raises_404(monkeypatch):
    # Cluster 2 is neither owned by nor granted to the caller (accessible is
    # {1}). A write must 404 (not leak its existence via a 403).
    _patch_cluster(monkeypatch, _cluster(id_=2, owner_principal_id=999))
    _patch_ops(monkeypatch)

    body = GPUInstanceTypeUpdate(name="it-a", spec=GPUInstanceTypeSpecUpdate())
    with pytest.raises(NotFoundException) as exc:
        await it_routes.update_gpu_instance_type(REQUEST, None, CTX_NON_WRITER, body, 2)
    assert exc.value.status_code == 404


def test_spec_update_allows_only_display_name():
    # The update spec exposes exactly the display name; every other field is
    # fixed after creation and must stay out of it, while the create spec still
    # carries the full set.
    assert set(GPUInstanceTypeSpecUpdate.model_fields) == {"display_name"}
    create_fields = set(GPUInstanceTypeSpec.model_fields)
    assert {
        "display_name",
        "unit_resources",
        "local_storage",
        "os",
        "arch",
        "accelerator_group",
    } <= create_fields


@pytest.mark.asyncio
async def test_delete_existing_returns_none(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, delete_existed=True, capture=capture)

    ret = await it_routes.delete_gpu_instance_type(REQUEST, None, CTX, "it-a", 1)

    assert ret is None
    assert capture["deleted"] == "it-a"


@pytest.mark.asyncio
async def test_delete_absent_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(monkeypatch, delete_existed=False)

    with pytest.raises(NotFoundException) as exc:
        await it_routes.delete_gpu_instance_type(REQUEST, None, CTX, "gone", 1)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_invisible_cluster_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, None)
    _patch_ops(monkeypatch, list_result={"items": []})

    with pytest.raises(NotFoundException) as exc:
        await it_routes.get_gpu_instance_types(REQUEST, None, CTX, 1)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_visible_but_not_writable_raises_403(monkeypatch):
    # Visible via grant (accessible_cluster_ids), owned by principal 999.
    _patch_cluster(monkeypatch, _cluster(owner_principal_id=999))
    _patch_ops(monkeypatch)

    body = GPUInstanceTypeCreate(name="x", spec=GPUInstanceTypeSpec())
    with pytest.raises(ForbiddenException) as exc:
        await it_routes.create_gpu_instance_type(REQUEST, None, CTX_NON_WRITER, body, 1)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_write_to_invisible_cluster_raises_404(monkeypatch):
    # Cluster 2 is neither owned by nor granted to the caller (accessible is
    # {1}). A write must 404 (not leak its existence via a 403).
    _patch_cluster(monkeypatch, _cluster(id_=2, owner_principal_id=999))
    _patch_ops(monkeypatch)

    body = GPUInstanceTypeCreate(name="x", spec=GPUInstanceTypeSpec())
    with pytest.raises(NotFoundException) as exc:
        await it_routes.create_gpu_instance_type(REQUEST, None, CTX_NON_WRITER, body, 2)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_aggregated_empty_clusters_short_circuits(monkeypatch):
    # Zero visible clusters must NOT reach the gateway, whose empty-cluster
    # filter would otherwise return the whole fleet.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(it_routes, "async_session", lambda: _FakeSession())

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        return []

    monkeypatch.setattr(it_routes.Cluster, "all_by_fields", fake_all)

    async def boom(*a, **kw):
        raise AssertionError("gateway must not be called for zero clusters")

    monkeypatch.setattr(it_routes.gateway_client, "list_instance_types", boom)

    out = await it_routes.get_gpu_aggregated_instance_types(CTX)
    assert out.items == []


def _agg_evt(type_, name, *, once="4", remaining="8"):
    """A gateway WorkerEvent line for the aggregated watch: a Kubernetes verb
    plus an already-aggregated ``object``, framed as ``<json>\\n\\n`` exactly
    like gateway_client._stream emits it."""
    return (
        json.dumps(
            {
                "type": type_,
                "object": {
                    "name": name,
                    "spec": {"acceleratable": True},
                    "status": {
                        "onceMaxRequest": {"accelerator": once},
                        "remaining": {"accelerator": remaining},
                    },
                },
            }
        )
        + "\n\n"
    )


@pytest.mark.asyncio
async def test_aggregated_watch_wraps_gateway_verbs(monkeypatch):
    # The gateway streams raw Kubernetes verbs; the route must map them to
    # GPUStack event types (ADDED→1, MODIFIED→2, DELETED→3), drop BOOKMARK, and
    # forward the cluster filter as strings with aggregated=True.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(it_routes, "async_session", lambda: _FakeSession())

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        return [_cluster(id_=1), _cluster(id_=2)]

    monkeypatch.setattr(it_routes.Cluster, "all_by_fields", fake_all)

    captured = {}

    async def fake_watch(clusters=None, aggregated=False):
        captured["clusters"] = clusters
        captured["aggregated"] = aggregated
        yield _agg_evt("ADDED", "a100")
        yield json.dumps({"type": "BOOKMARK", "object": None}) + "\n\n"
        # A delete carries the gateway's zero-valued object (only name is set);
        # its empty spec/status must still validate through the public model.
        yield (
            json.dumps(
                {
                    "type": "DELETED",
                    "object": {
                        "name": "a100",
                        "spec": {},
                        "status": {
                            "onceMaxRequest": {},
                            "remaining": {},
                            "tiers": None,
                        },
                    },
                }
            )
            + "\n\n"
        )

    monkeypatch.setattr(it_routes.gateway_client, "watch_instance_types", fake_watch)

    resp = await it_routes.get_gpu_aggregated_instance_types(CTX, watch=True)

    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    frames = [frame async for frame in resp.body_iterator]
    payloads = [json.loads(f) for f in frames if f != "\n\n"]

    # ADDED→1, BOOKMARK dropped, DELETED→3.
    assert [p["type"] for p in payloads] == [1, 3]
    assert payloads[0]["data"]["name"] == "a100"
    # The aggregated status survives, serialized by camelCase alias.
    assert payloads[0]["data"]["status"]["onceMaxRequest"]["accelerator"] == "4"
    # Cluster ids forwarded to the gateway as strings, aggregated=True.
    assert captured["clusters"] == ["1", "2"]
    assert captured["aggregated"] is True


def test_spec_create_display_name_camel_alias():
    # displayName must serialize by camelCase alias, none-excluded, so the
    # create route forwards it into the CR spec.
    dumped = GPUInstanceTypeSpec(display_name="A10G Pool").model_dump(
        by_alias=True, exclude_none=True
    )
    assert dumped["displayName"] == "A10G Pool"
    assert "display_name" not in dumped


def test_spec_display_name_round_trips_from_camel():
    # A read spec dict from the CR carries camelCase; it must populate
    # display_name on the read model.
    spec = GPUInstanceTypeSpec.model_validate({"displayName": "A10G Pool"})
    assert spec.display_name == "A10G Pool"


@pytest.mark.asyncio
async def test_create_forwards_display_name(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, capture=capture)

    body = GPUInstanceTypeCreate(
        name="new-it",
        spec=GPUInstanceTypeSpec(acceleratable=True, display_name="A10G Pool"),
    )
    await it_routes.create_gpu_instance_type(REQUEST, None, CTX, body, 1)

    assert capture["spec"]["displayName"] == "A10G Pool"


@pytest.mark.asyncio
async def test_deactivate_patches_inactive_and_maps(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, capture=capture)

    out = await it_routes.deactivate_gpu_instance_type(REQUEST, None, CTX, "it-a", 1)

    assert capture["name"] == "it-a"
    assert capture["inactive"] is True
    assert out.name == "it-a"
    assert out.status.phase == "Inactive"


@pytest.mark.asyncio
async def test_activate_patches_inactive_and_maps(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(monkeypatch, capture=capture)

    out = await it_routes.activate_gpu_instance_type(REQUEST, None, CTX, "it-a", 1)

    assert capture["name"] == "it-a"
    assert capture["inactive"] is False
    assert out.name == "it-a"
    assert out.status.phase == "Active"


@pytest.mark.asyncio
async def test_deactivate_absent_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(monkeypatch, patch_absent=True)

    with pytest.raises(NotFoundException) as exc:
        await it_routes.deactivate_gpu_instance_type(REQUEST, None, CTX, "gone", 1)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_activate_absent_raises_404(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_ops(monkeypatch, patch_absent=True)

    with pytest.raises(NotFoundException) as exc:
        await it_routes.activate_gpu_instance_type(REQUEST, None, CTX, "gone", 1)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_deactivate_visible_but_not_writable_raises_403(monkeypatch):
    _patch_cluster(monkeypatch, _cluster(owner_principal_id=999))
    _patch_ops(monkeypatch)

    with pytest.raises(ForbiddenException) as exc:
        await it_routes.deactivate_gpu_instance_type(
            REQUEST, None, CTX_NON_WRITER, "it-a", 1
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_activate_visible_but_not_writable_raises_403(monkeypatch):
    _patch_cluster(monkeypatch, _cluster(owner_principal_id=999))
    _patch_ops(monkeypatch)

    with pytest.raises(ForbiddenException) as exc:
        await it_routes.activate_gpu_instance_type(
            REQUEST, None, CTX_NON_WRITER, "it-a", 1
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_activation_write_to_invisible_cluster_raises_404(monkeypatch):
    # Cluster 2 is neither owned by nor granted to the caller (accessible is
    # {1}). A mutation must 404 (not leak its existence via a 403).
    _patch_cluster(monkeypatch, _cluster(id_=2, owner_principal_id=999))
    _patch_ops(monkeypatch)

    with pytest.raises(NotFoundException) as exc:
        await it_routes.deactivate_gpu_instance_type(
            REQUEST, None, CTX_NON_WRITER, "it-a", 2
        )
    assert exc.value.status_code == 404


#
# Watch wiring (get_gpu_instance_types?watch=true) tests.
#


@pytest.mark.asyncio
async def test_get_watch_returns_streamed_frames(monkeypatch):
    _patch_cluster(monkeypatch, _cluster())
    _patch_watch_ops(
        monkeypatch,
        [
            {
                "type": "ADDED",
                "raw_object": {
                    "metadata": {"name": "it-a"},
                    "spec": {"acceleratable": True},
                    "status": {"phase": "Active"},
                },
            },
            {"type": "BOOKMARK", "raw_object": {"metadata": {"resourceVersion": "9"}}},
            {
                "type": "DELETED",
                "raw_object": {
                    "metadata": {"name": "it-a"},
                    "spec": {},
                    "status": {"phase": "Terminating"},
                },
            },
        ],
    )

    resp = await it_routes.get_gpu_instance_types(REQUEST, None, CTX, 1, watch=True)

    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    frames = [frame async for frame in resp.body_iterator]
    payloads = [json.loads(f) for f in frames if f != "\n\n"]
    # ADDED→1, BOOKMARK dropped, DELETED→3.
    assert [p["type"] for p in payloads] == [1, 3]
    assert payloads[0]["data"]["name"] == "it-a"


@pytest.mark.asyncio
async def test_get_watch_invisible_cluster_raises_404(monkeypatch):
    # The visibility check must 404 before any stream is opened.
    _patch_cluster(monkeypatch, None)
    _patch_watch_ops(monkeypatch, [{"type": "ADDED", "raw_object": {}}])

    with pytest.raises(NotFoundException) as exc:
        await it_routes.get_gpu_instance_types(REQUEST, None, CTX, 1, watch=True)
    assert exc.value.status_code == 404


#
# Per-cluster watch stream tests: the shared SSE helper fed by the per-cluster
# source, exactly as the watch route composes it inline.
#


def _cluster_stream(ops):
    """Rebuild the per-cluster watch stream the route composes inline —
    ``watch_event_stream`` fed by the per-cluster source + CR→public mapper."""
    return helper.watch_event_stream(
        it_routes._cluster_instance_type_events(ops),
        it_routes._to_instance_type_public,
    )


def _watch_evt(type_, name, phase="Active"):
    """A native kubernetes_asyncio watch event dict (``type`` + ``raw_object``)."""
    return {
        "type": type_,
        "raw_object": {
            "metadata": {"name": name},
            "spec": {"acceleratable": True},
            "status": {"phase": phase},
        },
    }


def _watch_ops(events, *, pre_delay=0.0, error=None):
    """A minimal ClusterOps stand-in whose watch yields a scripted sequence,
    optionally after an idle gap (``pre_delay``) and/or ending in ``error``."""

    class FakeWatchOps:
        def __init__(self):
            self.cluster_id = 1
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

        async def watch_instance_types(self, resource_version=None):
            if pre_delay:
                await asyncio.sleep(pre_delay)
            for evt in events:
                yield evt
            if error is not None:
                raise error

    return FakeWatchOps()


async def _collect(agen):
    return [frame async for frame in agen]


@pytest.mark.asyncio
async def test_watch_stream_maps_verbs_and_drops_bookmark():
    ops = _watch_ops(
        [
            _watch_evt("ADDED", "it-a"),
            {"type": "BOOKMARK", "raw_object": {"metadata": {"resourceVersion": "42"}}},
            _watch_evt("MODIFIED", "it-a"),
            _watch_evt("DELETED", "it-a", phase="Terminating"),
        ]
    )

    frames = await _collect(_cluster_stream(ops))
    payloads = [json.loads(f) for f in frames if f != "\n\n"]

    # ADDED→1, MODIFIED→2, DELETED→3; BOOKMARK produces no frame.
    assert [p["type"] for p in payloads] == [1, 2, 3]
    assert all(p["data"]["name"] == "it-a" for p in payloads)
    # The DELETED frame carries the object's pre-deletion state.
    assert payloads[2]["data"]["status"]["phase"] == "Terminating"
    assert ops.closed  # client released on stream teardown


@pytest.mark.asyncio
async def test_watch_stream_emits_heartbeat_when_idle(monkeypatch):
    monkeypatch.setattr(helper, "_HEARTBEAT_INTERVAL", 0.01)
    ops = _watch_ops([_watch_evt("ADDED", "it-a")], pre_delay=0.05)

    frames = await _collect(_cluster_stream(ops))

    assert "\n\n" in frames  # ≥1 keepalive during the idle gap
    data_frames = [f for f in frames if f != "\n\n"]
    assert len(data_frames) == 1
    assert json.loads(data_frames[0])["type"] == 1


@pytest.mark.asyncio
async def test_watch_stream_absorbs_error_without_error_frame():
    # A watch ERROR surfaces as an ApiException; it must end the stream, never
    # become a data frame.
    ops = _watch_ops(
        [_watch_evt("ADDED", "it-a")],
        error=client.exceptions.ApiException(status=500, reason="boom"),
    )

    frames = await _collect(_cluster_stream(ops))
    payloads = [json.loads(f) for f in frames if f != "\n\n"]

    # Only the CREATED frame; the watch error ends the stream, never a frame.
    assert [p["type"] for p in payloads] == [1]
    assert ops.closed


def _watch_ops_unbounded():
    """A ClusterOps stand-in whose watch emits without ever awaiting or ending —
    it fills the bounded queue and parks the producer on ``put`` so the
    cancellation/teardown path can be exercised against a full queue."""

    class FakeUnboundedWatchOps:
        def __init__(self):
            self.cluster_id = 1
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

        async def watch_instance_types(self, resource_version=None):
            i = 0
            while True:
                yield _watch_evt("ADDED", f"it-{i}")
                i += 1

    return FakeUnboundedWatchOps()


@pytest.mark.asyncio
async def test_watch_stream_cancellation_does_not_deadlock_on_full_queue(monkeypatch):
    # Regression: a slow client leaves the producer parked on a full queue. On
    # disconnect the consumer cancels the producer; if the producer's teardown
    # enqueued _DONE onto the still-full queue it would block forever, deadlocking
    # the consumer awaiting it. Cancellation must instead end promptly and still
    # release the client.
    monkeypatch.setattr(helper, "_WATCH_QUEUE_MAXSIZE", 1)
    ops = _watch_ops_unbounded()

    agen = _cluster_stream(ops)
    # Pull one frame so the producer is running; then let it refill the size-1
    # queue and block on the next put while the consumer is idle at the yield.
    first = await agen.__anext__()
    assert json.loads(first)["type"] == 1
    await asyncio.sleep(0.05)

    # Client disconnect: closing the generator must not hang. A deadlocked
    # teardown is only unblocked when wait_for cancels aclose at the timeout
    # (the consumer suppresses that cancel, so aclose still returns) — so assert
    # on elapsed time, not just completion.
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.wait_for(agen.aclose(), timeout=1.0)
    assert loop.time() - start < 0.5, "stream teardown deadlocked on a full queue"
    assert ops.closed  # client released even on the cancellation path


def test_routes_registered():
    by_path = {}
    for r in it_routes.router.routes:
        methods = getattr(r, "methods", None)
        if methods:
            by_path.setdefault(r.path, set()).update(methods)

    assert "GET" in by_path["/aggregated"]
    assert "GET" in by_path[""]
    assert "POST" in by_path[""]
    assert "PUT" in by_path[""]
    assert "DELETE" in by_path["/{name}"]
    assert "PUT" in by_path["/{name}/deactivate"]
    assert "PUT" in by_path["/{name}/activate"]
