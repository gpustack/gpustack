"""GPU instance-type route tests.

The list route reads the ``gpu_instance_types`` record table, so its tests run
the handler against a real in-memory sqlite table: the fuzzy name search and the
``deleted_at IS NULL`` filter are claims about the SQL ``paginated_by_query``
builds, which asserting on captured kwargs cannot verify.

The write routes still proxy into one cluster, so theirs call the handlers
directly with a fake ``ctx`` / ``request`` and a monkeypatched ``ClusterOps`` /
``Cluster.one_by_id`` — no live cluster.
"""

import asyncio
import inspect
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kubernetes_asyncio import client
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.routes import gpu_instance_types as it_routes
from gpustack.routes import gpu_instances_helper as helper
from gpustack.schemas.clusters import ClusterProvider, GpuInstanceOptions, K8sOptions
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeCreate,
    GPUInstanceTypeDetail,
    GPUInstanceTypeListParams,
    GPUInstanceTypeSpec,
    GPUInstanceTypeSpecUpdate,
    GPUInstanceTypeStatusPublic,
    GPUInstanceTypeUpdate,
    GPUInstanceTypesPublic,
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
    list_error=None,
    create_result=None,
    create_error=None,
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
            if list_error is not None:
                raise list_error
            return list_result

        async def create_instance_type(self, name, spec, ignore_existed=True):
            capture["name"] = name
            capture["spec"] = spec
            capture["ignore_existed"] = ignore_existed
            if create_error is not None:
                raise create_error
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


def _patch_ready_workers(monkeypatch, count=1):
    """Stub the route's reachability probe (``count_ready_workers``).

    A live read whose ops are faked simulates a contactable cluster, so the
    tests default to one READY worker; the worker-less test passes 0.
    """

    async def fake_count(cluster_id: int) -> int:
        return count

    monkeypatch.setattr(it_routes, "count_ready_workers", fake_count)


def _cluster(id_=1, owner_principal_id=None, gpu_service=True):
    """A cluster fixture, GPU Service by default.

    Every route in this module is a GPU Service route, so a cluster that can be
    used here is the norm; a Model Service cluster is the exception, and each
    test that exercises one passes ``gpu_service=False`` explicitly. The purpose
    signal is the presence of ``gpu_instance_options`` — see
    ``schemas/clusters.is_gpu_service_k8s_options``.
    """
    return SimpleNamespace(
        id=id_,
        name=f"cluster-{id_}",
        owner_principal_id=owner_principal_id,
        registration_token="tok",
        k8s_options=K8sOptions(
            gpu_instance_options=GpuInstanceOptions() if gpu_service else None
        ),
    )


#
# Table-backed list (GET "") tests.
#
# The list serves the ``gpu_instance_types`` record table, so these run the
# handler against a real in-memory sqlite table and stub only the visible-cluster
# query. Asserting on the kwargs handed to ``paginated_by_query`` would not show
# what matters here: which rows the SQL actually returns.
#


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


def _row(
    cluster_id=1,
    name="a10g",
    *,
    display_name=None,
    accelerator_group=None,
    phase=None,
    phase_message=None,
    detail=None,
    derived_from_node=False,
    deleted_at=None,
):
    """A record row as ``GPUInstanceTypeController`` would have projected it.

    ``snapshot`` is computed rather than faked because it is UNIQUE: two rows of
    the same ``(cluster_id, name, spec)`` collide here exactly as they would in
    production, which is why a retired row differs by a definitional field.
    """
    status = None
    if phase or phase_message or detail:
        status = GPUInstanceTypeStatusPublic(
            phase=phase, phase_message=phase_message, detail=detail
        )
    row = GPUInstanceType(
        cluster_id=cluster_id,
        name=name,
        spec=GPUInstanceTypeSpec(
            display_name=display_name, accelerator_group=accelerator_group
        ),
        status=status,
        derived_from_node=derived_from_node,
        deleted_at=deleted_at,
    )
    row.snapshot = row.compute_snapshot()
    return row


async def _seed(engine, *rows):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        for row in rows:
            session.add(row)
        await session.commit()


def _patch_db(monkeypatch, engine):
    """Point the route's session factory at the sqlite engine."""
    monkeypatch.setattr(
        it_routes, "async_session", lambda: AsyncSession(engine, expire_on_commit=False)
    )


def _patch_visible_clusters(monkeypatch, *clusters, capture=None):
    """Stub the query the route resolves its allowed cluster set from.

    Session-agnostic on purpose: the sqlite engine holds the instance-type table
    only, and what these tests exercise is what the route does with the set.
    """
    capture = capture if capture is not None else {}

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        capture["fields"] = fields
        capture["extra_conditions"] = extra_conditions
        # Honour the ``id`` narrowing the route pushes into the query, so a test
        # asking for one cluster still exercises what the route returns and not
        # merely the kwargs it passed. ``extra_conditions`` stays unmodelled:
        # visibility is asserted through ``capture``, since these tests hold no
        # clusters table to evaluate it against.
        wanted = (fields or {}).get("id")
        if wanted is None:
            return list(clusters)
        return [c for c in clusters if c.id == wanted]

    monkeypatch.setattr(it_routes.Cluster, "all_by_fields", fake_all)


def _patch_streaming(monkeypatch):
    """Capture the kwargs the watch path hands to ``GPUInstanceType.streaming``."""
    capture = {}

    def fake_streaming(**kwargs):
        capture.update(kwargs)

        async def _events():
            for frame in ():
                yield frame

        return _events()

    monkeypatch.setattr(it_routes.GPUInstanceType, "streaming", fake_streaming)
    return capture


async def _list(ctx=CTX, **kwargs):
    """Call the list handler, splitting the ``ListParams`` out of the filters."""
    params = GPUInstanceTypeListParams(
        **{
            key: kwargs.pop(key)
            for key in ("page", "perPage", "watch", "sort_by")
            if key in kwargs
        }
    )
    return await it_routes.get_gpu_instance_types(REQUEST, ctx, params, **kwargs)


@pytest.mark.asyncio
async def test_list_spans_every_visible_cluster(monkeypatch, engine):
    # The page opens without picking a cluster: one request returns the types of
    # every visible cluster, each row carrying the cluster that owns it.
    await _seed(
        engine,
        _row(1, "a10g"),
        _row(1, "h100"),
        _row(1, "cpu"),
        _row(2, "l40s"),
        _row(2, "a100"),
    )
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1), _cluster(2))

    out = await _list()

    assert {(i.cluster_id, i.name) for i in out.items} == {
        (1, "a10g"),
        (1, "h100"),
        (1, "cpu"),
        (2, "l40s"),
        (2, "a100"),
    }
    assert out.pagination.total == 5


@pytest.mark.asyncio
async def test_list_projects_a_row_into_the_public_model(monkeypatch, engine):
    # Replaces the CR→public mapping test: nothing maps ``metadata.name`` on the
    # read path any more, the row IS the projection. Validating the handler's
    # return through the response model is the path FastAPI takes, so it pins the
    # wire shape the frontend codes against — ``clusterId`` / ``derivedFromNode``
    # camelCase while ``id`` and the timestamps stay literal.
    await _seed(
        engine,
        _row(
            2,
            "a10g",
            display_name="A10G Pool",
            phase="Active",
            phase_message="ClusterQueue is admitting workloads",
            detail=GPUInstanceTypeDetail(manufacturer="nvidia", memory="24576Mi"),
            derived_from_node=True,
        ),
    )
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))

    out = await _list()

    served = GPUInstanceTypesPublic.model_validate(out.model_dump(by_alias=True))
    (item,) = served.items
    assert (item.cluster_id, item.name) == (2, "a10g")
    assert item.spec.display_name == "A10G Pool"
    assert item.status.phase == "Active"
    assert item.status.phase_message == "ClusterQueue is admitting workloads"
    assert item.status.detail.manufacturer == "nvidia"
    assert item.derived_from_node is True
    # The record table supplies the identity and timestamps the live CR cannot.
    assert item.id is not None
    assert item.created_at is not None and item.updated_at is not None
    # The resource ledger is not persisted: absent means "not served here",
    # never zero. Remaining capacity comes from /aggregated only.
    assert item.status.accelerator is None
    assert item.status.cpu is None

    wire = item.model_dump(by_alias=True, exclude_none=True)
    assert set(wire) == {
        "id",
        "clusterId",
        "name",
        "spec",
        "status",
        "derivedFromNode",
        "created_at",
        "updated_at",
    }
    assert wire["spec"]["displayName"] == "A10G Pool"


@pytest.mark.asyncio
async def test_list_reads_the_derived_from_node_marker(monkeypatch, engine):
    # Retargeted at the persisted column: the marker was read off the CR's labels
    # on every request and is now projected once by the controller.
    await _seed(
        engine,
        _row(1, "derived", derived_from_node=True),
        _row(1, "hand-made"),
    )
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    out = await _list()

    assert {i.name: i.derived_from_node for i in out.items} == {
        "derived": True,
        "hand-made": False,
    }


@pytest.mark.asyncio
async def test_cluster_id_narrows_the_list(monkeypatch, engine):
    # cluster_id is a filter now, not the scope: the same route serves one
    # cluster when asked and the whole visible fleet when not.
    await _seed(engine, _row(1, "a10g"), _row(2, "l40s"))
    _patch_db(monkeypatch, engine)
    capture = {}
    _patch_visible_clusters(monkeypatch, _cluster(1), _cluster(2), capture=capture)

    out = await _list(cluster_id=2)

    assert [(i.cluster_id, i.name) for i in out.items] == [(2, "l40s")]
    assert out.pagination.total == 1
    # Narrowed by the query, not by filtering its results: asking for one
    # cluster must not load every visible cluster to keep one.
    assert capture["fields"]["id"] == 2


@pytest.mark.asyncio
async def test_foreign_cluster_id_returns_an_empty_page_rather_than_403(
    monkeypatch, engine
):
    """A cluster_id outside the caller's visible set is answered with an empty
    page, never 403/404: a status that differs from an empty result is itself a
    probe for the existence of another tenant's cluster."""
    await _seed(engine, _row(9, "a10g"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    out = await _list(cluster_id=9)

    assert out.items == []
    assert (out.pagination.total, out.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
async def test_purpose_gpu_service_excludes_a_model_service_cluster(
    monkeypatch, engine
):
    # Naming the cluster explicitly must not get round the filter: purpose
    # narrows the allowed set, and cluster_id intersects with it.
    await _seed(engine, _row(1, "a10g"), _row(2, "l40s"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1), _cluster(2, gpu_service=False))

    everything = await _list(purpose="gpu_service")
    named = await _list(purpose="gpu_service", cluster_id=2)

    assert [i.name for i in everything.items] == ["a10g"]
    assert named.items == []


@pytest.mark.asyncio
async def test_purpose_model_service_returns_only_model_service_clusters(
    monkeypatch, engine
):
    await _seed(engine, _row(1, "a10g"), _row(2, "l40s"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1), _cluster(2, gpu_service=False))

    out = await _list(purpose="model_service")

    assert [i.name for i in out.items] == ["l40s"]


@pytest.mark.asyncio
async def test_search_matches_the_name_case_insensitively(monkeypatch, engine):
    await _seed(engine, _row(1, "A10G"), _row(1, "h100"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    out = await _list(search="a10")

    assert [i.name for i in out.items] == ["A10G"]
    # CAUTION: this passes here only because the fixture engine is SQLite, whose
    # LIKE is ASCII-case-insensitive. ``paginated_by_query`` lowers both sides for
    # the item query but neither side for the COUNT, so on PostgreSQL this same
    # request returns the item while reporting ``total == 0``. The assertion below
    # therefore does NOT protect production — it would keep passing straight
    # through that failure. It is kept as the tripwire that fires the first time
    # this suite is run against PostgreSQL; do not read it as coverage of the
    # count path. Recorded in the spec's Open Questions; the fix is one
    # ``func.lower`` on each side of the count predicate in a shared mixin, which
    # is outside this change's scope.
    assert out.pagination.total == 1


@pytest.mark.asyncio
async def test_search_does_not_match_the_spec_display_name(monkeypatch, engine):
    # Only real columns are fuzzy-filterable: ``paginated_by_query`` builds its
    # LIKE predicates with ``getattr(cls, key)``, and display_name lives inside
    # the ``spec`` JSON column.
    await _seed(engine, _row(1, "h100", display_name="A10G Pool"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    out = await _list(search="a10")

    assert out.items == []


@pytest.mark.asyncio
async def test_retired_rows_are_excluded(monkeypatch, engine):
    # A definition change retires the superseded row and inserts a new one, so
    # ``deleted_at IS NULL`` is what keeps one name from appearing twice.
    await _seed(
        engine,
        _row(1, "a10g", accelerator_group="nvidia-a10g"),
        _row(
            1,
            "a10g",
            accelerator_group="nvidia-a10g-superseded",
            deleted_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        ),
    )
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    out = await _list()

    assert [(i.name, i.spec.accelerator_group) for i in out.items] == [
        ("a10g", "nvidia-a10g")
    ]
    assert out.pagination.total == 1


@pytest.mark.asyncio
async def test_empty_visible_set_returns_an_empty_page_without_querying(
    monkeypatch, engine
):
    # An empty allowed set is answered directly. Letting it reach the query is the
    # shape of the tenant leak guarded against in the aggregated route: an empty
    # filter that reads as "everything".
    await _seed(engine, _row(1, "a10g"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch)

    async def boom(*args, **kwargs):
        raise AssertionError("must not query with an empty allowed-cluster set")

    monkeypatch.setattr(it_routes.GPUInstanceType, "paginated_by_query", boom)

    out = await _list()

    assert out.items == []
    assert (out.pagination.page, out.pagination.perPage) == (1, 100)
    assert (out.pagination.total, out.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
async def test_visible_clusters_come_from_the_cluster_visibility_filter(
    monkeypatch, engine
):
    # The table carries no owner_principal_id, so the authorization boundary is
    # the cluster set — resolved with the same filter the cluster list uses
    # (own-principal OR cluster_access grant), over Kubernetes clusters only:
    # no other provider has an InstanceType catalog to project.
    await _seed(engine, _row(1, "a10g"))
    _patch_db(monkeypatch, engine)
    capture = {}
    _patch_visible_clusters(monkeypatch, _cluster(1), capture=capture)

    await _list(ctx=CTX_NON_WRITER)

    assert capture["fields"] == {"provider": ClusterProvider.Kubernetes}
    assert [str(c) for c in capture["extra_conditions"]] == [
        str(c)
        for c in it_routes.cluster_visibility_conditions(
            CTX_NON_WRITER, it_routes.Cluster
        )
    ]


@pytest.mark.asyncio
async def test_sort_by_orders_the_list(monkeypatch, engine):
    await _seed(engine, _row(1, "b"), _row(1, "a"), _row(1, "c"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))

    ascending = await _list(sort_by="name")
    descending = await _list(sort_by="-name")

    assert [i.name for i in ascending.items] == ["a", "b", "c"]
    assert [i.name for i in descending.items] == ["c", "b", "a"]


#
# source=live tests: the per-cluster proxy read the model deploy form's slicing
# GPU type picker needs, because the record table does not persist the ledger.
#

# A live CR as the operator reports it, ledger included. The picker reads
# ``status.acceleratorSliced.onceMaxRequest`` to size and enable its sliced input
# and ``status.acceleratorPartitioned`` to fill its profile dropdown.
LIVE_ITEM = {
    "metadata": {
        "name": "a10g",
        "labels": {"schedule.gpustack.ai/derived-from-node": "true"},
    },
    "spec": {
        "acceleratable": True,
        "displayName": "A10G Pool",
        "acceleratorGroup": "nvidia-a10g",
    },
    "status": {
        "phase": "Active",
        "detail": {"manufacturer": "nvidia"},
        "accelerator": {"onceMaxRequest": "1", "remaining": "2", "capacity": "4"},
        "acceleratorSliced": {
            "onceMaxRequest": "4",
            "remaining": "8",
            "capacity": "16",
        },
        "acceleratorPartitioned": {
            "onceMaxRequest": "1",
            "remaining": "3",
            "capacity": "7",
            "remainingProfiles": [{"name": "1g.10gb", "count": 3}],
        },
        "cpu": {"onceMaxRequest": "4", "remaining": "8", "capacity": "16"},
    },
}


def _patch_no_cluster_contact(monkeypatch):
    """Fail the test if the route builds a client for any cluster.

    ``build_cluster_ops`` is resolved from this module's globals, so the route's
    own reference is what has to be replaced.
    """

    async def boom(*args, **kwargs):
        raise AssertionError("must not contact a cluster on this path")

    monkeypatch.setattr(it_routes, "build_cluster_ops", boom)


@pytest.mark.asyncio
async def test_source_live_returns_the_resource_ledger(monkeypatch, engine):
    # The record table deliberately does not persist the ledger, and the model
    # deploy picker sizes its inputs from it: a missing acceleratorSliced makes
    # onceMaxRequest read 0, which renders the sliced input disabled and makes its
    # validator reject every value, and a missing acceleratorPartitioned empties
    # the profile dropdown. So the live read has to carry both, under the
    # camelCase aliases the client actually reads.
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_ready_workers(monkeypatch)
    _patch_ops(monkeypatch, list_result={"items": [LIVE_ITEM]})

    out = await _list(cluster_id=2, source="live")

    (item,) = out.items
    assert item.name == "a10g"
    assert item.derived_from_node is True
    assert item.status.accelerator_sliced.once_max_request == "4"
    assert item.status.accelerator_partitioned.remaining_profiles[0].name == "1g.10gb"
    status = item.model_dump(by_alias=True, exclude_none=True)["status"]
    assert status["acceleratorSliced"]["onceMaxRequest"] == "4"
    assert status["acceleratorPartitioned"]["remainingProfiles"][0]["count"] == 3
    # One synthesized page: the cluster's list is not paginated, so it is served
    # whole rather than sliced.
    assert (out.pagination.page, out.pagination.total, out.pagination.totalPage) == (
        1,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_source_live_requires_a_cluster_id(monkeypatch):
    # A live read proxies into exactly one cluster's apiserver, so it has to name
    # one. Rejected before anything is resolved: the request is malformed, which
    # is not the same as a request whose answer is empty.
    _patch_no_cluster_contact(monkeypatch)

    # 400, not the project's 422 ``InvalidException``: the parameter combination is
    # malformed rather than a field failing validation, and F8 specifies 400.
    with pytest.raises(BadRequestException) as exc:
        await _list(source="live")
    assert exc.value.status_code == 400
    assert "cluster_id" in exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_kwargs", [{"name": "a10g"}, {"search": "a10"}])
async def test_source_live_rejects_the_record_only_filters(monkeypatch, filter_kwargs):
    # The upstream list takes no filters, so narrowing cannot be honoured. Refusing
    # loudly beats answering with everything: a filter that silently vanishes is
    # the same failure shape as a parameter read as something it is not. Rejected
    # before the cluster is contacted, so a malformed request costs no round trip.
    _patch_no_cluster_contact(monkeypatch)

    with pytest.raises(BadRequestException) as exc:
        await _list(cluster_id=2, source="live", **filter_kwargs)

    assert exc.value.status_code == 400
    assert "name and search" in exc.value.message


@pytest.mark.asyncio
async def test_source_live_does_not_apply_pagination_or_sort(monkeypatch, engine):
    # Pagination and sort differ in kind from the filters above: they are not
    # applied, and the response says so, because the synthesized block reports the
    # single page actually returned rather than the slice that was asked for. A
    # caller can therefore see that it got everything.
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_ready_workers(monkeypatch)
    _patch_ops(
        monkeypatch,
        list_result={
            "items": [
                {"metadata": {"name": "b"}, "spec": {}},
                {"metadata": {"name": "a"}, "spec": {}},
                {"metadata": {"name": "c"}, "spec": {}},
            ]
        },
    )

    out = await _list(cluster_id=2, source="live", page=2, perPage=1, sort_by="-name")

    # Upstream order, whole catalog: neither the page window nor the sort applied.
    assert [i.name for i in out.items] == ["b", "a", "c"]
    assert (
        out.pagination.page,
        out.pagination.perPage,
        out.pagination.total,
        out.pagination.totalPage,
    ) == (1, 3, 3, 1)


@pytest.mark.asyncio
async def test_source_live_empty_catalog_matches_the_record_empty_page(
    monkeypatch, engine
):
    # One shape for "nothing here" across both sources. A frontend that switches
    # source must not have to special-case the empty response.
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_ready_workers(monkeypatch)
    _patch_ops(monkeypatch, list_result={"items": []})

    live = await _list(cluster_id=2, source="live", page=2, perPage=5)

    # Same params against the record read with nothing visible — its empty page.
    _patch_visible_clusters(monkeypatch)
    record = await _list(page=2, perPage=5)

    assert live.items == record.items == []
    assert live.pagination == record.pagination
    assert (live.pagination.total, live.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
async def test_source_live_with_a_foreign_cluster_id_is_invisible(monkeypatch, engine):
    # Same answer the record read gives, for the same reason: a status that differs
    # from an empty result is a probe for another tenant's cluster. The cluster is
    # never contacted either — visibility is decided before any client is built.
    await _seed(engine, _row(9, "a10g"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))
    _patch_no_cluster_contact(monkeypatch)

    out = await _list(cluster_id=9, source="live")

    assert out.items == []
    assert (out.pagination.total, out.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
async def test_source_live_applies_purpose_like_the_record_read(monkeypatch, engine):
    # purpose is independent of source: it narrows the same allowed cluster set,
    # so a Model Service cluster is out of a gpu_service-filtered live read too.
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2, gpu_service=False))
    _patch_no_cluster_contact(monkeypatch)

    out = await _list(cluster_id=2, source="live", purpose="gpu_service")

    assert out.items == []


@pytest.mark.asyncio
async def test_source_live_with_no_ready_worker_returns_an_empty_page(
    monkeypatch, engine
):
    """A cluster with no READY worker has no reachable proxy — the read could
    only 503 — and no GPUs to type, so the empty page IS the answer (#6096),
    in the same shape the record read gives for "nothing here". The catalog is
    never listed: the guard answers before the cluster is contacted."""
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_ready_workers(monkeypatch, 0)
    _patch_ops(
        monkeypatch,
        list_error=client.exceptions.ApiException(
            status=503, reason="Service Unavailable"
        ),
    )

    out = await _list(cluster_id=2, source="live", page=2, perPage=5)

    assert out.items == []
    assert (out.pagination.total, out.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
async def test_source_live_with_ready_workers_still_surfaces_an_upstream_503(
    monkeypatch, engine
):
    """The empty-page guard (#6096) answers only a cluster with no READY
    worker. A reachable cluster's own failure still surfaces: a transient
    proxy 503 — e.g. the last worker dropping between the probe and the
    call — is reported as ServiceUnavailable rather than masked as an empty
    catalog, and never downgraded to the 500 that contradicted its own
    message (#6071)."""
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_ready_workers(monkeypatch, 1)
    _patch_ops(
        monkeypatch,
        list_error=client.exceptions.ApiException(
            status=503, reason="Service Unavailable"
        ),
    )

    with pytest.raises(ServiceUnavailableException) as exc:
        await _list(cluster_id=2, source="live")

    assert exc.value.status_code == 503
    assert "Service Unavailable" in exc.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, "record"], ids=["omitted", "record"])
async def test_a_cluster_id_alone_stays_table_backed(monkeypatch, engine, source):
    # The case an implicit switch on cluster_id would have broken: this is the
    # page's cluster filter, and it must keep reading the record table, or a
    # worker-less cluster would 5xx the page again (#6071).
    await _seed(engine, _row(2, "recorded"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_no_cluster_contact(monkeypatch)

    out = await _list(cluster_id=2, source=source)

    assert [i.name for i in out.items] == ["recorded"]


@pytest.mark.asyncio
async def test_watch_source_live_uses_the_cluster_watch(monkeypatch, engine):
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(2))
    _patch_cluster(monkeypatch, _cluster(2))
    _patch_watch_ops(
        monkeypatch,
        [
            {
                "type": "ADDED",
                "raw_object": {
                    "metadata": {"name": "it-a"},
                    "spec": {"acceleratable": True},
                    "status": LIVE_ITEM["status"],
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

    def boom(**kwargs):
        raise AssertionError("source=live must not stream from the bus")

    monkeypatch.setattr(it_routes.GPUInstanceType, "streaming", boom)

    resp = await _list(watch=True, source="live", cluster_id=2)

    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    frames = [frame async for frame in resp.body_iterator]
    payloads = [json.loads(f) for f in frames if f != "\n\n"]
    # ADDED→1, BOOKMARK dropped, DELETED→3.
    assert [p["type"] for p in payloads] == [1, 3]
    assert payloads[0]["data"]["name"] == "it-a"
    # The ledger rides the live watch too — the picker watches while it is open.
    assert payloads[0]["data"]["status"]["acceleratorSliced"]["onceMaxRequest"] == "4"


@pytest.mark.asyncio
async def test_watch_source_live_with_a_foreign_cluster_id_streams_nothing(
    monkeypatch, engine
):
    # An unreadable cluster leaves nothing to watch, so this falls through to the
    # record path's watch answer: an empty stream, never a 404, and the cluster is
    # not contacted.
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1))
    _patch_no_cluster_contact(monkeypatch)
    capture = _patch_streaming(monkeypatch)

    resp = await _list(watch=True, source="live", cluster_id=9)

    assert isinstance(resp, StreamingResponse)
    assert capture["filter_func"](_row(9, "a10g")) is False


def test_list_declares_the_sortable_params_class():
    # The declared params type is what makes FastAPI run ``validate_sort_by``, so
    # an unsortable field is rejected before it can reach ORDER BY. Which four
    # fields are allowed is locked down in tests/schemas.
    annotation = (
        inspect.signature(it_routes.get_gpu_instance_types)
        .parameters["params"]
        .annotation
    )
    assert annotation is GPUInstanceTypeListParams
    with pytest.raises(InvalidException, match="not sortable"):
        GPUInstanceTypeListParams(sort_by="bogus")


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
async def test_create_refuses_a_duplicate_name(monkeypatch):
    """#6087: a taken name must surface as an actionable 409, not a 200 that
    reads the pre-existing object back — so the create must run with
    ``ignore_existed=False`` and the upstream conflict must be reported
    naming the instance type."""
    _patch_cluster(monkeypatch, _cluster())
    capture = {}
    _patch_ops(
        monkeypatch,
        create_error=client.exceptions.ApiException(status=409, reason="Conflict"),
        capture=capture,
    )

    body = GPUInstanceTypeCreate(
        name="test",
        spec=GPUInstanceTypeSpec(acceleratable=False, os="linux"),
    )
    with pytest.raises(AlreadyExistsException) as excinfo:
        await it_routes.create_gpu_instance_type(REQUEST, None, CTX, body, 1)

    assert excinfo.value.status_code == 409
    assert "test" in excinfo.value.message
    assert "cluster-1" in excinfo.value.message
    assert capture["ignore_existed"] is False


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
        # Cluster 3 is a Model Service cluster: visible, Kubernetes, and still
        # dropped from the watch's cluster filter, so no event can be sourced
        # from it in the first place.
        return [_cluster(id_=1), _cluster(id_=2), _cluster(id_=3, gpu_service=False)]

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
    # Cluster ids forwarded to the gateway as strings, aggregated=True, with
    # the Model Service cluster absent from the filter.
    assert captured["clusters"] == ["1", "2"]
    assert captured["aggregated"] is True


@pytest.mark.asyncio
async def test_aggregated_excludes_model_service_clusters(monkeypatch):
    # A Model Service cluster is Kubernetes and visible, but its capacity is
    # committed to model deployment, so its instance types must not reach the
    # GPU Instance create form.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(it_routes, "async_session", lambda: _FakeSession())

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        return [_cluster(id_=1, gpu_service=False), _cluster(id_=2)]

    monkeypatch.setattr(it_routes.Cluster, "all_by_fields", fake_all)

    captured = {}

    async def fake_list(clusters=None, aggregated=False):
        captured["clusters"] = clusters
        return {"items": []}

    monkeypatch.setattr(it_routes.gateway_client, "list_instance_types", fake_list)

    await it_routes.get_gpu_aggregated_instance_types(CTX)

    assert captured["clusters"] == ["2"]


@pytest.mark.asyncio
async def test_aggregated_all_model_service_short_circuits(monkeypatch):
    # The sharper form of the empty-cluster guard: clusters ARE visible, they
    # are just all Model Service. Forwarding the resulting empty filter would
    # make the gateway return the whole fleet.
    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(it_routes, "async_session", lambda: _FakeSession())

    async def fake_all(session, fields=None, extra_conditions=None, **kw):
        return [
            _cluster(id_=1, gpu_service=False),
            _cluster(id_=2, gpu_service=False),
        ]

    monkeypatch.setattr(it_routes.Cluster, "all_by_fields", fake_all)

    async def boom(*a, **kw):
        raise AssertionError("gateway must not be called for zero GPU Service clusters")

    monkeypatch.setattr(it_routes.gateway_client, "list_instance_types", boom)
    monkeypatch.setattr(it_routes.gateway_client, "watch_instance_types", boom)

    out = await it_routes.get_gpu_aggregated_instance_types(CTX)
    assert out.items == []


@pytest.mark.asyncio
async def test_per_cluster_read_still_serves_a_model_service_cluster(
    monkeypatch, engine
):
    # The purpose guard is on writes only, and on the read ``purpose`` is opt-in,
    # so omitting it narrows nothing. This read is what the model deploy form's
    # Scheduling > Manual > Slicing GPU Type picker calls, and that picker targets
    # Model Service clusters by definition — narrowing this route to GPU Service
    # unconditionally would silently break it.
    await _seed(engine, _row(1, "it-a"))
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1, gpu_service=False))

    out = await _list(cluster_id=1)

    assert [i.name for i in out.items] == ["it-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: it_routes.create_gpu_instance_type(
                REQUEST,
                None,
                CTX,
                GPUInstanceTypeCreate(
                    name="new-it",
                    spec=GPUInstanceTypeSpec(acceleratable=True),
                ),
                1,
            ),
            id="create",
        ),
        pytest.param(
            lambda: it_routes.update_gpu_instance_type(
                REQUEST,
                None,
                CTX,
                GPUInstanceTypeUpdate(
                    name="it-a",
                    spec=GPUInstanceTypeSpecUpdate(display_name="x"),
                ),
                1,
            ),
            id="update",
        ),
        pytest.param(
            lambda: it_routes.delete_gpu_instance_type(REQUEST, None, CTX, "it-a", 1),
            id="delete",
        ),
        pytest.param(
            lambda: it_routes.deactivate_gpu_instance_type(
                REQUEST, None, CTX, "it-a", 1
            ),
            id="deactivate",
        ),
        pytest.param(
            lambda: it_routes.activate_gpu_instance_type(REQUEST, None, CTX, "it-a", 1),
            id="activate",
        ),
    ],
)
async def test_writes_to_a_model_service_cluster_raise_409(monkeypatch, call):
    _patch_cluster(monkeypatch, _cluster(gpu_service=False))

    async def boom(*a, **kw):
        raise AssertionError("must refuse before reaching the cluster")

    monkeypatch.setattr(helper, "build_cluster_ops", boom)

    with pytest.raises(ConflictException) as excinfo:
        await call()

    assert excinfo.value.status_code == 409
    # The refusal names the cluster and its purpose, so an API caller is not
    # left guessing why an otherwise-visible, otherwise-owned cluster refused.
    assert "cluster-1" in excinfo.value.message
    assert "model service" in excinfo.value.message


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
async def test_watch_streams_from_the_bus_under_the_same_narrowing(monkeypatch, engine):
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch, _cluster(1), _cluster(2))
    capture = _patch_streaming(monkeypatch)

    resp = await _list(watch=True, search="a10")

    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"
    assert capture["fields"] == {"deleted_at": None}
    assert capture["fuzzy_fields"] == {"name": "a10"}
    # Bus events never see the SQL extra_conditions, so visibility has to ride on
    # the filter_func or the stream leaks rows the REST read hides.
    visible = capture["filter_func"]
    assert visible(_row(1, "a10g")) is True
    assert visible(_row(3, "a10g")) is False
    # A DELETED event can carry an id-only dict (no cached object to enrich it
    # with). A bare attribute read would raise, and ``streaming`` swallows that
    # and silently ends the whole stream, so it must be handled, not trusted.
    assert visible({"id": 7}) is False


@pytest.mark.asyncio
async def test_watch_with_no_visible_cluster_streams_nothing(monkeypatch, engine):
    """No visible cluster answers ``watch=true`` with an empty stream — not a 404
    (a status that differs from an empty result is a probe) and not a JSON page
    (the caller asked for text/event-stream, and a body its reader cannot parse
    is the class of failure this route is being fixed for)."""
    _patch_db(monkeypatch, engine)
    _patch_visible_clusters(monkeypatch)
    capture = _patch_streaming(monkeypatch)

    async def boom(*args, **kwargs):
        raise AssertionError("the watch path must not query")

    monkeypatch.setattr(it_routes.GPUInstanceType, "paginated_by_query", boom)

    resp = await _list(watch=True)

    assert isinstance(resp, StreamingResponse)
    assert capture["filter_func"](_row(1, "a10g")) is False


#
# Per-cluster watch stream tests: the shared SSE helper fed by the per-cluster
# source, exactly as the ``source=live`` watch route composes it inline.
#


def _ops_stream(ops):
    """Rebuild the per-cluster watch stream the live watch route composes inline —
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

    frames = await _collect(_ops_stream(ops))
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

    frames = await _collect(_ops_stream(ops))

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

    frames = await _collect(_ops_stream(ops))
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

    agen = _ops_stream(ops)
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
