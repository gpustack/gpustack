"""Every GPU Service list route searches the name it displays.

All of them render ``displayName || name`` in the Name column and all of them
send ``search=``, but each narrowed on ``name`` alone — so a display name was
unfindable by the very string the list shows (#6104, first fixed for instance
types, where the display name lives inside a JSON column).

Here it is a real column, so it simply joins ``fuzzy_fields`` — which each route
already hands to ``paginated_by_query`` *and* ``streaming``, so the read and the
watch stream are covered together and no per-route predicate is needed.

Parametrized over the family rather than split per module on purpose: the claim
under test is that the family is aligned. ``ROUTES`` is hand-maintained, so
``test_every_gpu_service_list_route_is_covered`` derives the family from the
route modules themselves and fails when one is added without being listed here —
otherwise a sixth route would be silently uncovered rather than caught.
"""

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.responses import StreamingResponse

from gpustack.routes import gpu_instance_persistent_volume_types as pvt_routes
from gpustack.routes import gpu_instance_persistent_volumes as pv_routes
from gpustack.routes import gpu_instance_ssh_public_keys as key_routes
from gpustack.routes import gpu_instance_templates as template_routes
from gpustack.routes import gpu_instances as instance_routes
from gpustack.routes.gpu_instances_helper import (
    display_name_label,
    order_by_display_label,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeListParams,
    GPUInstancePersistentVolumeTypeSpec,
)
from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolume,
    GPUInstancePersistentVolumeListParams,
    GPUInstancePersistentVolumeSpec,
)
from gpustack.schemas.gpu_instance_ssh_public_keys import (
    GPUInstanceSSHPublicKey,
    GPUInstanceSSHPublicKeyListParams,
    GPUInstanceSSHPublicKeySpec,
)
from gpustack.schemas.gpu_instance_templates import (
    GPUInstanceSpecTemplate,
    GPUInstanceTemplate,
    GPUInstanceTemplateListParams,
)
from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstanceListParams,
    GPUInstanceSpec,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import Event, EventType

# SYSTEM principal → bypasses every tenant filter, so each handler reduces to the
# narrowing under test rather than to its own visibility rules.
CTX = SimpleNamespace(
    user=SimpleNamespace(kind=PrincipalType.SYSTEM, id=1),
    is_platform_admin=True,
    current_principal_id=None,
    scoped_cluster_id=None,
)

# A user-supplied display name and an opaque generated name: the pair a user sees
# in the table versus the one the row is keyed by. Searched with a differently
# cased substring, so the match is case-insensitive on the display-name arm too.
TERM = "team a"
SHOWN = "Team A Pool"
OPAQUE = "res-7f3a91"

SEEDED = (
    (OPAQUE, SHOWN),  # findable only by what the table shows
    ("res-000002", None),  # no display name at all
    ("res-000003", ""),  # empty display name
    ("res-000004", "Team B Pool"),  # a display name that does not match
)


def _instance(name, display_name):
    return GPUInstance(
        owner_principal_id=1,
        cluster_id=1,
        name=name,
        display_name=display_name,
        spec=GPUInstanceSpec(type_="gpu", image="busybox"),
    )


def _template(name, display_name):
    return GPUInstanceTemplate(
        owner_principal_id=1,
        name=name,
        display_name=display_name,
        spec=GPUInstanceSpecTemplate(image="busybox"),
    )


def _volume(name, display_name):
    return GPUInstancePersistentVolume(
        owner_principal_id=1,
        name=name,
        display_name=display_name,
        persistent_volume_type_id=1,
        spec=GPUInstancePersistentVolumeSpec(type_="nfs"),
    )


def _volume_type(name, display_name):
    return GPUInstancePersistentVolumeType(
        owner_principal_id=1,
        name=name,
        display_name=display_name,
        spec=GPUInstancePersistentVolumeTypeSpec(),
    )


def _public_key(name, display_name):
    return GPUInstanceSSHPublicKey(
        owner_principal_id=1,
        name=name,
        display_name=display_name,
        spec=GPUInstanceSSHPublicKeySpec(data="ssh-ed25519 AAAA"),
    )


ROUTES = [
    pytest.param(
        instance_routes,
        instance_routes.get_gpu_instances,
        GPUInstance,
        GPUInstanceListParams,
        _instance,
        id="gpu_instances",
    ),
    pytest.param(
        template_routes,
        template_routes.get_gpu_instance_templates,
        GPUInstanceTemplate,
        GPUInstanceTemplateListParams,
        _template,
        id="gpu_instance_templates",
    ),
    pytest.param(
        pv_routes,
        pv_routes.get_gpu_instance_persistent_volumes,
        GPUInstancePersistentVolume,
        GPUInstancePersistentVolumeListParams,
        _volume,
        id="gpu_instance_persistent_volumes",
    ),
    pytest.param(
        pvt_routes,
        pvt_routes.get_gpu_instance_persistent_volume_types,
        GPUInstancePersistentVolumeType,
        GPUInstancePersistentVolumeTypeListParams,
        _volume_type,
        id="gpu_instance_persistent_volume_types",
    ),
    pytest.param(
        key_routes,
        key_routes.get_gpu_instance_ssh_public_keys,
        GPUInstanceSSHPublicKey,
        GPUInstanceSSHPublicKeyListParams,
        _public_key,
        id="gpu_instance_ssh_public_keys",
    ),
]

FAMILY = "module,handler,model,params_cls,build"

# Templates are excluded from the sort family: the page is a card view with no
# column sorter, so it has no displayed ordering to align — its ``sort_by=name``
# keeps meaning ``name``.
SORTABLE_ROUTES = [p for p in ROUTES if p.id != "gpu_instance_templates"]

# All labels lowercase on purpose: the claim under test is WHICH value is
# ordered. An uppercase label would instead be testing collation, which differs
# between this fixture engine (binary) and production PostgreSQL (en_US.utf8).
# By ``name`` these sort res-aaa, res-bbb, res-nnn, res-zzz; by label they sort
# "mmm pool", "nnn pool", res-aaa, res-bbb.
SORT_SEEDED = (
    ("res-zzz", "mmm pool"),
    ("res-aaa", None),  # no display name -> falls back to name
    ("res-bbb", ""),  # empty display name -> the UI's falsy ``||``
    ("res-nnn", "nnn pool"),
)
SORT_BY_LABEL = ["res-zzz", "res-nnn", "res-aaa", "res-bbb"]


@pytest_asyncio.fixture
async def engine():
    """Every table, not just the one under test: some handlers enrich their page
    from a sibling table (a PV reads its attachments), and the point here is the
    narrowing, not each route's incidental queries."""
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield e
    await e.dispose()


async def _seed(engine, *rows):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        for row in rows:
            session.add(row)
        await session.commit()


def _patch_db(monkeypatch, module, engine):
    monkeypatch.setattr(
        module, "async_session", lambda: AsyncSession(engine, expire_on_commit=False)
    )


def _patch_streaming(monkeypatch, model):
    capture = {}

    def fake_streaming(**kwargs):
        capture.update(kwargs)

        async def _events():
            for frame in ():
                yield frame

        return _events()

    monkeypatch.setattr(model, "streaming", fake_streaming)
    return capture


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, ROUTES)
async def test_search_matches_the_display_name(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    await _seed(engine, *(build(*row) for row in SEEDED))
    _patch_db(monkeypatch, module, engine)

    page = await handler(CTX, params_cls(), search=TERM)

    assert [i.name for i in page.items] == [OPAQUE]
    # ``total`` is asserted alongside: a count that disagrees with the items is
    # how this same search box reports "no results" over a page that has some.
    assert page.pagination.total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, ROUTES)
async def test_search_still_matches_the_name(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    # The arm that already worked, kept as a control: a row with no display name
    # at all must not fall out because the predicate grew a second arm. Its own
    # term, because a k8s object name and a human label share no substring.
    await _seed(engine, *(build(*row) for row in SEEDED))
    _patch_db(monkeypatch, module, engine)

    page = await handler(CTX, params_cls(), search="000002")

    assert [i.name for i in page.items] == ["res-000002"]
    assert page.pagination.total == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, ROUTES)
async def test_search_matching_neither_name_returns_an_empty_page(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    # The display-name arm must widen the match, not defeat it.
    await _seed(engine, build(OPAQUE, SHOWN), build("res-000002", None))
    _patch_db(monkeypatch, module, engine)

    page = await handler(CTX, params_cls(), search="l40s")

    assert page.items == []
    assert (page.pagination.total, page.pagination.totalPage) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, SORTABLE_ROUTES)
async def test_sort_by_name_orders_by_the_displayed_label(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    # ``sorter: true`` on the Name column sends ``sort_by=name``, so ordering by
    # the row's ``name`` sorted the column by a value it does not display.
    await _seed(engine, *(build(*row) for row in SORT_SEEDED))
    _patch_db(monkeypatch, module, engine)

    ascending = await handler(CTX, params_cls(sort_by="name"))
    descending = await handler(CTX, params_cls(sort_by="-name"))

    assert [i.name for i in ascending.items] == SORT_BY_LABEL
    assert [i.name for i in descending.items] == list(reversed(SORT_BY_LABEL))


def test_only_the_name_entry_is_translated():
    # Unit-level twin of the test above, over the shared translation itself: the
    # other sortable fields are real columns and must reach
    # ``paginated_by_query`` as the plain names it resolves by attribute lookup.
    label = display_name_label(GPUInstance)
    translated = order_by_display_label(
        [("name", "asc"), ("cluster_id", "desc"), ("created_at", "asc")], label
    )

    # The label is expanded, not substituted: ``name`` still follows it.
    assert translated[0] == (label, "asc")
    assert translated[1] == ("name", "asc")
    assert translated[2:4] == [("cluster_id", "desc"), ("created_at", "asc")]
    assert order_by_display_label(None, label) is None


def test_the_order_always_ends_on_a_unique_key():
    """Without it, LIMIT/OFFSET paging is not deterministic.

    Rows sharing a sort key have engine-defined relative order, which can differ
    between the page-1 and the page-2 query — one row comes back on both pages
    and another on neither. The label collides by construction (every cluster's
    collapsed generic pool is stamped ``CPU-only``) and ``name`` does not settle
    it: ``gpu_instance_types`` is unique on ``snapshot``, the others only per
    owner, while these lists span owners and clusters.
    """
    label = display_name_label(GPUInstance)

    assert order_by_display_label([("name", "asc")], label)[-1] == ("id", "asc")
    assert order_by_display_label([("name", "desc")], label)[-1] == ("id", "desc")
    # Not only the translated entry: any non-unique key has the same problem.
    assert order_by_display_label([("created_at", "desc")], label)[-1] == ("id", "desc")
    # Already asked for by the caller -> not appended twice.
    assert order_by_display_label([("id", "asc")], label) == [("id", "asc")]


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, SORTABLE_ROUTES)
async def test_rows_sharing_a_label_page_deterministically(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    # Walk the list one row at a time: every row must appear exactly once across
    # the pages, even though all four share the label the column sorts on.
    await _seed(engine, *(build(f"res-{i}", "Team A Pool") for i in range(4)))
    _patch_db(monkeypatch, module, engine)

    seen = []
    for page in range(1, 5):
        result = await handler(CTX, params_cls(page=page, perPage=1, sort_by="name"))
        seen += [i.name for i in result.items]

    assert sorted(seen) == ["res-0", "res-1", "res-2", "res-3"]


def test_every_gpu_service_list_route_is_covered():
    """The family is derived, not trusted to a hand-maintained list.

    A new GPU Service list route that takes ``search`` has to be added to
    ``ROUTES`` (or, like instance types, be covered by its own module) — without
    this it would simply go untested, which is exactly how the divergence this
    module guards against got in.
    """
    covered = {p.values[0].__name__ for p in ROUTES} | {
        # Instance types are exercised in tests/routes/test_gpu_instance_types.py,
        # which has the cluster-visibility scaffolding this module does not.
        "gpustack.routes.gpu_instance_types"
    }

    family = set()
    for path in sorted(Path(instance_routes.__file__).parent.glob("gpu_instance*.py")):
        module = importlib.import_module(f"gpustack.routes.{path.stem}")
        for name, obj in vars(module).items():
            if not (name.startswith("get_gpu") and inspect.iscoroutinefunction(obj)):
                continue
            if "search" in inspect.signature(obj).parameters:
                family.add(module.__name__)

    assert (
        family <= covered
    ), f"GPU Service list routes with no coverage: {family - covered}"


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, SORTABLE_ROUTES)
async def test_the_other_sortable_fields_are_untouched(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    # Only the Name column renders a label; the rest are real columns and must
    # reach ``paginated_by_query`` as the plain names it resolves by attribute
    # lookup.
    await _seed(engine, *(build(*row) for row in SORT_SEEDED))
    _patch_db(monkeypatch, module, engine)

    page = await handler(CTX, params_cls(sort_by="id"))

    assert [i.name for i in page.items] == [name for name, _ in SORT_SEEDED]


@pytest.mark.asyncio
@pytest.mark.parametrize(FAMILY, ROUTES)
async def test_the_watch_stream_narrows_on_both_names(
    monkeypatch, engine, module, handler, model, params_cls, build
):
    """Without this a searched page keeps its rows but stops being told about
    their changes, which reads as a stale table rather than a missing filter."""
    _patch_db(monkeypatch, module, engine)
    capture = _patch_streaming(monkeypatch, model)

    resp = await handler(CTX, params_cls(watch=True), search=TERM)

    assert isinstance(resp, StreamingResponse)
    assert capture["fuzzy_fields"] == {"name": TERM, "display_name": TERM}
    # ``_match_fuzzy_fields`` ORs the keys, so the display-name arm is what
    # carries a row the name arm cannot match.
    shown_only = Event(type=EventType.UPDATED, data=build(OPAQUE, SHOWN))
    unrelated = Event(type=EventType.UPDATED, data=build("res-000002", "Team B Pool"))
    assert model._match_fuzzy_fields(shown_only, capture["fuzzy_fields"]) is True
    assert model._match_fuzzy_fields(unrelated, capture["fuzzy_fields"]) is False
