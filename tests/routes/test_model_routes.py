import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import true

from gpustack.api.exceptions import InvalidException
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_routes
from gpustack.routes.model_common import (
    ModelStateFilterEnum,
    categories_filter,
    state_stream_filter,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteCreate,
    ModelRouteListParams,
    ModelRoutePublic,
    ModelRoutesPublic,
    ModelRouteTarget,
    ModelRouteTargetUpdateItem,
    MyModel,
    TargetStateEnum,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.principals import Principal, PrincipalType
import gpustack.server.lora_adapters_discovery as discovery


def _ctx(
    user_id: int, current_principal_id: int, is_admin: bool = False
) -> TenantContext:
    user = Principal(
        id=user_id,
        name=f"u{user_id}",
        kind=PrincipalType.USER,
        is_admin=is_admin,
    )
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=current_principal_id,
        org_role=None,
        current_is_personal_scope=current_principal_id == user_id,
    )


def _compile(expr) -> str:
    """Render a SQLAlchemy expression to a literal string for assertions."""
    return str(expr.compile(compile_kwargs={"literal_binds": True}))


@pytest.mark.asyncio
async def test_get_model_routes_filters_categories_on_target_class(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    def fake_build_category_conditions(session, target_class, categories):
        captured["target_class"] = target_class
        captured["categories"] = categories
        return [true()]

    async def fake_paginated_by_query(**kwargs):
        captured["fields"] = kwargs["fields"]
        captured["extra_conditions"] = kwargs["extra_conditions"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(
        model_routes, "build_category_conditions", fake_build_category_conditions
    )
    monkeypatch.setattr(MyModel, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        categories=["image"],
        target_class=MyModel,
        user_id=123,
    )

    assert captured["target_class"] is MyModel
    assert captured["categories"] == ["image"]
    assert captured["fields"]["user_id"] == 123
    assert captured["extra_conditions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target_class", [MyModel, ModelRoute])
@pytest.mark.parametrize(
    "state, expected_fragments",
    [
        (ModelStateFilterEnum.READY, ["ready_targets > 0"]),
        (ModelStateFilterEnum.NOT_READY, ["ready_targets = 0", "targets > 0"]),
        (ModelStateFilterEnum.STOPPED, ["targets = 0"]),
    ],
)
async def test_get_model_routes_filters_state_on_target_class(
    monkeypatch, target_class, state, expected_fragments
):
    """The ``state`` filter mirrors Model readiness against a route's
    target counts, on both the MyModel view and the ModelRoute table."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_paginated_by_query(**kwargs):
        captured["extra_conditions"] = kwargs["extra_conditions"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(target_class, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        state=state,
        target_class=target_class,
    )

    rendered = " ".join(_compile(c) for c in captured["extra_conditions"])
    for fragment in expected_fragments:
        assert fragment in rendered


@pytest.mark.asyncio
async def test_get_model_routes_without_state_adds_no_readiness_condition(monkeypatch):
    """Omitting ``state`` leaves the query unconstrained by target counts."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_paginated_by_query(**kwargs):
        captured["extra_conditions"] = kwargs["extra_conditions"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(MyModel, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        target_class=MyModel,
        user_id=123,
    )

    rendered = " ".join(_compile(c) for c in captured["extra_conditions"])
    assert "ready_targets" not in rendered
    assert "targets >" not in rendered
    assert "targets =" not in rendered


@pytest.mark.parametrize(
    "ready, total, state, expected",
    [
        (2, 3, ModelStateFilterEnum.READY, True),
        (0, 3, ModelStateFilterEnum.READY, False),
        (0, 3, ModelStateFilterEnum.NOT_READY, True),
        (2, 3, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.NOT_READY, False),
        (0, 0, ModelStateFilterEnum.STOPPED, True),
        (0, 3, ModelStateFilterEnum.STOPPED, False),
        (0, 3, None, True),
    ],
)
def test_state_stream_filter_matches_sql_semantics(ready, total, state, expected):
    """The watch-path predicate agrees with the SQL branch for routes."""
    data = SimpleNamespace(ready_targets=ready, targets=total)
    assert state_stream_filter(data, state, "ready_targets", "targets") is expected


def test_state_stream_filter_passes_id_only_delete_events():
    """ID-only DELETED payloads carry no target counts; the filter must let
    them through so watch clients drop the row instead of going stale."""
    assert (
        state_stream_filter(
            {"id": 9}, ModelStateFilterEnum.READY, "ready_targets", "targets"
        )
        is True
    )


def test_categories_filter_matches_and_passes_id_only_events():
    """Categories match against ``data.categories``; ID-only payloads that
    lack the attribute pass through rather than erroring the stream."""
    assert categories_filter(SimpleNamespace(categories=["image"]), ["image"]) is True
    assert categories_filter(SimpleNamespace(categories=["text"]), ["image"]) is False
    assert categories_filter(SimpleNamespace(categories=None), [""]) is True
    # No categories requested is always a match, regardless of payload shape.
    assert categories_filter({"id": 9}, None) is True
    # ID-only DELETED payload with an active category filter must not raise.
    assert categories_filter({"id": 9}, ["image"]) is True


@pytest.mark.asyncio
async def test_get_model_routes_stream_applies_state_filter(monkeypatch):
    """The watch/streaming branch honors ``state`` too, not just the
    paginated query."""
    captured = {}

    def fake_streaming(**kwargs):
        captured["filter_func"] = kwargs["filter_func"]

        async def _gen():
            return
            yield

        return _gen()

    monkeypatch.setattr(MyModel, "streaming", fake_streaming)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24, watch=True),
        state=ModelStateFilterEnum.READY,
        target_class=MyModel,
    )

    filter_func = captured["filter_func"]
    assert filter_func(SimpleNamespace(ready_targets=1, targets=1)) is True
    assert filter_func(SimpleNamespace(ready_targets=0, targets=1)) is False


@pytest.mark.asyncio
async def test_apply_effective_name_to_my_models_prefixes_by_owner(monkeypatch):
    """Items get rewritten in place: platform Org's route stays bare,
    non-platform Orgs prefix with the owner's ``name``. This is what
    lets the My Models page render and Open-in-Playground submit the
    OpenAI-style id even for cross-Org grants (granting Org isn't in
    the caller's client-side cache, but the server has it)."""
    monkeypatch.setattr(model_routes, "platform_principal_id", lambda: 1)

    items = [
        SimpleNamespace(name="qwen3-0.6b", owner_principal_id=1),
        SimpleNamespace(name="qwen3-0.6b", owner_principal_id=2),
        SimpleNamespace(name="bge-m3", owner_principal_id=3),
    ]

    session = SimpleNamespace()
    exec_result = MagicMock()
    exec_result.all.return_value = [(2, "alpha"), (3, "beta")]
    session.exec = AsyncMock(return_value=exec_result)

    await model_routes._apply_effective_name_to_my_models(session, items, enabled=True)

    assert items[0].name == "qwen3-0.6b"  # platform Org → no prefix
    assert items[1].name == "alpha/qwen3-0.6b"
    assert items[2].name == "beta/bge-m3"


@pytest.mark.asyncio
async def test_apply_effective_name_to_my_models_skips_when_disabled():
    """The rewrite is gated by ``enabled``, not the item's class —
    callers that don't opt in (the management list/get surfaces) leave
    ``name`` raw and avoid the owner-name lookup round-trip entirely."""
    session = SimpleNamespace(exec=AsyncMock())
    item = SimpleNamespace(name="qwen3-0.6b", owner_principal_id=2)

    await model_routes._apply_effective_name_to_my_models(
        session, [item], enabled=False
    )

    session.exec.assert_not_called()
    assert item.name == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_apply_effective_name_to_my_models_prefixes_admin_model_routes(
    monkeypatch,
):
    """When enabled, ``ModelRoute`` rows (the admin ``get_my_models``
    path) get prefixed just like the non-admin ``MyModel`` view, so a
    single endpoint returns one consistent ``name`` format for both."""
    monkeypatch.setattr(model_routes, "platform_principal_id", lambda: 1)

    item = ModelRoute(name="qwen3-0.6b", owner_principal_id=2)
    session = SimpleNamespace()
    exec_result = MagicMock()
    exec_result.all.return_value = [(2, "alpha")]
    session.exec = AsyncMock(return_value=exec_result)

    await model_routes._apply_effective_name_to_my_models(session, [item], enabled=True)

    assert item.name == "alpha/qwen3-0.6b"


def test_model_route_create_rejects_slashed_name():
    """The no-slash rule must still gate user input on the write path —
    otherwise a hand-typed ``foo/bar`` would collide with the
    ``<owner>/<name>`` shape the read path synthesizes."""
    with pytest.raises(ValueError, match="must start with a letter"):
        ModelRouteCreate(name="org1/qwen3-0.6b", targets=[])


def test_model_route_public_accepts_enriched_slashed_name():
    """Regression: the My Models response serializes through
    ``ModelRoutePublic``; once ``_apply_effective_name_to_my_models``
    rewrites ``name`` to ``<owner>/<name>``, response validation must
    accept it (the prior inherited validator rejected it with
    ``Unexpected error occurred: 1 validation error``)."""
    now = datetime(2026, 6, 7, tzinfo=timezone.utc)
    public = ModelRoutePublic.model_validate(
        {
            "id": 1,
            "name": "org1/qwen3-0.6b",
            "owner_principal_id": 6,
            "created_at": now,
            "updated_at": now,
        }
    )
    assert public.name == "org1/qwen3-0.6b"


@pytest.mark.asyncio
async def test_apply_effective_name_to_my_models_empty_is_noop():
    """Empty input must short-circuit before any session query — the
    helper runs on every MyModel list (including 0-result pages) and a
    spurious round-trip per page would be a regression."""
    session = SimpleNamespace(exec=AsyncMock())

    await model_routes._apply_effective_name_to_my_models(session, [], enabled=True)

    session.exec.assert_not_called()


def test_assert_target_tenant_aligned_same_org():
    # Same Org is always allowed.
    model_routes._assert_target_tenant_aligned(
        route_owner_principal_id=2,
        target_owner_principal_id=2,
        target_kind="Model",
        target_id=10,
    )


def test_assert_target_tenant_aligned_cross_org_rejected():
    """Routes are the only cross-tenant sharing surface; targeting a
    resource that lives in a different Org must fail at validation time
    rather than silently drift usage attribution to the wrong tenant."""
    with pytest.raises(InvalidException):
        model_routes._assert_target_tenant_aligned(
            route_owner_principal_id=2,
            target_owner_principal_id=3,
            target_kind="Model",
            target_id=10,
        )


def test_assert_target_tenant_aligned_skips_when_either_owner_null():
    # NULL on either side is treated as "global / legacy" — strict
    # equality only kicks in when both sides explicitly carry an owner.
    model_routes._assert_target_tenant_aligned(
        route_owner_principal_id=None,
        target_owner_principal_id=3,
        target_kind="Model",
        target_id=10,
    )
    model_routes._assert_target_tenant_aligned(
        route_owner_principal_id=2,
        target_owner_principal_id=None,
        target_kind="ModelProvider",
        target_id=11,
    )


# ----------------------------------------------------------------------
# MyModel visibility — partition by Personal vs Org act-as
# ----------------------------------------------------------------------


def test_my_model_visibility_sql_personal_scope():
    """Personal view: PUBLIC/AUTHED (NULL via) + USER/GROUP grants only —
    Org-mediated grants must be excluded."""
    ctx = _ctx(user_id=7, current_principal_id=7)
    sql = _compile(model_routes._my_model_visibility_sql(ctx))
    assert "via_principal_id IS NULL" in sql
    assert "via_principal_kind IN ('USER', 'GROUP')" in sql
    # No specific principal-id equality — this branch is kind-based.
    assert "via_principal_id =" not in sql


def test_my_model_visibility_sql_org_act_as():
    """Org act-as view: only grants tied to the active org (plus PUBLIC /
    AUTHED). User/Group-mediated grants don't bleed into the org view."""
    ctx = _ctx(user_id=7, current_principal_id=42)
    sql = _compile(model_routes._my_model_visibility_sql(ctx))
    assert "via_principal_id IS NULL" in sql
    assert "via_principal_id = 42" in sql
    assert "via_principal_kind" not in sql


def test_my_model_visibility_predicate_personal():
    """Mirror :func:`_my_model_visibility_sql` for the streaming path —
    Personal scope keeps PUBLIC/AUTHED and USER/GROUP rows, drops ORG."""
    pred = model_routes._my_model_visibility_predicate(_ctx(7, 7))
    assert pred(SimpleNamespace(via_principal_id=None, via_principal_kind=None))
    assert pred(SimpleNamespace(via_principal_id=7, via_principal_kind="USER"))
    assert pred(SimpleNamespace(via_principal_id=99, via_principal_kind="GROUP"))
    assert not pred(SimpleNamespace(via_principal_id=42, via_principal_kind="ORG"))


def test_my_model_visibility_predicate_org_act_as():
    pred = model_routes._my_model_visibility_predicate(_ctx(7, 42))
    assert pred(SimpleNamespace(via_principal_id=None, via_principal_kind=None))
    assert pred(SimpleNamespace(via_principal_id=42, via_principal_kind="ORG"))
    # Personal grants belong to the user's Personal view, not org 42's.
    assert not pred(SimpleNamespace(via_principal_id=7, via_principal_kind="USER"))
    # And another org's grant must not leak into this org's view.
    assert not pred(SimpleNamespace(via_principal_id=43, via_principal_kind="ORG"))


def test_my_model_visibility_returns_none_without_context():
    """No current_principal_id => no extra filter (defensive bypass)."""
    user = Principal(id=7, name="u7", kind=PrincipalType.USER, is_admin=True)
    ctx = TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    assert model_routes._my_model_visibility_sql(ctx) is None
    assert model_routes._my_model_visibility_predicate(ctx) is None


@pytest.mark.asyncio
async def test_get_model_routes_applies_my_model_visibility_filter(monkeypatch):
    """End-to-end through ``_get_model_routes``: the SQL OR predicate
    must show up in ``extra_conditions`` when target is MyModel + ctx
    has a current_principal_id."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_paginated_by_query(**kwargs):
        captured["extra_conditions"] = kwargs["extra_conditions"]
        captured["fields"] = kwargs["fields"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(MyModel, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        target_class=MyModel,
        user_id=7,
        ctx=_ctx(user_id=7, current_principal_id=42),
    )

    rendered = " ".join(_compile(c) for c in captured["extra_conditions"])
    assert "via_principal_id IS NULL" in rendered
    assert "via_principal_id = 42" in rendered
    # Fields filter for the user is still enforced.
    assert captured["fields"]["user_id"] == 7


# ----------------------------------------------------------------------
# Admin act-as parity — ``ModelRoute`` with ``include_grants=True``
# ----------------------------------------------------------------------


def test_model_route_grant_conditions_includes_owner_public_and_grants():
    """Admin act-as should see: owner-equality OR PUBLIC/AUTHED OR an
    explicit grant in ``model_route_principals`` — the same set a non-
    admin member of that org would see via the ``non_admin_user_models``
    view. Owner-only is too narrow."""
    ctx = _ctx(user_id=99, current_principal_id=42, is_admin=True)
    conds = model_routes._model_route_grant_conditions(ctx)
    assert len(conds) == 1
    rendered = _compile(conds[0])
    assert "access_policy IN ('PUBLIC', 'AUTHED')" in rendered
    assert "owner_principal_id = 42" in rendered
    # The EXISTS subquery links the route to model_route_principals.
    assert "model_route_principals" in rendered
    assert "principal_id = 42" in rendered


def test_model_route_grant_conditions_empty_without_principal():
    """All-mode admin (no current_principal_id) gets no filter — caller
    is expected to skip the helper or use it as a no-op."""
    user = Principal(id=99, name="admin", kind=PrincipalType.USER, is_admin=True)
    ctx = TenantContext(
        user=user,
        is_platform_admin=True,
        current_principal_id=None,
        org_role=None,
    )
    assert model_routes._model_route_grant_conditions(ctx) == []


@pytest.mark.asyncio
async def test_get_model_routes_admin_act_as_uses_grant_conditions(monkeypatch):
    """When the my-models admin branch calls ``_get_model_routes`` with
    ``include_grants=True``, the broader OR set replaces the narrow
    owner-equality filter and the field-level owner injection is
    skipped — otherwise the OR would be ANDed with owner=x and degrade
    to owner-only."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_paginated_by_query(**kwargs):
        captured["extra_conditions"] = kwargs["extra_conditions"]
        captured["fields"] = kwargs["fields"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    async def fake_fetch_granted_route_ids(ctx):
        return set()

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(ModelRoute, "paginated_by_query", fake_paginated_by_query)
    monkeypatch.setattr(
        model_routes, "_fetch_granted_route_ids", fake_fetch_granted_route_ids
    )

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        target_class=ModelRoute,
        ctx=_ctx(user_id=99, current_principal_id=42, is_admin=True),
        include_grants=True,
    )

    assert "owner_principal_id" not in captured["fields"]
    rendered = " ".join(_compile(c) for c in captured["extra_conditions"])
    assert "access_policy IN ('PUBLIC', 'AUTHED')" in rendered
    assert "owner_principal_id = 42" in rendered
    assert "model_route_principals" in rendered


@pytest.mark.asyncio
async def test_get_model_routes_admin_management_keeps_owner_narrow(monkeypatch):
    """``include_grants=False`` (the management path on
    ``/v1/model-routes``) keeps the historical owner-only filter — admins
    managing routes shouldn't be looking at routes they don't own just
    because someone granted them access."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_paginated_by_query(**kwargs):
        captured["fields"] = kwargs["fields"]
        captured["extra_conditions"] = kwargs["extra_conditions"]
        return ModelRoutesPublic(
            items=[],
            pagination=Pagination(page=1, perPage=24, total=0, totalPage=0),
        )

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(ModelRoute, "paginated_by_query", fake_paginated_by_query)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24),
        target_class=ModelRoute,
        ctx=_ctx(user_id=99, current_principal_id=42, is_admin=True),
    )

    # Owner gets pinned via the fields dict (used by both watch and
    # paginated paths), not via an OR predicate.
    assert captured["fields"].get("owner_principal_id") == 42
    rendered = " ".join(_compile(c) for c in captured["extra_conditions"])
    assert "model_route_principals" not in rendered


def test_model_route_grant_predicate_owner_public_and_grant():
    """Streaming mirror of the grant OR-set: PUBLIC/AUTHED or owner-match
    or a snapshotted grant id must all let the event through; everything
    else is dropped."""
    ctx = _ctx(user_id=99, current_principal_id=42, is_admin=True)
    pred = model_routes._model_route_grant_predicate(ctx, {17, 18})
    # PUBLIC / AUTHED: visible regardless of owner.
    assert pred(
        SimpleNamespace(
            id=1, access_policy=AccessPolicyEnum.PUBLIC, owner_principal_id=999
        )
    )
    assert pred(
        SimpleNamespace(
            id=2, access_policy=AccessPolicyEnum.AUTHED, owner_principal_id=999
        )
    )
    # Owner-equality on the current principal.
    assert pred(
        SimpleNamespace(
            id=3,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=42,
        )
    )
    # Snapshotted grant.
    assert pred(
        SimpleNamespace(
            id=17,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=999,
        )
    )
    # Foreign + private + not granted: dropped.
    assert not pred(
        SimpleNamespace(
            id=99,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=999,
        )
    )


@pytest.mark.asyncio
async def test_get_model_routes_admin_act_as_watch_applies_grant_filter(monkeypatch):
    """Streaming path for admin act-as my-models: the watch SSE filter
    must apply the same OR-set the paginated path uses — owner /
    PUBLIC/AUTHED / snapshotted grant — instead of emitting every
    ``ModelRoute`` globally."""
    captured = {}

    @asynccontextmanager
    async def fake_async_session():
        yield object()

    async def fake_fetch_granted_route_ids(ctx):
        return {17}

    def fake_streaming(fields=None, fuzzy_fields=None, filter_func=None):
        captured["fields"] = fields
        captured["filter_func"] = filter_func

        async def _empty():
            return
            yield ""  # pragma: no cover

        return _empty()

    monkeypatch.setattr(model_routes, "async_session", fake_async_session)
    monkeypatch.setattr(
        model_routes, "_fetch_granted_route_ids", fake_fetch_granted_route_ids
    )
    monkeypatch.setattr(ModelRoute, "streaming", fake_streaming)

    await model_routes._get_model_routes(
        params=ModelRouteListParams(page=1, perPage=24, watch=True),
        target_class=ModelRoute,
        ctx=_ctx(user_id=99, current_principal_id=42, is_admin=True),
        include_grants=True,
    )

    flt = captured["filter_func"]
    assert flt is not None
    # Cross-Org private route with no grant: dropped (the pre-fix bug
    # would have let this through).
    assert not flt(
        SimpleNamespace(
            id=5,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=999,
        )
    )
    # PUBLIC / owner / snapshotted grant: visible.
    assert flt(
        SimpleNamespace(
            id=6, access_policy=AccessPolicyEnum.PUBLIC, owner_principal_id=999
        )
    )
    assert flt(
        SimpleNamespace(
            id=7,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=42,
        )
    )
    assert flt(
        SimpleNamespace(
            id=17,
            access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS,
            owner_principal_id=999,
        )
    )
    # The narrow field-equality filter must NOT be applied — it would
    # AND with the OR predicate and drop PUBLIC/AUTHED + foreign-owned
    # grants entirely.
    assert "owner_principal_id" not in captured["fields"]


# ---------------------------------------------------------------------------
# list_adapters_for_base (backs GET /v2/models/adapters)
# ---------------------------------------------------------------------------


def _adapters_session(model_files):
    """A session whose exec(...).all() returns the given local ModelFile rows."""
    session = MagicMock()
    result = MagicMock()
    result.all = MagicMock(return_value=model_files)
    session.exec = AsyncMock(return_value=result)
    return session


def _local_lora(repo_id, base="qwen/qwen3-8b"):
    model_file = MagicMock()
    model_file.source = SourceEnum.HUGGING_FACE
    model_file.huggingface_repo_id = repo_id
    model_file.base_model = base
    return model_file


@pytest.mark.asyncio
async def test_list_adapters_slow_remote_degrades_to_local(monkeypatch):
    # A poor network stalls the remote calls; local results must not wait on them.
    monkeypatch.setattr(discovery, "REMOTE_ADAPTER_DISCOVERY_BUDGET", 0.2)

    async def slow_remote(*args, **kwargs):
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(discovery, "_cached_hf_adapters", slow_remote)
    monkeypatch.setattr(discovery, "_cached_ms_adapters", slow_remote)

    session = _adapters_session([_local_lora("org/my-lora")])

    start = time.monotonic()
    result = await discovery.list_adapters_for_base(session, "Qwen/Qwen3-8B", q="lora")
    elapsed = time.monotonic() - start

    assert elapsed < 2.0
    names = [item["lora_repo_name"] for item in result["lora_list"]]
    assert names == ["org/my-lora"]


@pytest.mark.asyncio
async def test_list_adapters_remote_exception_degrades_to_local(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(discovery, "_cached_hf_adapters", boom)
    monkeypatch.setattr(discovery, "_cached_ms_adapters", boom)

    session = _adapters_session([_local_lora("org/my-lora")])

    result = await discovery.list_adapters_for_base(session, "Qwen/Qwen3-8B", q="lora")

    names = [item["lora_repo_name"] for item in result["lora_list"]]
    assert names == ["org/my-lora"]


@pytest.mark.asyncio
async def test_list_adapters_merges_remote_with_local(monkeypatch):
    async def hf(*args, **kwargs):
        return [
            {
                "lora_repo_name": "remote/hf-lora",
                "source": SourceEnum.HUGGING_FACE.value,
                "is_local": False,
            }
        ]

    async def empty(*args, **kwargs):
        return []

    monkeypatch.setattr(discovery, "_cached_hf_adapters", hf)
    monkeypatch.setattr(discovery, "_cached_ms_adapters", empty)

    session = _adapters_session([_local_lora("org/my-lora")])

    result = await discovery.list_adapters_for_base(session, "Qwen/Qwen3-8B", q="lora")

    names = [item["lora_repo_name"] for item in result["lora_list"]]
    # Local first, then remote.
    assert names == ["org/my-lora", "remote/hf-lora"]


@pytest.mark.asyncio
async def test_list_adapters_no_query_skips_remote(monkeypatch):
    # No q: local only, remote must not be called.
    calls = {"hf": 0, "ms": 0}

    async def track_hf(*args, **kwargs):
        calls["hf"] += 1
        return []

    async def track_ms(*args, **kwargs):
        calls["ms"] += 1
        return []

    monkeypatch.setattr(discovery, "_cached_hf_adapters", track_hf)
    monkeypatch.setattr(discovery, "_cached_ms_adapters", track_ms)

    session = _adapters_session([_local_lora("org/my-lora")])

    result = await discovery.list_adapters_for_base(session, "Qwen/Qwen3-8B")

    names = [item["lora_repo_name"] for item in result["lora_list"]]
    assert names == ["org/my-lora"]
    assert calls == {"hf": 0, "ms": 0}


@pytest.mark.asyncio
async def test_list_local_loras_filters_by_q():
    # q filters local by lora_repo_name, case-insensitive.
    session = _adapters_session(
        [_local_lora("org/chat-lora"), _local_lora("org/math-lora")]
    )

    result = await discovery.list_local_loras(session, "Qwen/Qwen3-8B", q="CHAT")

    names = [item["lora_repo_name"] for item in result]
    assert names == ["org/chat-lora"]


def _capture_target_updates(monkeypatch):
    """Patch ModelRouteTarget.update to record the source dict of each
    persisted update without touching a database."""
    captured = []

    async def fake_update(self, session=None, source=None, auto_commit=False):
        captured.append(source)
        for key, value in (source or {}).items():
            setattr(self, key, value)
        return self

    monkeypatch.setattr(ModelRouteTarget, "update", fake_update)
    return captured


@pytest.mark.asyncio
async def test_update_targets_model_to_provider_clears_model_id(monkeypatch):
    # Switching a local-model target to a provider target must null model_id;
    # leaving both ids set persists a row the schema validator rejects.
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        model_id=5,
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    result = await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[
            ModelRouteTargetUpdateItem(
                id=1, provider_id=10, overridden_model_name="gpt-4", weight=1
            )
        ],
        existing_target_map={1: existing},
    )

    assert len(captured) == 1
    source = captured[0]
    assert source["model_id"] is None
    assert source["provider_id"] == 10
    # provider targets carry no readiness, so they go straight to ACTIVE.
    assert source["state"] == TargetStateEnum.ACTIVE
    assert result[0].model_id is None
    assert result[0].provider_id == 10


@pytest.mark.asyncio
async def test_update_targets_provider_to_model_clears_provider_id(monkeypatch):
    # The reverse switch must null provider_id and go UNAVAILABLE so the
    # controller re-validates against current model readiness.
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        provider_id=10,
        overridden_model_name="gpt-4",
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[
            ModelRouteTargetUpdateItem(
                id=1, model_id=5, overridden_model_name="base:lora", weight=1
            )
        ],
        existing_target_map={1: existing},
    )

    assert len(captured) == 1
    source = captured[0]
    assert source["provider_id"] is None
    assert source["model_id"] == 5
    assert source["state"] == TargetStateEnum.UNAVAILABLE


@pytest.mark.asyncio
async def test_update_targets_noop_does_not_write(monkeypatch):
    # Re-sending the same values must not trigger a spurious update/state reset.
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        provider_id=10,
        overridden_model_name="gpt-4",
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[
            ModelRouteTargetUpdateItem(
                id=1, provider_id=10, overridden_model_name="gpt-4", weight=1
            )
        ],
        existing_target_map={1: existing},
    )

    assert captured == []


@pytest.mark.asyncio
async def test_update_targets_clears_nulled_optional_fields(monkeypatch):
    # An explicit null clears optional fields (e.g. removing a LoRA override or
    # a fallback config) instead of being silently dropped by exclude_none.
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        model_id=5,
        overridden_model_name="base:lora",
        fallback_status_codes=["5xx"],
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[
            ModelRouteTargetUpdateItem(
                id=1,
                model_id=5,
                overridden_model_name=None,
                fallback_status_codes=None,
                weight=1,
            )
        ],
        existing_target_map={1: existing},
    )

    assert len(captured) == 1
    source = captured[0]
    assert source["overridden_model_name"] is None
    assert source["fallback_status_codes"] is None


@pytest.mark.asyncio
async def test_update_targets_switch_resets_type_coupled_overridden_name(monkeypatch):
    # provider -> model switch that omits overridden_model_name must NOT keep
    # the provider's name (which is invalid for a model target); it resets to
    # the item's value (None here).
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        provider_id=10,
        overridden_model_name="gpt-4",
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[ModelRouteTargetUpdateItem(id=1, model_id=5, weight=1)],
        existing_target_map={1: existing},
    )

    assert len(captured) == 1
    source = captured[0]
    assert source["model_id"] == 5
    assert source["provider_id"] is None
    assert source["overridden_model_name"] is None
    assert source["state"] == TargetStateEnum.UNAVAILABLE


@pytest.mark.asyncio
async def test_update_targets_keeps_omitted_fields(monkeypatch):
    # A field the client omits is left untouched (partial update); only the
    # provided field changes.
    captured = _capture_target_updates(monkeypatch)
    existing = ModelRouteTarget(
        id=1,
        name="r1-abcde",
        route_name="r1",
        route_id=1,
        model_id=5,
        overridden_model_name="base:lora",
        weight=1,
        state=TargetStateEnum.ACTIVE,
    )

    await model_routes.update_model_route_targets(
        session=MagicMock(),
        targets=[ModelRouteTargetUpdateItem(id=1, model_id=5, weight=9)],
        existing_target_map={1: existing},
    )

    assert len(captured) == 1
    source = captured[0]
    assert source["weight"] == 9
    # overridden_model_name was not sent, so it is preserved.
    assert source["overridden_model_name"] == "base:lora"
