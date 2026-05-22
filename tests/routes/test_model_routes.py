from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import true

from gpustack.api.exceptions import InvalidException
from gpustack.api.tenant import TenantContext
from gpustack.routes import model_routes
from gpustack.schemas.common import Pagination
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteListParams,
    ModelRoutesPublic,
    MyModel,
)
from gpustack.schemas.principals import Principal, PrincipalType


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
            id=3, access_policy=AccessPolicyEnum.ALLOWED_USERS, owner_principal_id=42
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
            id=7, access_policy=AccessPolicyEnum.ALLOWED_USERS, owner_principal_id=42
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
