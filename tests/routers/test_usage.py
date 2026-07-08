from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ForbiddenException, InvalidException
from gpustack.routes.usage import (
    _self_scope_consumer_condition,
    get_usage_breakdown,
    get_usage_meta,
)
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User
from gpustack.schemas.usage import (
    UsageBreakdownRequest,
    UsageFilterItem,
    UsageFilterRequest,
    UsageIdentity,
    UsageIdentityValue,
    UsageSummary,
)


def _mock_exec_result(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


def _ctx_for(user):
    """Minimal TenantContext stub matching the route's read paths.

    Admins resolve with no current_principal_id (cross-Org "All" mode); regular
    users carry their default Org so they read their own usage rows. The
    route only touches ``is_platform_admin`` / ``current_principal_id`` /
    ``user``, so a MagicMock with those fields is enough."""
    ctx = MagicMock()
    ctx.user = user
    ctx.is_platform_admin = bool(getattr(user, "is_admin", False))
    ctx.current_principal_id = None if ctx.is_platform_admin else 1
    ctx.org_role = None
    ctx.current_is_personal_scope = False
    return ctx


def _org_owner_ctx(user, org_id=1):
    ctx = _ctx_for(user)
    ctx.current_principal_id = org_id
    ctx.org_role = OrgRole.OWNER
    return ctx


@pytest.mark.asyncio
async def test_get_usage_meta_returns_identity_filters_for_admin():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(group_user_name="alice", group_user_id=12),
                    SimpleNamespace(group_user_name="alice", group_user_id=None),
                ]
            ),
            # _existing_principal_ids: user 12 still exists (13 would be gone).
            _mock_exec_result([12]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_user_name="alice",
                        group_api_key_name="test",
                        group_access_key="abcd1234",
                        group_api_key_is_custom=False,
                        group_user_id=12,
                        group_api_key_id=34,
                    ),
                    SimpleNamespace(
                        group_user_name="alice",
                        group_api_key_name="custom",
                        group_access_key="hash1234",
                        group_api_key_is_custom=True,
                        group_user_id=12,
                        group_api_key_id=35,
                    ),
                ]
            ),
            # api_key existence: both 34 and 35 still exist.
            _mock_exec_result([34, 35]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_model_route_name="qwen-route",
                        group_model_route_id=21,
                    ),
                    SimpleNamespace(
                        group_model_route_name="legacy-route",
                        group_model_route_id=None,
                    ),
                    SimpleNamespace(
                        group_model_route_name=None,
                        group_model_route_id=None,
                    ),
                ]
            ),
            # route existence: route 21 still exists.
            _mock_exec_result([21]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)

    response = await get_usage_meta(session=session, user=user, ctx=_ctx_for(user))

    assert [item.key for item in response.group_bys] == [
        "date",
        "user",
        "api_key",
        "route",
    ]
    assert [item.key for item in response.metrics] == [
        "input_tokens",
        "output_tokens",
        "input_cached_tokens",
        "total_tokens",
        "api_requests",
    ]
    # user 12 still exists → not deleted; the NULL-id legacy row is deleted.
    assert response.filters.users[0].label == "alice"
    assert response.filters.users[0].deleted is False
    # Label is the pure name; deletion is carried by the ``deleted`` flag.
    assert response.filters.users[1].label == "alice"
    assert response.filters.users[1].deleted is True
    assert response.filters.api_keys[0].label == "alice / test"
    assert response.filters.api_keys[0].identity.value.access_key == "abcd1234"
    assert response.filters.api_keys[0].identity.value.api_key_is_custom is False
    assert response.filters.api_keys[0].identity.current.api_key_id == 34
    assert response.filters.api_keys[1].label == "alice / custom"
    assert response.filters.api_keys[1].identity.value.access_key == "hash1234"
    assert response.filters.api_keys[1].identity.value.api_key_is_custom is True
    assert response.filters.routes[0].label == "qwen-route"
    assert response.filters.routes[0].deleted is False
    assert response.filters.routes[0].identity.current.route_id == 21
    assert response.filters.routes[1].label == "legacy-route"
    assert response.filters.routes[1].deleted is True
    assert response.filters.routes[1].identity.current is None
    assert response.filters.routes[2].label == "Untracked"
    assert response.filters.routes[2].deleted is False
    assert [item.key for item in response.granularities] == ["day", "week", "month"]

    # exec order: user options / user existence / api_key options / ...
    # call_args_list[1] is the user-existence lookup, so the api_key options
    # query is at index 2.
    api_key_statement = str(session.exec.call_args_list[2].args[0])
    assert "api_key_name IS NOT NULL" in api_key_statement
    assert "access_key IS NOT NULL" in api_key_statement


@pytest.mark.asyncio
async def test_get_usage_meta_hides_admin_only_options_for_regular_user():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_user_name="alice",
                        group_api_key_name="test",
                        group_access_key="abcd1234",
                        group_api_key_is_custom=False,
                        group_user_id=12,
                        group_api_key_id=34,
                    )
                ]
            ),
            # api_key existence: key 34 still exists.
            _mock_exec_result([34]),
            # route options: none in scope (empty → no existence query follows).
            _mock_exec_result([]),
        ]
    )
    user = User(id=2, name="alice", is_admin=False)

    response = await get_usage_meta(session=session, user=user, ctx=_ctx_for(user))

    assert [item.key for item in response.group_bys] == [
        "date",
        "api_key",
        "route",
    ]
    assert response.filters.users == []
    assert response.filters.routes == []
    assert response.filters.api_keys[0].label == "alice / test"


def test_self_scope_personal_includes_null_consumer_rows():
    # Personal scope (org_id == the caller's own USER-principal): cookie-authed
    # direct chats land with consumer_principal_id NULL, so the caller's own
    # usage must still surface — the condition is an OR that admits NULL.
    condition = _self_scope_consumer_condition(user_id=7, org_id=7)
    sql = str(condition.compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" in sql
    assert " OR " in sql


def test_self_scope_org_context_excludes_null_consumer_rows():
    # Acting inside a real Org (org_id != caller's USER-principal): keep the
    # strict equality so personal direct usage doesn't bleed into the Org view.
    condition = _self_scope_consumer_condition(user_id=7, org_id=42)
    sql = str(condition.compile(compile_kwargs={"literal_binds": True}))
    assert "IS NULL" not in sql
    assert " OR " not in sql


def test_breakdown_request_allows_page_minus_one_sentinel():
    # page=-1 is the no-pagination sentinel used by the trend chart.
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        page=-1,
    )
    assert request.page == -1


def test_breakdown_request_allows_legacy_large_per_page():
    # perPage=10000 stays valid (older UIs fetch the whole set this way); the
    # whole series is otherwise fetched via page=-1.
    req = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        perPage=10000,
    )
    assert req.perPage == 10000


def test_breakdown_request_rejects_per_page_over_cap():
    # Beyond the abuse ceiling is still rejected.
    with pytest.raises(ValueError):
        UsageBreakdownRequest(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
            group_by=["date"],
            perPage=10001,
        )


def test_breakdown_request_rejects_page_zero():
    with pytest.raises(ValueError):
        UsageBreakdownRequest(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
            group_by=["date"],
            page=0,
        )


def test_breakdown_request_rejects_other_negative_page():
    # Only -1 is the sentinel; other negatives must not slip through as
    # "no pagination" and echo back as a bogus pagination.page.
    for bad in (-2, -42):
        with pytest.raises(ValueError):
            UsageBreakdownRequest(
                start_date=date(2026, 4, 1),
                end_date=date(2026, 4, 30),
                group_by=["date"],
                page=bad,
            )


@pytest.mark.asyncio
async def test_get_usage_breakdown_no_pagination_returns_all_date_buckets():
    # With page=-1 every date bucket is returned (no offset/limit), so a trend
    # spanning more buckets than the default perPage doesn't lose recent dates.
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=600,
                        output_tokens=240,
                        input_cached_tokens=0,
                        total_tokens=840,
                        api_requests=6,
                        models_called=1,
                    ),
                ]
            ),
            _mock_exec_result([3]),
            _mock_exec_result(
                [
                    SimpleNamespace(group_date=date(2026, 4, 1), total_tokens=500),
                    SimpleNamespace(group_date=date(2026, 4, 15), total_tokens=300),
                    SimpleNamespace(group_date=date(2026, 4, 30), total_tokens=40),
                ]
            ),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        granularity="day",
        page=-1,
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.pagination.page == -1
    assert response.pagination.total == 3
    assert response.pagination.totalPage == 1
    assert len(response.items) == 3
    assert response.items[-1].date.value == date(2026, 4, 30)


@pytest.mark.asyncio
async def test_get_usage_breakdown_no_pagination_rejects_oversized(monkeypatch):
    # page=-1 over a result larger than the configured cap is rejected before
    # the items are fetched — never silently truncated.
    monkeypatch.setattr(
        "gpustack.routes.usage.envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS", 2
    )
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=1,
                        output_tokens=0,
                        input_cached_tokens=0,
                        total_tokens=1,
                        api_requests=1,
                        models_called=1,
                    ),
                ]
            ),
            _mock_exec_result([3]),  # total = 3 > cap of 2
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        granularity="day",
        page=-1,
    )

    with pytest.raises(InvalidException):
        await get_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )


@pytest.mark.asyncio
async def test_get_usage_breakdown_returns_paginated_route_items():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=400,
                        output_tokens=140,
                        input_cached_tokens=100,
                        total_tokens=540,
                        api_requests=4,
                        models_called=2,
                    ),
                ]
            ),
            _mock_exec_result([2]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_model_route_name="qwen-route",
                        group_model_route_id=21,
                        input_tokens=300,
                        output_tokens=120,
                        input_cached_tokens=90,
                        total_tokens=420,
                        api_requests=3,
                        models_called=1,
                        api_keys_used=2,
                        last_active=date(2026, 4, 2),
                    ),
                    SimpleNamespace(
                        group_model_route_name="deepseek-route",
                        group_model_route_id=22,
                        input_tokens=100,
                        output_tokens=20,
                        input_cached_tokens=10,
                        total_tokens=120,
                        api_requests=1,
                        models_called=1,
                        api_keys_used=1,
                        last_active=date(2026, 4, 1),
                    ),
                ]
            ),
            # route existence: both routes still exist.
            _mock_exec_result([21, 22]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["route"],
        sort_by="-total_tokens",
        page=1,
        perPage=20,
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.summary.input_tokens == 400
    assert response.summary.output_tokens == 140
    assert response.summary.input_cached_tokens == 100
    assert response.summary.total_tokens == 540
    assert response.summary.api_requests == 4
    assert response.summary.models_called == 2
    assert response.group_by == ["route"]
    assert response.pagination.page == 1
    assert response.pagination.perPage == 20
    assert response.pagination.total == 2
    assert response.pagination.totalPage == 1
    assert len(response.items) == 2
    item = response.items[0]
    assert item.route.identity.value.route_name == "qwen-route"
    assert item.route.identity.current.route_id == 21
    assert item.route.label == "qwen-route"
    assert item.input_cached_tokens == 90
    assert item.avg_tokens_per_request == 140
    assert item.last_active == date(2026, 4, 2)


@pytest.mark.asyncio
async def test_get_usage_breakdown_returns_multidimensional_export_rows_with_no_api_key():
    session = MagicMock()
    session.get_bind.return_value = SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql")
    )
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=400,
                        output_tokens=160,
                        total_tokens=560,
                        api_requests=7,
                        models_called=2,
                    ),
                ]
            ),
            _mock_exec_result([2]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_date=date(2026, 3, 30),
                        group_user_name="alice",
                        group_user_id=12,
                        group_api_key_name="test",
                        group_access_key="abcd1234",
                        group_api_key_is_custom=False,
                        group_api_key_id=34,
                        group_model_route_name="gpt-route",
                        group_model_route_id=11,
                        input_tokens=300,
                        output_tokens=120,
                        total_tokens=420,
                        api_requests=5,
                        models_called=1,
                        api_keys_used=1,
                        last_active=date(2026, 4, 1),
                    ),
                    SimpleNamespace(
                        group_date=date(2026, 3, 30),
                        group_user_name="alice",
                        group_user_id=12,
                        group_api_key_name=None,
                        group_access_key=None,
                        group_api_key_is_custom=None,
                        group_api_key_id=None,
                        group_model_route_name="qwen-route",
                        group_model_route_id=12,
                        input_tokens=100,
                        output_tokens=40,
                        total_tokens=140,
                        api_requests=2,
                        models_called=1,
                        api_keys_used=0,
                        last_active=date(2026, 4, 2),
                    ),
                ]
            ),
            # Existence lookups, in dimension order user / route / api_key.
            _mock_exec_result([12]),  # user 12 exists
            _mock_exec_result([11, 12]),  # routes 11 & 12 exist
            _mock_exec_result([34]),  # api_key 34 exists
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date", "user", "api_key", "route"],
        granularity="week",
        sort_by="date",
        page=1,
        perPage=20,
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.group_by == ["date", "user", "api_key", "route"]
    assert response.granularity == "week"
    assert response.pagination.total == 2
    assert response.items[0].date.value == date(2026, 3, 30)
    assert response.items[0].user.label == "alice"
    assert response.items[0].api_key.label == "alice / test"
    assert response.items[0].route.label == "gpt-route"
    assert response.items[1].date.value == date(2026, 3, 30)
    assert response.items[1].api_key.identity is None
    assert response.items[1].api_key.label == "-"
    assert response.items[1].route.label == "qwen-route"

    count_sql = str(session.exec.call_args_list[1].args[0])
    items_sql = str(session.exec.call_args_list[2].args[0])
    assert "date_trunc" in count_sql
    assert "LIMIT" in items_sql
    assert "api_key_name IS NOT NULL" not in count_sql
    assert "access_key IS NOT NULL" not in count_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_flags_deleted_user_by_live_existence():
    """``user_id`` is FK-less, so a deleted user keeps its (dangling) id on the
    row instead of nulling out. The breakdown must resolve deletion by live
    principal existence — the gone user is tagged ``(Deleted)`` while the
    surviving user (with the same login-name snapshot) is not."""
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=0,
                        output_tokens=0,
                        input_cached_tokens=0,
                        total_tokens=0,
                        api_requests=0,
                        models_called=0,
                    ),
                ]
            ),
            _mock_exec_result([2]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_user_name="alice",
                        group_user_id=12,
                        input_tokens=300,
                        output_tokens=120,
                        total_tokens=420,
                        api_requests=5,
                        models_called=1,
                        api_keys_used=1,
                        last_active=date(2026, 4, 1),
                    ),
                    SimpleNamespace(
                        group_user_name="bob",
                        group_user_id=99,
                        input_tokens=100,
                        output_tokens=40,
                        total_tokens=140,
                        api_requests=2,
                        models_called=1,
                        api_keys_used=1,
                        last_active=date(2026, 4, 2),
                    ),
                ]
            ),
            # _existing_principal_ids: only user 12 survives; 99 was deleted.
            _mock_exec_result([12]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["user"],
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.items[0].user.label == "alice"
    assert response.items[0].user.deleted is False
    # id retained for attribution/filtering even though the principal is gone.
    assert response.items[0].user.identity.current.user_id == 12
    # Label stays the pure name; ``deleted`` + the retained id carry the state.
    assert response.items[1].user.label == "bob"
    assert response.items[1].user.deleted is True
    assert response.items[1].user.identity.current.user_id == 99


@pytest.mark.asyncio
async def test_get_usage_breakdown_flags_deleted_route_and_api_key_by_live_existence():
    """route / api_key are FK-less too — a deleted route or key keeps its
    dangling id, so the breakdown resolves deletion by live existence and tags
    the gone ones ``(Deleted)`` while the surviving ones are not."""
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=0,
                        output_tokens=0,
                        input_cached_tokens=0,
                        total_tokens=0,
                        api_requests=0,
                        models_called=0,
                    ),
                ]
            ),
            _mock_exec_result([2]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_user_name="alice",
                        group_api_key_name="live-key",
                        group_access_key="live1234",
                        group_api_key_is_custom=False,
                        group_user_id=12,
                        group_api_key_id=34,
                        group_model_route_name="live-route",
                        group_model_route_id=21,
                        input_tokens=300,
                        output_tokens=120,
                        total_tokens=420,
                        api_requests=5,
                        last_active=date(2026, 4, 1),
                    ),
                    SimpleNamespace(
                        group_user_name="alice",
                        group_api_key_name="gone-key",
                        group_access_key="gone1234",
                        group_api_key_is_custom=False,
                        group_user_id=12,
                        group_api_key_id=99,
                        group_model_route_name="gone-route",
                        group_model_route_id=88,
                        input_tokens=100,
                        output_tokens=40,
                        total_tokens=140,
                        api_requests=2,
                        last_active=date(2026, 4, 2),
                    ),
                ]
            ),
            # Existence lookups, dimension order route / api_key: only the
            # "live" ids survive; 88 (route) and 99 (key) were deleted.
            _mock_exec_result([21]),
            _mock_exec_result([34]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["api_key", "route"],
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.items[0].route.label == "live-route"
    assert response.items[0].route.deleted is False
    assert response.items[0].api_key.label == "alice / live-key"
    assert response.items[0].api_key.deleted is False
    # gone route/key keep their id but are tagged deleted.
    assert response.items[1].route.label == "gone-route"
    assert response.items[1].route.deleted is True
    assert response.items[1].route.identity.current.route_id == 88
    assert response.items[1].api_key.label == "alice / gone-key"
    assert response.items[1].api_key.deleted is True
    assert response.items[1].api_key.identity.current.api_key_id == 99


@pytest.mark.asyncio
async def test_get_usage_breakdown_ignores_incomplete_api_key_identity_groups():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=0,
                        output_tokens=0,
                        input_cached_tokens=0,
                        total_tokens=0,
                        api_requests=0,
                        models_called=0,
                    ),
                ]
            ),
            _mock_exec_result([0]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["api_key"],
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.summary == UsageSummary()
    assert response.items == []
    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "api_key_name IS NOT NULL" in executed_sql
    assert "access_key IS NOT NULL" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_formats_month_date_label_as_year_month():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=100,
                        output_tokens=40,
                        input_cached_tokens=10,
                        total_tokens=150,
                        api_requests=2,
                        models_called=1,
                    ),
                ]
            ),
            _mock_exec_result([1]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_date=date(2026, 4, 1),
                        input_tokens=100,
                        output_tokens=40,
                        input_cached_tokens=10,
                        total_tokens=150,
                        api_requests=2,
                        models_called=1,
                        api_keys_used=1,
                        last_active=date(2026, 4, 20),
                    ),
                ]
            ),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        granularity="month",
    )

    response = await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    assert response.granularity == "month"
    assert response.items[0].date.value == date(2026, 4, 1)
    assert response.items[0].date.label == "2026-04"


@pytest.mark.asyncio
async def test_get_usage_breakdown_filters_deleted_api_key_by_value_and_current():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([SimpleNamespace()]),
            _mock_exec_result([0]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=1, name="admin", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["api_key"],
        filters=UsageFilterRequest(
            api_keys=[
                UsageFilterItem(
                    identity=UsageIdentity(
                        value=UsageIdentityValue(
                            user_name="alice",
                            api_key_name="test",
                            access_key="abcd1234",
                            api_key_is_custom=False,
                        ),
                        current=None,
                    )
                )
            ]
        ),
    )

    await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "api_key_id IS NULL" in executed_sql
    assert "user_name" in executed_sql
    assert "api_key_name" in executed_sql
    assert "access_key" in executed_sql
    assert "api_key_is_custom" in executed_sql
    assert "api_key_name IS NOT NULL" in executed_sql
    assert "access_key IS NOT NULL" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_defaults_regular_user_to_self_scope():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([SimpleNamespace()]),
            _mock_exec_result([0]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=2, name="alice", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["api_key"],
    )

    await get_usage_breakdown(
        session=session, user=user, ctx=_ctx_for(user), request=request
    )

    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "model_usages.user_id =" in executed_sql
    assert "model_usages.consumer_principal_id =" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_org_owner_scopes_by_consumer_principal():
    """Org owner with default ``scope=all`` must filter the org
    dimension by the API-key owner (``consumer_principal_id``), not
    the deployment owner. A pure consumer (Org that only uses models
    granted from elsewhere) would otherwise see an empty page even
    though their members are actively spending."""
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([SimpleNamespace()]),
            _mock_exec_result([0]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=2, name="owner", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["route"],
    )

    await get_usage_breakdown(
        session=session, user=user, ctx=_org_owner_ctx(user), request=request
    )

    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "model_usages.consumer_principal_id =" in executed_sql
    assert "model_usages.owner_principal_id =" not in executed_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_rejects_regular_user_user_group():
    session = MagicMock()
    user = User(id=2, name="alice", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["user"],
    )

    with pytest.raises(ForbiddenException):
        await get_usage_breakdown(
            session=session, user=user, ctx=_ctx_for(user), request=request
        )
