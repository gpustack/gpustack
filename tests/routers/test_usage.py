from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ForbiddenException
from gpustack.routes.usage import (
    get_usage_breakdown,
    get_usage_meta,
    get_usage_timeseries,
)
from gpustack.schemas.users import User
from gpustack.schemas.usage import (
    UsageBreakdownRequest,
    UsageFilterItem,
    UsageFilterRequest,
    UsageIdentity,
    UsageIdentityValue,
    UsageTimeSeriesRequest,
)


def _mock_exec_result(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.mark.asyncio
async def test_get_usage_meta_returns_identity_filters_for_admin():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=7,
                    ),
                    SimpleNamespace(
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=None,
                    ),
                ]
            ),
            _mock_exec_result(
                [
                    SimpleNamespace(group_user_name="alice", group_user_id=12),
                    SimpleNamespace(group_user_name="alice", group_user_id=None),
                ]
            ),
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
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)

    response = await get_usage_meta(session=session, user=user)

    assert [item.key for item in response.group_bys] == [
        "model",
        "user",
        "api_key",
    ]
    assert [item.key for item in response.scopes] == ["all", "self"]
    assert response.filters.models[0].label == "cluster-a / qwen3.5-9b"
    assert response.filters.models[0].deleted is False
    assert response.filters.models[0].identity.current.model_id == 7
    assert response.filters.models[1].label == "cluster-a / qwen3.5-9b (Deleted)"
    assert response.filters.models[1].identity.current is None
    assert response.filters.users[1].label == "alice (Deleted)"
    assert response.filters.api_keys[0].label == "alice / test / gpustack_abcd***"
    assert response.filters.api_keys[0].identity.value.access_key == "abcd1234"
    assert response.filters.api_keys[0].identity.value.api_key_is_custom is False
    assert response.filters.api_keys[0].identity.current.api_key_id == 34
    assert response.filters.api_keys[1].label == "alice / custom / Custom API Key"
    assert response.filters.api_keys[1].identity.value.access_key == "hash1234"
    assert response.filters.api_keys[1].identity.value.api_key_is_custom is True
    assert [item.key for item in response.granularities] == ["day", "week", "month"]

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
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=7,
                    )
                ]
            ),
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
        ]
    )
    user = User(id=2, username="alice", hashed_password="x", is_admin=False)

    response = await get_usage_meta(session=session, user=user)

    assert [item.key for item in response.group_bys] == ["model", "api_key"]
    assert [item.key for item in response.scopes] == ["self"]
    assert response.filters.users == []
    assert response.filters.models[0].label == "cluster-a / qwen3.5-9b"


@pytest.mark.asyncio
async def test_get_usage_timeseries_returns_weekly_identity_series():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result(
                [
                    SimpleNamespace(
                        input_tokens=500,
                        output_tokens=200,
                        total_tokens=700,
                        api_requests=3,
                        models_called=2,
                    ),
                ]
            ),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_user_name="alice",
                        group_user_id=12,
                        date=date(2026, 4, 1),
                        value=100,
                    ),
                    SimpleNamespace(
                        group_user_name="alice",
                        group_user_id=12,
                        date=date(2026, 4, 2),
                        value=200,
                    ),
                    SimpleNamespace(
                        group_user_name="bob",
                        group_user_id=None,
                        date=date(2026, 4, 2),
                        value=200,
                    ),
                ]
            ),
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageTimeSeriesRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="all",
        metric="input_tokens",
        group_by="user",
        granularity="week",
    )

    response = await get_usage_timeseries(session=session, user=user, request=request)

    assert response.summary.input_tokens == 500
    assert response.summary.models_called == 2
    assert response.metric == "input_tokens"
    assert response.group_by == "user"
    assert response.granularity == "week"
    assert len(response.series) == 2

    alice = next(item for item in response.series if item.label == "alice")
    assert alice.identity.current.user_id == 12
    assert [(point.date, point.value) for point in alice.timeline] == [
        (date(2026, 3, 30), 300),
    ]

    bob = next(item for item in response.series if item.label == "bob (Deleted)")
    assert bob.deleted is True
    assert bob.identity.current is None
    assert [(point.date, point.value) for point in bob.timeline] == [
        (date(2026, 3, 30), 200),
    ]


@pytest.mark.asyncio
async def test_get_usage_breakdown_returns_paginated_model_items():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([2]),
            _mock_exec_result(
                [
                    SimpleNamespace(
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=7,
                        input_tokens=300,
                        output_tokens=120,
                        total_tokens=420,
                        api_requests=3,
                        models_called=1,
                        api_keys_used=2,
                        last_active=date(2026, 4, 2),
                    )
                ]
            ),
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="all",
        group_by="model",
        sort_by="-total_tokens",
        page=1,
        perPage=20,
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

    assert response.group_by == "model"
    assert response.pagination.page == 1
    assert response.pagination.perPage == 20
    assert response.pagination.total == 2
    assert response.pagination.totalPage == 1
    assert len(response.items) == 1
    item = response.items[0]
    assert item.identity.value.model_name == "qwen3.5-9b"
    assert item.identity.current.model_id == 7
    assert item.label == "cluster-a / qwen3.5-9b"
    assert item.cluster_name == "cluster-a"
    assert item.model_name == "qwen3.5-9b"
    assert item.avg_tokens_per_request == 140
    assert item.last_active == date(2026, 4, 2)


@pytest.mark.asyncio
async def test_get_usage_breakdown_ignores_incomplete_api_key_identity_groups():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([0]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="all",
        group_by="api_key",
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

    assert response.items == []
    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "api_key_name IS NOT NULL" in executed_sql
    assert "access_key IS NOT NULL" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_timeseries_filters_deleted_api_key_by_value_and_current():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _mock_exec_result([SimpleNamespace()]),
            _mock_exec_result([]),
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageTimeSeriesRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="all",
        metric="input_tokens",
        group_by="api_key",
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

    await get_usage_timeseries(session=session, user=user, request=request)

    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "api_key_id IS NULL" in executed_sql
    assert "user_name" in executed_sql
    assert "api_key_name" in executed_sql
    assert "access_key" in executed_sql
    assert "api_key_is_custom" in executed_sql
    assert "api_key_name IS NOT NULL" in executed_sql
    assert "access_key IS NOT NULL" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_timeseries_rejects_non_admin_global_scope():
    session = MagicMock()
    user = User(id=2, username="alice", hashed_password="x", is_admin=False)
    request = UsageTimeSeriesRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="all",
        metric="input_tokens",
        group_by="model",
    )

    with pytest.raises(ForbiddenException):
        await get_usage_timeseries(session=session, user=user, request=request)


@pytest.mark.asyncio
async def test_get_usage_breakdown_rejects_regular_user_user_group():
    session = MagicMock()
    user = User(id=2, username="alice", hashed_password="x", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        scope="self",
        group_by="user",
    )

    with pytest.raises(ForbiddenException):
        await get_usage_breakdown(session=session, user=user, request=request)
