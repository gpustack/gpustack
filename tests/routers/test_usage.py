from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ForbiddenException
from gpustack.routes.usage import (
    get_usage_breakdown,
    get_usage_meta,
)
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
                        group_provider_id=None,
                        group_provider_name=None,
                        group_provider_type=None,
                    ),
                    SimpleNamespace(
                        group_cluster_name=None,
                        group_model_name="gpt-4o",
                        group_model_id=None,
                        group_provider_id=9,
                        group_provider_name="openai-prod",
                        group_provider_type="openai",
                    ),
                    SimpleNamespace(
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=None,
                        group_provider_id=None,
                        group_provider_name=None,
                        group_provider_type=None,
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
        "date",
        "model",
        "user",
        "api_key",
    ]
    assert [item.key for item in response.metrics] == [
        "input_tokens",
        "output_tokens",
        "input_cached_tokens",
        "total_tokens",
        "api_requests",
    ]
    assert response.filters.models[0].label == "cluster-a / qwen3.5-9b"
    assert response.filters.models[0].deleted is False
    assert response.filters.models[0].identity.current.model_id == 7
    assert response.filters.models[1].label == "openai-prod / gpt-4o"
    assert response.filters.models[1].deleted is False
    assert response.filters.models[1].identity.current.provider_id == 9
    assert response.filters.models[1].identity.value.provider_name == "openai-prod"
    assert response.filters.models[1].identity.value.provider_type == "openai"
    assert response.filters.models[2].label == "cluster-a / qwen3.5-9b (Deleted)"
    assert response.filters.models[2].identity.current is None
    assert response.filters.users[1].label == "alice (Deleted)"
    assert response.filters.api_keys[0].label == "alice / test"
    assert response.filters.api_keys[0].identity.value.access_key == "abcd1234"
    assert response.filters.api_keys[0].identity.value.api_key_is_custom is False
    assert response.filters.api_keys[0].identity.current.api_key_id == 34
    assert response.filters.api_keys[1].label == "alice / custom"
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

    assert [item.key for item in response.group_bys] == ["date", "model", "api_key"]
    assert response.filters.users == []
    assert response.filters.models[0].label == "cluster-a / qwen3.5-9b"


@pytest.mark.asyncio
async def test_get_usage_breakdown_returns_paginated_model_items():
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
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=7,
                        group_provider_id=None,
                        group_provider_name=None,
                        group_provider_type=None,
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
                        group_cluster_name="cluster-b",
                        group_model_name="deepseek-v3",
                        group_model_id=8,
                        group_provider_id=None,
                        group_provider_name=None,
                        group_provider_type=None,
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
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["model"],
        sort_by="-total_tokens",
        page=1,
        perPage=20,
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

    assert response.summary.input_tokens == 400
    assert response.summary.output_tokens == 140
    assert response.summary.input_cached_tokens == 100
    assert response.summary.total_tokens == 540
    assert response.summary.api_requests == 4
    assert response.summary.models_called == 2
    assert response.group_by == ["model"]
    assert response.pagination.page == 1
    assert response.pagination.perPage == 20
    assert response.pagination.total == 2
    assert response.pagination.totalPage == 1
    assert len(response.items) == 2
    item = response.items[0]
    assert item.model.identity.value.model_name == "qwen3.5-9b"
    assert item.model.identity.current.model_id == 7
    assert item.model.label == "cluster-a / qwen3.5-9b"
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
                        group_cluster_name="cluster-a",
                        group_model_name="gpt-4o",
                        group_model_id=7,
                        group_provider_id=3,
                        group_provider_name="openai-prod",
                        group_provider_type="openai",
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
                        group_cluster_name="cluster-a",
                        group_model_name="qwen3.5-9b",
                        group_model_id=8,
                        group_provider_id=None,
                        group_provider_name=None,
                        group_provider_type=None,
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
        ]
    )
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["date", "user", "api_key", "model"],
        granularity="week",
        sort_by="date",
        page=1,
        perPage=20,
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

    assert response.group_by == ["date", "user", "api_key", "model"]
    assert response.granularity == "week"
    assert response.pagination.total == 2
    assert response.items[0].date.value == date(2026, 3, 30)
    assert response.items[0].user.label == "alice"
    assert response.items[0].api_key.label == "alice / test"
    assert response.items[0].model.label == "cluster-a / openai-prod / gpt-4o"
    assert response.items[1].date.value == date(2026, 3, 30)
    assert response.items[1].api_key.identity is None
    assert response.items[1].api_key.label == "-"

    count_sql = str(session.exec.call_args_list[1].args[0])
    items_sql = str(session.exec.call_args_list[2].args[0])
    assert "date_trunc" in count_sql
    assert "LIMIT" in items_sql
    assert "api_key_name IS NOT NULL" not in count_sql
    assert "access_key IS NOT NULL" not in count_sql


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
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["api_key"],
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

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
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        group_by=["date"],
        granularity="month",
    )

    response = await get_usage_breakdown(session=session, user=user, request=request)

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
    user = User(id=1, username="admin", hashed_password="x", is_admin=True)
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

    await get_usage_breakdown(session=session, user=user, request=request)

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
    user = User(id=2, username="alice", hashed_password="x", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["model"],
    )

    await get_usage_breakdown(session=session, user=user, request=request)

    executed_sql = str(session.exec.call_args_list[0].args[0])
    assert "model_usages.user_id =" in executed_sql


@pytest.mark.asyncio
async def test_get_usage_breakdown_rejects_regular_user_user_group():
    session = MagicMock()
    user = User(id=2, username="alice", hashed_password="x", is_admin=False)
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["user"],
    )

    with pytest.raises(ForbiddenException):
        await get_usage_breakdown(session=session, user=user, request=request)
