"""Image generation usage in the usage API.

Query construction is checked on the compiled SQL; response mapping and
export are checked against mocked session results.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from gpustack.routes.usage import (
    _apply_usage_scope_and_filters,
    _base_statement,
    _metric_columns,
    get_usage_breakdown,
)
from gpustack.schemas.model_usage import OperationEnum
from gpustack.schemas.usage import (
    USAGE_SCOPE_ALL,
    USAGE_SCOPE_SELF,
    UsageBreakdownDimension,
    UsageBreakdownItem,
    UsageBreakdownRequest,
    UsageFilterRequest,
    UsageIdentity,
    UsageIdentityCurrent,
    UsageIdentityValue,
)
from gpustack.schemas.users import User
from gpustack.utils.usage_export import build_export_columns, export_row

ADMIN = User(id=1, name="admin", is_admin=True)
ALICE = User(id=2, name="alice", is_admin=False)
MP_1024 = 1024 * 1024
IMAGE_ONLY = UsageFilterRequest(operations=[OperationEnum.IMAGE_GENERATION])


def _sql(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def _scoped(filters, *, user=ADMIN, scope=USAGE_SCOPE_ALL, org_id=None) -> str:
    statement = _apply_usage_scope_and_filters(
        _base_statement().with_only_columns(*_metric_columns().values()),
        user=user,
        filters=filters,
        scope=scope,
        org_id=org_id,
    )
    return _sql(statement)


def _ctx_for(user):
    ctx = MagicMock()
    ctx.user = user
    ctx.is_platform_admin = bool(user.is_admin)
    ctx.current_principal_id = None if user.is_admin else 1
    ctx.org_role = None
    ctx.current_is_personal_scope = False
    return ctx


def _exec_result(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


def _summary_row(**metrics):
    base = dict(
        input_tokens=0,
        output_tokens=0,
        input_cached_tokens=0,
        total_tokens=0,
        api_requests=0,
        models_called=0,
        image_count=0,
        image_pixels=0,
    )
    return SimpleNamespace(**{**base, **metrics})


def test_operations_filter_restricts_rows_to_the_given_operations():
    sql = _scoped(IMAGE_ONLY)

    assert "model_usages.operation IN ('IMAGE_GENERATION')" in sql


def test_no_operations_filter_keeps_every_operation():
    sql = _scoped(UsageFilterRequest())

    assert "model_usages.operation" not in sql


def test_operations_filter_never_widens_self_scope():
    sql = _scoped(IMAGE_ONLY, user=ALICE, scope=USAGE_SCOPE_SELF, org_id=1)

    assert "model_usages.user_id = 2" in sql
    assert "model_usages.operation IN ('IMAGE_GENERATION')" in sql


def test_image_metrics_sum_the_image_columns():
    sql = _scoped(UsageFilterRequest())

    assert "sum(model_usages.image_count)" in sql
    assert "sum(model_usages.image_pixel_count)" in sql


@pytest.mark.parametrize(
    "operations",
    [["video_generation"], ["IMAGE_GENERATION"], [""]],
)
def test_unknown_operation_is_rejected(operations):
    with pytest.raises(ValidationError):
        UsageFilterRequest(operations=operations)


@pytest.mark.asyncio
async def test_breakdown_reports_image_units_per_group_and_in_summary():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _exec_result(
                [
                    _summary_row(
                        api_requests=3,
                        models_called=1,
                        image_count=6,
                        image_pixels=4 * MP_1024 + 2 * 512 * 512,
                    )
                ]
            ),
            _exec_result([1]),
            _exec_result(
                [
                    SimpleNamespace(
                        group_model_route_name="sd-turbo",
                        group_model_route_id=21,
                        **vars(
                            _summary_row(
                                api_requests=3,
                                models_called=1,
                                image_count=6,
                                image_pixels=4 * MP_1024 + 2 * 512 * 512,
                            )
                        ),
                        api_keys_used=0,
                        last_active=date(2026, 4, 2),
                    )
                ]
            ),
            _exec_result([21]),
        ]
    )
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        group_by=["route"],
        filters=IMAGE_ONLY,
        sort_by="-image_count",
    )

    response = await get_usage_breakdown(
        session=session, user=ADMIN, ctx=_ctx_for(ADMIN), request=request
    )

    assert response.summary.image_count == 6
    assert response.summary.image_pixels == 4 * MP_1024 + 2 * 512 * 512
    assert response.summary.total_tokens == 0
    (item,) = response.items
    assert item.route.label == "sd-turbo"
    assert item.image_count == 6
    assert item.image_pixels == 4 * MP_1024 + 2 * 512 * 512
    breakdown_sql = " ".join(_sql(session.exec.call_args_list[2].args[0]).split())
    assert "model_usages.operation IN ('IMAGE_GENERATION')" in breakdown_sql
    assert "ORDER BY coalesce(sum(model_usages.image_count), 0) DESC" in breakdown_sql


@pytest.mark.asyncio
async def test_breakdown_without_image_usage_reports_zero_image_units():
    session = MagicMock()
    session.exec = AsyncMock(
        side_effect=[
            _exec_result([SimpleNamespace()]),
            _exec_result([0]),
            _exec_result([]),
        ]
    )
    request = UsageBreakdownRequest(
        start_date=date(2026, 4, 1), end_date=date(2026, 4, 2), group_by=["route"]
    )

    response = await get_usage_breakdown(
        session=session, user=ADMIN, ctx=_ctx_for(ADMIN), request=request
    )

    assert response.summary.image_count == 0
    assert response.summary.image_pixels == 0


def test_export_carries_image_columns_in_header_and_row():
    item = UsageBreakdownItem(
        route=UsageBreakdownDimension(
            identity=UsageIdentity(
                value=UsageIdentityValue(route_name="sd-turbo"),
                current=UsageIdentityCurrent(route_id=21),
            ),
            label="sd-turbo",
            deleted=False,
        ),
        api_requests=3,
        image_count=6,
        image_pixels=4 * MP_1024,
    )

    header = build_export_columns(["route"])
    row = dict(zip(header, export_row(item, ["route"])))

    assert row["Images Generated"] == 6
    assert row["Image Pixels"] == 4 * MP_1024
