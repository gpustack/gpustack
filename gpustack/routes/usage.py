from collections import defaultdict
from datetime import date, timedelta
from math import ceil
from typing import Any, Dict, List

from fastapi import APIRouter
from sqlalchemy import Select, String, and_, asc, cast, desc, literal
from sqlmodel import func, or_, select

from gpustack.api.exceptions import ForbiddenException
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.users import User
from gpustack.schemas.usage import (
    USAGE_GRANULARITY_MONTH,
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_WEEK,
    USAGE_GROUP_BY_API_KEY,
    USAGE_GROUP_BY_MODEL,
    USAGE_GROUP_BY_USER,
    USAGE_METRIC_API_KEYS_USED,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_LAST_ACTIVE,
    USAGE_METRIC_MODELS_CALLED,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_SCOPE_ALL,
    USAGE_SCOPE_SELF,
    USAGE_SORT_DESC,
    UsageBreakdownItem,
    UsageBreakdownRequest,
    UsageBreakdownResponse,
    UsageFilterItem,
    UsageFilterOption,
    UsageFilters,
    UsageIdentity,
    UsageIdentityCurrent,
    UsageIdentityValue,
    UsageMetaResponse,
    UsageOption,
    UsageSeries,
    UsageSummary,
    UsageTimeSeriesRequest,
    UsageTimeSeriesResponse,
    UsageTimelinePoint,
)
from gpustack.schemas.common import Pagination
from gpustack.server.deps import CurrentUserDep, SessionDep
from gpustack.utils.api_keys import get_masked_api_key_value
from gpustack.utils.usage_snapshots import format_model_snapshot_label

router = APIRouter()

METRIC_OPTIONS = [
    UsageOption(key=USAGE_METRIC_INPUT_TOKENS, label="Input Tokens"),
    UsageOption(key=USAGE_METRIC_OUTPUT_TOKENS, label="Output Tokens"),
    UsageOption(key=USAGE_METRIC_TOTAL_TOKENS, label="Total Tokens"),
    UsageOption(key=USAGE_METRIC_API_REQUESTS, label="API Requests"),
]
GRANULARITY_OPTIONS = [
    UsageOption(key=USAGE_GRANULARITY_DAY, label="Day"),
    UsageOption(key=USAGE_GRANULARITY_WEEK, label="Week"),
    UsageOption(key=USAGE_GRANULARITY_MONTH, label="Month"),
]
SELF_GROUP_BY_OPTIONS = [
    UsageOption(key=USAGE_GROUP_BY_MODEL, label="Model"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
]
ADMIN_GROUP_BY_OPTIONS = [
    UsageOption(key=USAGE_GROUP_BY_MODEL, label="Model"),
    UsageOption(key=USAGE_GROUP_BY_USER, label="User"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
]
ADMIN_SCOPES = [
    UsageOption(key=USAGE_SCOPE_ALL, label="All Users"),
    UsageOption(key=USAGE_SCOPE_SELF, label="My Usage"),
]
SELF_SCOPES = [UsageOption(key=USAGE_SCOPE_SELF, label="My Usage")]


def _null_safe_column_filter(column, value: Any):
    if value is None:
        return column.is_(None)
    return column == value


def _model_key_expression():
    return (
        func.coalesce(ModelUsage.cluster_name + literal("/"), literal(""))
        + ModelUsage.model_name
    )


def _metric_columns() -> Dict[str, Any]:
    model_identity = (
        func.coalesce(cast(ModelUsage.model_id, String), literal("deleted"))
        + literal("|")
        + _model_key_expression()
    )
    return {
        USAGE_METRIC_INPUT_TOKENS: func.coalesce(
            func.sum(ModelUsage.prompt_token_count), 0
        ),
        USAGE_METRIC_OUTPUT_TOKENS: func.coalesce(
            func.sum(ModelUsage.completion_token_count), 0
        ),
        USAGE_METRIC_TOTAL_TOKENS: func.coalesce(
            func.sum(ModelUsage.prompt_token_count + ModelUsage.completion_token_count),
            0,
        ),
        USAGE_METRIC_API_REQUESTS: func.coalesce(func.sum(ModelUsage.request_count), 0),
        USAGE_METRIC_MODELS_CALLED: func.count(func.distinct(model_identity)),
    }


def _group_columns(group_by: str):
    if group_by == USAGE_GROUP_BY_MODEL:
        return [
            ModelUsage.cluster_name.label("group_cluster_name"),
            ModelUsage.model_name.label("group_model_name"),
            ModelUsage.model_id.label("group_model_id"),
        ], [
            ModelUsage.cluster_name,
            ModelUsage.model_name,
            ModelUsage.model_id,
        ]
    if group_by == USAGE_GROUP_BY_USER:
        return [
            ModelUsage.user_name.label("group_user_name"),
            ModelUsage.user_id.label("group_user_id"),
        ], [ModelUsage.user_name, ModelUsage.user_id]
    return [
        ModelUsage.user_name.label("group_user_name"),
        ModelUsage.api_key_name.label("group_api_key_name"),
        ModelUsage.access_key.label("group_access_key"),
        ModelUsage.api_key_is_custom.label("group_api_key_is_custom"),
        ModelUsage.user_id.label("group_user_id"),
        ModelUsage.api_key_id.label("group_api_key_id"),
    ], [
        ModelUsage.user_name,
        ModelUsage.api_key_name,
        ModelUsage.access_key,
        ModelUsage.api_key_is_custom,
        ModelUsage.user_id,
        ModelUsage.api_key_id,
    ]


def _row_identity(group_by: str, row: Any) -> UsageIdentity:
    if group_by == USAGE_GROUP_BY_MODEL:
        model_id = getattr(row, "group_model_id", None)
        current = None
        if model_id is not None:
            current = UsageIdentityCurrent(
                model_id=model_id,
            )
        return UsageIdentity(
            value=UsageIdentityValue(
                cluster_name=getattr(row, "group_cluster_name", None),
                model_name=getattr(row, "group_model_name", None),
            ),
            current=current,
        )
    if group_by == USAGE_GROUP_BY_USER:
        user_id = getattr(row, "group_user_id", None)
        return UsageIdentity(
            value=UsageIdentityValue(user_name=getattr(row, "group_user_name", None)),
            current=None if user_id is None else UsageIdentityCurrent(user_id=user_id),
        )
    api_key_id = getattr(row, "group_api_key_id", None)
    current = None
    if api_key_id is not None:
        current = UsageIdentityCurrent(
            user_id=getattr(row, "group_user_id", None),
            api_key_id=api_key_id,
        )
    return UsageIdentity(
        value=UsageIdentityValue(
            user_name=getattr(row, "group_user_name", None),
            api_key_name=getattr(row, "group_api_key_name", None),
            access_key=getattr(row, "group_access_key", None),
            api_key_is_custom=getattr(row, "group_api_key_is_custom", None),
        ),
        current=current,
    )


def _identity_key(identity: UsageIdentity) -> tuple:
    return (
        identity.value.cluster_name,
        identity.value.model_name,
        identity.value.user_name,
        identity.value.api_key_name,
        identity.value.access_key,
        identity.value.api_key_is_custom,
        None if identity.current is None else identity.current.model_id,
        None if identity.current is None else identity.current.user_id,
        None if identity.current is None else identity.current.api_key_id,
    )


def _identity_deleted(identity: UsageIdentity) -> bool:
    return identity.current is None


def _identity_label(group_by: str, identity: UsageIdentity) -> str:
    value = identity.value
    if group_by == USAGE_GROUP_BY_MODEL:
        label = (
            format_model_snapshot_label(value.model_name, value.cluster_name)
            if value.model_name
            else "Unknown Model"
        )
    elif group_by == USAGE_GROUP_BY_USER:
        label = value.user_name or "Unknown User"
    else:
        parts = []
        if value.user_name:
            parts.append(value.user_name)
        if value.api_key_name:
            parts.append(value.api_key_name)
        if value.access_key:
            parts.append(
                get_masked_api_key_value(value.access_key, value.api_key_is_custom)
            )
        else:
            parts.append("Unknown API Key")
        label = " / ".join(parts) if parts else "Unknown API Key"

    if _identity_deleted(identity):
        label = f"{label} (Deleted)"
    return label


def _filter_condition(group_by: str, item: UsageFilterItem):
    value = item.identity.value
    current = item.identity.current
    conditions = []
    if group_by == USAGE_GROUP_BY_MODEL:
        conditions.extend(
            [
                _null_safe_column_filter(ModelUsage.cluster_name, value.cluster_name),
                _null_safe_column_filter(ModelUsage.model_name, value.model_name),
            ]
        )
        if current is None or current.model_id is None:
            conditions.append(ModelUsage.model_id.is_(None))
        else:
            conditions.append(ModelUsage.model_id == current.model_id)
    elif group_by == USAGE_GROUP_BY_USER:
        conditions.append(
            _null_safe_column_filter(ModelUsage.user_name, value.user_name)
        )
        if current is None or current.user_id is None:
            conditions.append(ModelUsage.user_id.is_(None))
        else:
            conditions.append(ModelUsage.user_id == current.user_id)
    else:
        conditions.extend(
            [
                _null_safe_column_filter(ModelUsage.user_name, value.user_name),
                _null_safe_column_filter(ModelUsage.api_key_name, value.api_key_name),
                _null_safe_column_filter(ModelUsage.access_key, value.access_key),
            ]
        )
        if value.api_key_is_custom is not None:
            conditions.append(
                _null_safe_column_filter(
                    ModelUsage.api_key_is_custom, value.api_key_is_custom
                )
            )
        if current is None or current.api_key_id is None:
            conditions.append(ModelUsage.api_key_id.is_(None))
        else:
            conditions.append(ModelUsage.api_key_id == current.api_key_id)
            conditions.append(
                _null_safe_column_filter(ModelUsage.user_id, current.user_id)
            )
    return and_(*conditions)


def _apply_usage_scope_and_filters(
    statement: Select,
    *,
    user: User,
    scope: str,
    filters,
) -> Select:
    if not user.is_admin or scope == USAGE_SCOPE_SELF:
        statement = statement.where(ModelUsage.user_id == user.id)

    for group_by, items in [
        (USAGE_GROUP_BY_MODEL, filters.models),
        (USAGE_GROUP_BY_USER, filters.users),
        (USAGE_GROUP_BY_API_KEY, filters.api_keys),
    ]:
        if not items:
            continue
        if group_by == USAGE_GROUP_BY_USER and not user.is_admin:
            raise ForbiddenException(message="No permission to filter by user")
        statement = statement.where(
            or_(*[_filter_condition(group_by, item) for item in items])
        )
    return statement


def _base_statement() -> Select:
    return select().select_from(ModelUsage)


def _date_scoped_statement(
    statement: Select, start_date: date, end_date: date
) -> Select:
    return statement.where(ModelUsage.date >= start_date).where(
        ModelUsage.date <= end_date
    )


def _exclude_incomplete_api_key_identity(statement: Select) -> Select:
    return (
        statement.where(ModelUsage.api_key_name.is_not(None))
        .where(ModelUsage.api_key_name != "")
        .where(ModelUsage.access_key.is_not(None))
        .where(ModelUsage.access_key != "")
    )


def _bucket_date(value: date, granularity: str) -> date:
    if granularity == USAGE_GRANULARITY_WEEK:
        return value - timedelta(days=value.weekday())
    if granularity == USAGE_GRANULARITY_MONTH:
        return date(value.year, value.month, 1)
    return value


async def _get_rows(session, statement: Select):
    return (await session.exec(statement)).all()


async def _get_first_row(session, statement: Select):
    rows = await _get_rows(session, statement)
    return rows[0] if rows else None


def _summary_from_row(row: Any) -> UsageSummary:
    if row is None:
        return UsageSummary()
    return UsageSummary(
        input_tokens=int(getattr(row, USAGE_METRIC_INPUT_TOKENS, 0) or 0),
        output_tokens=int(getattr(row, USAGE_METRIC_OUTPUT_TOKENS, 0) or 0),
        total_tokens=int(getattr(row, USAGE_METRIC_TOTAL_TOKENS, 0) or 0),
        api_requests=int(getattr(row, USAGE_METRIC_API_REQUESTS, 0) or 0),
        models_called=int(getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0),
    )


def _option_from_identity(group_by: str, identity: UsageIdentity) -> UsageFilterOption:
    return UsageFilterOption(
        identity=identity,
        label=_identity_label(group_by, identity),
        deleted=_identity_deleted(identity),
    )


async def _get_filter_options(
    session,
    *,
    base_statement: Select,
    group_by: str,
) -> List[UsageFilterOption]:
    select_columns, group_columns = _group_columns(group_by)
    if group_by == USAGE_GROUP_BY_API_KEY:
        base_statement = _exclude_incomplete_api_key_identity(base_statement)
    rows = await _get_rows(
        session,
        base_statement.with_only_columns(*select_columns)
        .distinct()
        .order_by(*group_columns),
    )
    return [
        _option_from_identity(group_by, _row_identity(group_by, row)) for row in rows
    ]


@router.get("/meta", response_model=UsageMetaResponse)
async def get_usage_meta(session: SessionDep, user: CurrentUserDep):
    base_statement = _base_statement()
    if not user.is_admin:
        base_statement = base_statement.where(ModelUsage.user_id == user.id)

    model_options = await _get_filter_options(
        session, base_statement=base_statement, group_by=USAGE_GROUP_BY_MODEL
    )
    user_options: List[UsageFilterOption] = []
    if user.is_admin:
        user_options = await _get_filter_options(
            session, base_statement=base_statement, group_by=USAGE_GROUP_BY_USER
        )
    api_key_options = await _get_filter_options(
        session, base_statement=base_statement, group_by=USAGE_GROUP_BY_API_KEY
    )

    return UsageMetaResponse(
        scopes=ADMIN_SCOPES if user.is_admin else SELF_SCOPES,
        metrics=METRIC_OPTIONS,
        granularities=GRANULARITY_OPTIONS,
        group_bys=ADMIN_GROUP_BY_OPTIONS if user.is_admin else SELF_GROUP_BY_OPTIONS,
        filters=UsageFilters(
            models=model_options,
            users=user_options,
            api_keys=api_key_options,
        ),
    )


def _check_permission(user: User, request) -> None:
    if request.scope == USAGE_SCOPE_ALL and not user.is_admin:
        raise ForbiddenException(message="No permission to access all users usage")
    if request.group_by == USAGE_GROUP_BY_USER and not user.is_admin:
        raise ForbiddenException(message="No permission to group by user")
    if request.filters.users and not user.is_admin:
        raise ForbiddenException(message="No permission to filter by user")


@router.post("/timeseries", response_model=UsageTimeSeriesResponse)
async def get_usage_timeseries(
    session: SessionDep,
    user: CurrentUserDep,
    request: UsageTimeSeriesRequest,
):
    _check_permission(user, request)

    metric_columns = _metric_columns()
    base_statement = _apply_usage_scope_and_filters(
        _base_statement(),
        user=user,
        scope=request.scope,
        filters=request.filters,
    )
    if request.group_by == USAGE_GROUP_BY_API_KEY:
        base_statement = _exclude_incomplete_api_key_identity(base_statement)
    scoped_statement = _date_scoped_statement(
        base_statement, request.start_date, request.end_date
    )

    summary_columns = [metric_columns[item].label(item) for item in metric_columns]
    summary_row = await _get_first_row(
        session, scoped_statement.with_only_columns(*summary_columns)
    )

    select_columns, group_columns = _group_columns(request.group_by)
    timeline_rows = await _get_rows(
        session,
        scoped_statement.with_only_columns(
            *select_columns,
            ModelUsage.date.label("date"),
            metric_columns[request.metric].label("value"),
        )
        .group_by(*group_columns, ModelUsage.date)
        .order_by(*group_columns, ModelUsage.date),
    )

    order: List[tuple] = []
    identities: Dict[tuple, UsageIdentity] = {}
    timelines: Dict[tuple, Dict[date, int]] = defaultdict(lambda: defaultdict(int))
    for row in timeline_rows:
        identity = _row_identity(request.group_by, row)
        key = _identity_key(identity)
        if key not in identities:
            order.append(key)
            identities[key] = identity
        point_date = _bucket_date(row.date, request.granularity)
        timelines[key][point_date] += int(getattr(row, "value", 0) or 0)

    return UsageTimeSeriesResponse(
        summary=_summary_from_row(summary_row),
        metric=request.metric,
        group_by=request.group_by,
        granularity=request.granularity,
        series=[
            UsageSeries(
                identity=identities[key],
                label=_identity_label(request.group_by, identities[key]),
                deleted=_identity_deleted(identities[key]),
                timeline=[
                    UsageTimelinePoint(date=item[0], value=item[1])
                    for item in sorted(timelines[key].items())
                ],
            )
            for key in order
        ],
    )


def _sort_expression(sort_by: str, metric_columns: Dict[str, Any]):
    if sort_by == USAGE_METRIC_AVG_TOKENS_PER_REQUEST:
        return metric_columns[USAGE_METRIC_TOTAL_TOKENS] / func.nullif(
            metric_columns[USAGE_METRIC_API_REQUESTS], 0
        )
    if sort_by == USAGE_METRIC_API_KEYS_USED:
        return func.count(func.distinct(ModelUsage.access_key))
    if sort_by == USAGE_METRIC_LAST_ACTIVE:
        return func.max(ModelUsage.date)
    if sort_by in metric_columns:
        return metric_columns[sort_by]
    return metric_columns[USAGE_METRIC_TOTAL_TOKENS]


def _order_expression(order_by: List[tuple[str, str]], metric_columns: Dict[str, Any]):
    if not order_by:
        order_by = [(USAGE_METRIC_TOTAL_TOKENS, USAGE_SORT_DESC)]

    sort_exprs = []
    for sort_by, direction in order_by:
        sort_expr = _sort_expression(sort_by, metric_columns)
        sort_exprs.append(
            desc(sort_expr) if direction == USAGE_SORT_DESC else asc(sort_expr)
        )
    return sort_exprs


def _row_count_value(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, tuple):
        return int(row[0] or 0)
    return int(row or 0)


def _build_breakdown_item(group_by: str, row: Any) -> UsageBreakdownItem:
    identity = _row_identity(group_by, row)
    api_requests = int(getattr(row, USAGE_METRIC_API_REQUESTS, 0) or 0)
    total_tokens = int(getattr(row, USAGE_METRIC_TOTAL_TOKENS, 0) or 0)
    item = UsageBreakdownItem(
        identity=identity,
        label=_identity_label(group_by, identity),
        deleted=_identity_deleted(identity),
        input_tokens=int(getattr(row, USAGE_METRIC_INPUT_TOKENS, 0) or 0),
        output_tokens=int(getattr(row, USAGE_METRIC_OUTPUT_TOKENS, 0) or 0),
        total_tokens=total_tokens,
        api_requests=api_requests,
        avg_tokens_per_request=total_tokens / api_requests if api_requests else 0,
        last_active=getattr(row, USAGE_METRIC_LAST_ACTIVE, None),
    )
    if group_by == USAGE_GROUP_BY_MODEL:
        item.cluster_name = identity.value.cluster_name
        item.model_name = identity.value.model_name
    elif group_by == USAGE_GROUP_BY_USER:
        item.user_name = identity.value.user_name or "Unknown User"
        item.models_called = int(getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0)
        item.api_keys_used = int(getattr(row, USAGE_METRIC_API_KEYS_USED, 0) or 0)
    else:
        item.user_name = identity.value.user_name or "Unknown User"
        item.api_key_name = identity.value.api_key_name
        item.models_called = int(getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0)
    return item


@router.post("/breakdown", response_model=UsageBreakdownResponse)
async def get_usage_breakdown(
    session: SessionDep,
    user: CurrentUserDep,
    request: UsageBreakdownRequest,
):
    _check_permission(user, request)

    metric_columns = _metric_columns()
    base_statement = _apply_usage_scope_and_filters(
        _base_statement(),
        user=user,
        scope=request.scope,
        filters=request.filters,
    )
    if request.group_by == USAGE_GROUP_BY_API_KEY:
        base_statement = _exclude_incomplete_api_key_identity(base_statement)
    scoped_statement = _date_scoped_statement(
        base_statement, request.start_date, request.end_date
    )
    select_columns, group_columns = _group_columns(request.group_by)
    aggregate_columns = [
        metric_columns[USAGE_METRIC_INPUT_TOKENS].label(USAGE_METRIC_INPUT_TOKENS),
        metric_columns[USAGE_METRIC_OUTPUT_TOKENS].label(USAGE_METRIC_OUTPUT_TOKENS),
        metric_columns[USAGE_METRIC_TOTAL_TOKENS].label(USAGE_METRIC_TOTAL_TOKENS),
        metric_columns[USAGE_METRIC_API_REQUESTS].label(USAGE_METRIC_API_REQUESTS),
        metric_columns[USAGE_METRIC_MODELS_CALLED].label(USAGE_METRIC_MODELS_CALLED),
        func.count(func.distinct(ModelUsage.access_key)).label(
            USAGE_METRIC_API_KEYS_USED
        ),
        func.max(ModelUsage.date).label(USAGE_METRIC_LAST_ACTIVE),
    ]
    grouped_statement = scoped_statement.with_only_columns(
        *select_columns, *aggregate_columns
    ).group_by(*group_columns)
    count_statement = select(func.count()).select_from(grouped_statement.subquery())

    sort_exprs = _order_expression(request.order_by, metric_columns)
    items_statement = (
        grouped_statement.order_by(*sort_exprs)
        .offset((request.page - 1) * request.perPage)
        .limit(request.perPage)
    )

    total = _row_count_value(await _get_first_row(session, count_statement))
    item_rows = await _get_rows(session, items_statement)

    return UsageBreakdownResponse(
        group_by=request.group_by,
        pagination=Pagination(
            page=request.page,
            perPage=request.perPage,
            total=total,
            totalPage=ceil(total / request.perPage) if total else 0,
        ),
        items=[_build_breakdown_item(request.group_by, row) for row in item_rows],
    )
