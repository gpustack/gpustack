from collections import defaultdict
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any, Dict, List

from fastapi import APIRouter
from sqlalchemy import Date, Select, String, and_, asc, cast, desc, literal
from sqlmodel import func, or_, select

from gpustack.api.exceptions import ForbiddenException
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.users import User
from gpustack.schemas.usage import (
    USAGE_GRANULARITY_MONTH,
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_WEEK,
    USAGE_GROUP_BY_API_KEY,
    USAGE_GROUP_BY_DATE,
    USAGE_GROUP_BY_MODEL,
    USAGE_GROUP_BY_USER,
    USAGE_METRIC_API_KEYS_USED,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_INPUT_CACHED_TOKENS,
    USAGE_METRIC_DATE,
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_LAST_ACTIVE,
    USAGE_METRIC_MODELS_CALLED,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_SORT_DESC,
    UsageBreakdownDateDimension,
    UsageBreakdownDimension,
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
from gpustack.utils.usage_snapshots import (
    format_usage_api_key_label,
    format_usage_model_label,
    format_usage_user_label,
)

router = APIRouter()

METRIC_OPTIONS = [
    UsageOption(key=USAGE_METRIC_INPUT_TOKENS, label="Input Tokens"),
    UsageOption(key=USAGE_METRIC_OUTPUT_TOKENS, label="Output Tokens"),
    UsageOption(key=USAGE_METRIC_INPUT_CACHED_TOKENS, label="Input Cached Tokens"),
    UsageOption(key=USAGE_METRIC_TOTAL_TOKENS, label="Total Tokens"),
    UsageOption(key=USAGE_METRIC_API_REQUESTS, label="API Requests"),
]
GRANULARITY_OPTIONS = [
    UsageOption(key=USAGE_GRANULARITY_DAY, label="Day"),
    UsageOption(key=USAGE_GRANULARITY_WEEK, label="Week"),
    UsageOption(key=USAGE_GRANULARITY_MONTH, label="Month"),
]
SELF_GROUP_BY_OPTIONS = [
    UsageOption(key=USAGE_GROUP_BY_DATE, label="Date", scope=["breakdown"]),
    UsageOption(key=USAGE_GROUP_BY_MODEL, label="Model"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
]
ADMIN_GROUP_BY_OPTIONS = [
    UsageOption(key=USAGE_GROUP_BY_DATE, label="Date", scope=["breakdown"]),
    UsageOption(key=USAGE_GROUP_BY_MODEL, label="Model"),
    UsageOption(key=USAGE_GROUP_BY_USER, label="User"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
]


def _null_safe_column_filter(column, value: Any):
    if value is None:
        return column.is_(None)
    return column == value


def _model_key_expression():
    return (
        func.coalesce(ModelUsage.cluster_name + literal("/"), literal(""))
        + ModelUsage.model_name
    )


def _model_identity_expression():
    """Distinct identity for counting models called.

    Include current IDs and snapshot fields so direct deployments, provider
    models, and deleted historical records with the same model name remain
    separate identities.
    """
    return (
        func.coalesce(cast(ModelUsage.model_id, String), literal("deleted"))
        + literal("|")
        + func.coalesce(cast(ModelUsage.provider_id, String), literal("no-provider"))
        + literal("|")
        + func.coalesce(ModelUsage.provider_name, literal(""))
        + literal("|")
        + func.coalesce(ModelUsage.provider_type, literal(""))
        + literal("|")
        + _model_key_expression()
    )


def _metric_columns() -> Dict[str, Any]:
    model_identity = _model_identity_expression()
    return {
        USAGE_METRIC_INPUT_TOKENS: func.coalesce(
            func.sum(ModelUsage.prompt_token_count), 0
        ),
        USAGE_METRIC_OUTPUT_TOKENS: func.coalesce(
            func.sum(ModelUsage.completion_token_count), 0
        ),
        USAGE_METRIC_INPUT_CACHED_TOKENS: func.coalesce(
            func.sum(ModelUsage.prompt_cached_token_count), 0
        ),
        USAGE_METRIC_TOTAL_TOKENS: func.coalesce(
            func.sum(ModelUsage.prompt_token_count + ModelUsage.completion_token_count),
            0,
        ),
        USAGE_METRIC_API_REQUESTS: func.coalesce(func.sum(ModelUsage.request_count), 0),
        USAGE_METRIC_MODELS_CALLED: func.count(func.distinct(model_identity)),
    }


def _date_bucket_expression(session, granularity: str):
    if granularity == USAGE_GRANULARITY_DAY:
        return ModelUsage.date

    dialect = session.get_bind().dialect.name
    if granularity == USAGE_GRANULARITY_WEEK:
        if dialect == "postgresql":
            return cast(func.date_trunc("week", ModelUsage.date), Date)
        if dialect == "mysql":
            return func.subdate(ModelUsage.date, func.weekday(ModelUsage.date))
    if granularity == USAGE_GRANULARITY_MONTH:
        if dialect == "postgresql":
            return cast(func.date_trunc("month", ModelUsage.date), Date)
        if dialect == "mysql":
            return func.str_to_date(
                func.date_format(ModelUsage.date, "%Y-%m-01"), "%Y-%m-%d"
            )

    return ModelUsage.date


def _group_columns(group_by: str, *, date_bucket_expr=None):
    if group_by == USAGE_GROUP_BY_DATE:
        if date_bucket_expr is None:
            date_bucket_expr = ModelUsage.date
        return [date_bucket_expr.label("group_date")], [date_bucket_expr]
    if group_by == USAGE_GROUP_BY_MODEL:
        return [
            ModelUsage.cluster_name.label("group_cluster_name"),
            ModelUsage.model_name.label("group_model_name"),
            ModelUsage.model_id.label("group_model_id"),
            ModelUsage.provider_name.label("group_provider_name"),
            ModelUsage.provider_type.label("group_provider_type"),
            ModelUsage.provider_id.label("group_provider_id"),
        ], [
            ModelUsage.cluster_name,
            ModelUsage.model_name,
            ModelUsage.model_id,
            ModelUsage.provider_name,
            ModelUsage.provider_type,
            ModelUsage.provider_id,
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


def _combined_group_columns(group_bys: List[str], *, date_bucket_expr=None):
    select_columns = []
    select_column_keys = set()
    group_columns = []
    group_column_keys = set()
    for group_by in group_bys:
        current_select_columns, current_group_columns = _group_columns(
            group_by, date_bucket_expr=date_bucket_expr
        )
        for column in current_select_columns:
            key = getattr(column, "key", None) or str(column)
            if key in select_column_keys:
                continue
            select_column_keys.add(key)
            select_columns.append(column)
        for column in current_group_columns:
            key = str(column)
            if key in group_column_keys:
                continue
            group_column_keys.add(key)
            group_columns.append(column)
    return select_columns, group_columns


def _row_identity(group_by: str, row: Any) -> UsageIdentity:
    if group_by == USAGE_GROUP_BY_MODEL:
        model_id = getattr(row, "group_model_id", None)
        provider_id = getattr(row, "group_provider_id", None)
        current = None
        if model_id is not None or provider_id is not None:
            current = UsageIdentityCurrent(
                model_id=model_id,
                provider_id=provider_id,
            )
        return UsageIdentity(
            value=UsageIdentityValue(
                cluster_name=getattr(row, "group_cluster_name", None),
                model_name=getattr(row, "group_model_name", None),
                provider_name=getattr(row, "group_provider_name", None),
                provider_type=getattr(row, "group_provider_type", None),
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


def _row_date_dimension(row: Any, granularity: str) -> UsageBreakdownDateDimension:
    value = _coerce_date(row.group_date)
    return UsageBreakdownDateDimension(value=value, label=value.isoformat())


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def _row_dimension(group_by: str, row: Any) -> UsageBreakdownDimension:
    if group_by == USAGE_GROUP_BY_API_KEY and not getattr(
        row, "group_api_key_name", None
    ):
        return UsageBreakdownDimension(
            identity=None,
            label="-",
            deleted=False,
        )
    identity = _row_identity(group_by, row)
    return UsageBreakdownDimension(
        identity=identity,
        label=_identity_label(group_by, identity),
        deleted=_identity_deleted(identity),
    )


def _identity_key(identity: UsageIdentity) -> tuple:
    return (
        identity.value.cluster_name,
        identity.value.model_name,
        identity.value.provider_name,
        identity.value.provider_type,
        identity.value.user_name,
        identity.value.api_key_name,
        identity.value.access_key,
        identity.value.api_key_is_custom,
        None if identity.current is None else identity.current.model_id,
        None if identity.current is None else identity.current.provider_id,
        None if identity.current is None else identity.current.user_id,
        None if identity.current is None else identity.current.api_key_id,
    )


def _identity_deleted(identity: UsageIdentity) -> bool:
    return identity.current is None


def _identity_label(group_by: str, identity: UsageIdentity) -> str:
    value = identity.value
    if group_by == USAGE_GROUP_BY_MODEL:
        label = format_usage_model_label(
            model_name=value.model_name,
            cluster_name=value.cluster_name,
            provider_name=value.provider_name,
        )
    elif group_by == USAGE_GROUP_BY_USER:
        label = format_usage_user_label(value.user_name)
    else:
        label = format_usage_api_key_label(
            user_name=value.user_name,
            api_key_name=value.api_key_name,
        )

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
                _null_safe_column_filter(ModelUsage.provider_name, value.provider_name),
                _null_safe_column_filter(ModelUsage.provider_type, value.provider_type),
            ]
        )
        if current is None or current.model_id is None:
            conditions.append(ModelUsage.model_id.is_(None))
        else:
            conditions.append(ModelUsage.model_id == current.model_id)
        if current is None or current.provider_id is None:
            conditions.append(ModelUsage.provider_id.is_(None))
        else:
            conditions.append(ModelUsage.provider_id == current.provider_id)
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
    filters,
) -> Select:
    if not user.is_admin:
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
        input_cached_tokens=int(getattr(row, USAGE_METRIC_INPUT_CACHED_TOKENS, 0) or 0),
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


@router.get("/meta", response_model=UsageMetaResponse, response_model_exclude_none=True)
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
    group_by = request.group_by
    group_bys = group_by if isinstance(group_by, list) else [group_by]
    if USAGE_GROUP_BY_USER in group_bys and not user.is_admin:
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

    if request.group_by is None:
        timeline_rows = await _get_rows(
            session,
            scoped_statement.with_only_columns(
                ModelUsage.date.label("date"),
                metric_columns[request.metric].label("value"),
            )
            .group_by(ModelUsage.date)
            .order_by(ModelUsage.date),
        )
        timeline: Dict[date, int] = defaultdict(int)
        for row in timeline_rows:
            point_date = _bucket_date(row.date, request.granularity)
            timeline[point_date] += int(getattr(row, "value", 0) or 0)
        return UsageTimeSeriesResponse(
            summary=_summary_from_row(summary_row),
            metric=request.metric,
            group_by=None,
            granularity=request.granularity,
            series=[
                UsageSeries(
                    identity=None,
                    label="All",
                    deleted=False,
                    timeline=[
                        UsageTimelinePoint(date=item[0], value=item[1])
                        for item in sorted(timeline.items())
                    ],
                )
            ],
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


def _sort_expression(sort_by: str, metric_columns: Dict[str, Any], date_sort_expr=None):
    if sort_by == USAGE_METRIC_DATE:
        return date_sort_expr if date_sort_expr is not None else ModelUsage.date
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


def _order_expression(
    order_by: List[tuple[str, str]], metric_columns: Dict[str, Any], date_sort_expr=None
):
    if not order_by:
        order_by = [(USAGE_METRIC_TOTAL_TOKENS, USAGE_SORT_DESC)]

    sort_exprs = []
    for sort_by, direction in order_by:
        sort_expr = _sort_expression(sort_by, metric_columns, date_sort_expr)
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


def _single_group_by(group_bys: List[str], group_by: str) -> bool:
    return len(group_bys) == 1 and group_bys[0] == group_by


def _breakdown_bucket_granularity(request: UsageBreakdownRequest) -> str:
    if USAGE_GROUP_BY_DATE not in request.group_by:
        return USAGE_GRANULARITY_DAY
    return request.granularity or USAGE_GRANULARITY_DAY


def _build_breakdown_item(
    group_bys: List[str], row: Any, granularity: str
) -> UsageBreakdownItem:
    api_requests = int(getattr(row, USAGE_METRIC_API_REQUESTS, 0) or 0)
    total_tokens = int(getattr(row, USAGE_METRIC_TOTAL_TOKENS, 0) or 0)
    breakdown_item = UsageBreakdownItem(
        input_tokens=int(getattr(row, USAGE_METRIC_INPUT_TOKENS, 0) or 0),
        output_tokens=int(getattr(row, USAGE_METRIC_OUTPUT_TOKENS, 0) or 0),
        input_cached_tokens=int(getattr(row, USAGE_METRIC_INPUT_CACHED_TOKENS, 0) or 0),
        total_tokens=total_tokens,
        api_requests=api_requests,
        avg_tokens_per_request=total_tokens / api_requests if api_requests else 0,
        last_active=getattr(row, USAGE_METRIC_LAST_ACTIVE, None),
    )
    if USAGE_GROUP_BY_DATE in group_bys:
        breakdown_item.date = _row_date_dimension(row, granularity)
    if USAGE_GROUP_BY_MODEL in group_bys:
        breakdown_item.model = _row_dimension(USAGE_GROUP_BY_MODEL, row)
    if USAGE_GROUP_BY_USER in group_bys:
        breakdown_item.user = _row_dimension(USAGE_GROUP_BY_USER, row)
    if USAGE_GROUP_BY_API_KEY in group_bys:
        breakdown_item.api_key = _row_dimension(USAGE_GROUP_BY_API_KEY, row)

    if _single_group_by(group_bys, USAGE_GROUP_BY_USER):
        breakdown_item.models_called = int(
            getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0
        )
        breakdown_item.api_keys_used = int(
            getattr(row, USAGE_METRIC_API_KEYS_USED, 0) or 0
        )
    elif _single_group_by(group_bys, USAGE_GROUP_BY_API_KEY):
        breakdown_item.models_called = int(
            getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0
        )
    return breakdown_item


@router.post(
    "/breakdown",
    response_model=UsageBreakdownResponse,
    response_model_exclude_none=True,
)
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
        filters=request.filters,
    )
    if _single_group_by(request.group_by, USAGE_GROUP_BY_API_KEY):
        base_statement = _exclude_incomplete_api_key_identity(base_statement)
    scoped_statement = _date_scoped_statement(
        base_statement, request.start_date, request.end_date
    )
    granularity = _breakdown_bucket_granularity(request)
    date_bucket_expr = _date_bucket_expression(session, granularity)
    select_columns, group_columns = _combined_group_columns(
        request.group_by, date_bucket_expr=date_bucket_expr
    )
    aggregate_columns = [
        metric_columns[USAGE_METRIC_INPUT_TOKENS].label(USAGE_METRIC_INPUT_TOKENS),
        metric_columns[USAGE_METRIC_OUTPUT_TOKENS].label(USAGE_METRIC_OUTPUT_TOKENS),
        metric_columns[USAGE_METRIC_INPUT_CACHED_TOKENS].label(
            USAGE_METRIC_INPUT_CACHED_TOKENS
        ),
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
    date_sort_expr = (
        date_bucket_expr
        if USAGE_GROUP_BY_DATE in request.group_by
        else func.max(ModelUsage.date)
    )
    sort_exprs = _order_expression(request.order_by, metric_columns, date_sort_expr)
    items_statement = (
        grouped_statement.order_by(*sort_exprs)
        .offset((request.page - 1) * request.perPage)
        .limit(request.perPage)
    )

    total = _row_count_value(await _get_first_row(session, count_statement))
    item_rows = await _get_rows(session, items_statement)

    return UsageBreakdownResponse(
        group_by=request.group_by,
        granularity=granularity if USAGE_GROUP_BY_DATE in request.group_by else None,
        pagination=Pagination(
            page=request.page,
            perPage=request.perPage,
            total=total,
            totalPage=ceil(total / request.perPage) if total else 0,
        ),
        items=[
            _build_breakdown_item(request.group_by, row, granularity)
            for row in item_rows
        ],
    )
