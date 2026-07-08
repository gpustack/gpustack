from datetime import date, datetime
from math import ceil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from sqlalchemy import Date, Select, and_, asc, cast, desc
from sqlmodel import func, or_, select

from gpustack import envs
from gpustack.api.exceptions import ForbiddenException, InvalidException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.users import User
from gpustack.schemas.principals import OrgRole, Principal
from gpustack.schemas.usage import (
    USAGE_GRANULARITY_MONTH,
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_WEEK,
    USAGE_GROUP_BY_API_KEY,
    USAGE_GROUP_BY_DATE,
    USAGE_GROUP_BY_ROUTE,
    USAGE_GROUP_BY_USER,
    USAGE_SCOPE_ALL,
    USAGE_SCOPE_SELF,
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
    UsageSummary,
)
from gpustack.schemas.common import Pagination
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep
from gpustack.utils.usage_snapshots import (
    format_usage_api_key_label,
    format_usage_date_label,
    format_usage_route_label,
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
    UsageOption(key=USAGE_GROUP_BY_DATE, label="Date"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
    UsageOption(key=USAGE_GROUP_BY_ROUTE, label="Route"),
]
ADMIN_GROUP_BY_OPTIONS = [
    UsageOption(key=USAGE_GROUP_BY_DATE, label="Date"),
    UsageOption(key=USAGE_GROUP_BY_USER, label="User"),
    UsageOption(key=USAGE_GROUP_BY_API_KEY, label="API Key"),
    UsageOption(key=USAGE_GROUP_BY_ROUTE, label="Route"),
]


def _null_safe_column_filter(column, value: Any):
    if value is None:
        return column.is_(None)
    return column == value


def _metric_columns() -> Dict[str, Any]:
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
        # "Models called" counts distinct logical model endpoints — i.e.
        # distinct routes the caller hit. Untracked usage (NULL
        # ``model_route_id``) is dropped by COUNT(DISTINCT ...) and shows
        # up only as its own row in the route breakdown.
        USAGE_METRIC_MODELS_CALLED: func.count(
            func.distinct(ModelUsage.model_route_id)
        ),
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
    if group_by == USAGE_GROUP_BY_USER:
        return [
            ModelUsage.user_name.label("group_user_name"),
            ModelUsage.user_id.label("group_user_id"),
        ], [ModelUsage.user_name, ModelUsage.user_id]
    if group_by == USAGE_GROUP_BY_ROUTE:
        return [
            ModelUsage.model_route_name.label("group_model_route_name"),
            ModelUsage.model_route_id.label("group_model_route_id"),
        ], [ModelUsage.model_route_name, ModelUsage.model_route_id]
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
    if group_by == USAGE_GROUP_BY_USER:
        user_id = getattr(row, "group_user_id", None)
        return UsageIdentity(
            value=UsageIdentityValue(user_name=getattr(row, "group_user_name", None)),
            current=None if user_id is None else UsageIdentityCurrent(user_id=user_id),
        )
    if group_by == USAGE_GROUP_BY_ROUTE:
        route_id = getattr(row, "group_model_route_id", None)
        return UsageIdentity(
            value=UsageIdentityValue(
                route_name=getattr(row, "group_model_route_name", None)
            ),
            current=(
                None if route_id is None else UsageIdentityCurrent(route_id=route_id)
            ),
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
    return UsageBreakdownDateDimension(
        value=value, label=format_usage_date_label(value, granularity)
    )


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


# Dimension → the model whose live row proves the referenced entity still
# exists. model_usages is now fully FK-less, so a deleted entity keeps its
# (dangling) id on the row; the read path checks existence here to tag such
# rows ``(Deleted)``. ``model_id`` / ``provider_id`` aren't listed — they have
# no display dimension.
_DIMENSION_ENTITY_MODEL = {
    USAGE_GROUP_BY_USER: Principal,
    USAGE_GROUP_BY_ROUTE: ModelRoute,
    USAGE_GROUP_BY_API_KEY: ApiKey,
}

# Dimension → the aggregated row attribute holding that entity's id, used to
# collect ids for the existence lookup in the breakdown handler.
_DIMENSION_ROW_ID_ATTR = {
    USAGE_GROUP_BY_USER: "group_user_id",
    USAGE_GROUP_BY_ROUTE: "group_model_route_id",
    USAGE_GROUP_BY_API_KEY: "group_api_key_id",
}


def _identity_entity_id(group_by: str, identity: UsageIdentity) -> Optional[int]:
    """The entity id this identity is keyed on (``None`` when unresolvable)."""
    current = identity.current
    if current is None:
        return None
    if group_by == USAGE_GROUP_BY_USER:
        return current.user_id
    if group_by == USAGE_GROUP_BY_ROUTE:
        return current.route_id
    return current.api_key_id


def _row_dimension(
    group_by: str, row: Any, existing_ids: Optional[set] = None
) -> UsageBreakdownDimension:
    if group_by == USAGE_GROUP_BY_API_KEY and not getattr(
        row, "group_api_key_name", None
    ):
        return UsageBreakdownDimension(
            identity=None,
            label="-",
            deleted=False,
        )
    # NULL route_id + NULL name collapses to "Untracked" — exclusively
    # pre-upgrade historical rollup rows once the ingest invariant (every
    # request carries model_route_id) is enforced. Marked deleted=False so the
    # bucket isn't tagged "(Deleted)" — its NULL has a structural cause, not a
    # reference loss.
    if (
        group_by == USAGE_GROUP_BY_ROUTE
        and getattr(row, "group_model_route_id", None) is None
        and not getattr(row, "group_model_route_name", None)
    ):
        return UsageBreakdownDimension(
            identity=None,
            label="Untracked",
            deleted=False,
        )
    identity = _row_identity(group_by, row)
    deleted = _identity_deleted(group_by, identity, existing_ids)
    return UsageBreakdownDimension(
        identity=identity,
        label=_identity_label(group_by, identity),
        deleted=deleted,
    )


def _identity_deleted(
    group_by: str, identity: UsageIdentity, existing_ids: Optional[set] = None
) -> bool:
    # Every id column on model_usages is FK-less, so a deleted user / route /
    # api_key keeps its (now dangling) id on the row rather than nulling out.
    # Resolve deletion by live existence: gone if the id isn't among the
    # entities that still exist. ``existing_ids is None`` means the caller
    # didn't resolve existence, so fall back to the id-present heuristic. A
    # ``None`` id is a legacy SET NULL row whose parent is gone → deleted.
    entity_id = _identity_entity_id(group_by, identity)
    if entity_id is None:
        return True
    if existing_ids is None:
        return False
    return entity_id not in existing_ids


def _identity_label(group_by: str, identity: UsageIdentity) -> str:
    # Pure display name — deletion is carried by the ``deleted`` field and the
    # entity id by ``identity.current``. The client composes any "(Deleted)"
    # marker itself so it stays localizable and so row keys / de-duplication key
    # off the id rather than the (non-unique) name text.
    value = identity.value
    if group_by == USAGE_GROUP_BY_USER:
        return format_usage_user_label(value.user_name)
    if group_by == USAGE_GROUP_BY_ROUTE:
        return format_usage_route_label(value.route_name)
    return format_usage_api_key_label(
        user_name=value.user_name,
        api_key_name=value.api_key_name,
    )


async def _existing_entity_ids(session, model, ids) -> set:
    """The subset of ``ids`` that still resolve to a live ``model`` row.

    model_usages ids are FK-less (kept on parent delete for attribution), so a
    gone entity no longer nulls out — the read path checks existence here to
    tag such rows ``(Deleted)`` while keeping them attributable / filterable.
    Existence is by raw id (no soft-delete filter), matching the old FK
    ``SET NULL`` behavior which only fired on hard delete.
    """
    unique = {i for i in ids if i is not None}
    if not unique:
        return set()
    rows = await _get_rows(session, select(model.id).where(model.id.in_(unique)))
    return set(rows)


async def _existing_ids_for_dimension(session, group_by: str, ids) -> Optional[set]:
    """Resolve which ``ids`` still exist for a display dimension, or ``None``
    for a dimension with no backing entity (e.g. date)."""
    model = _DIMENSION_ENTITY_MODEL.get(group_by)
    if model is None:
        return None
    return await _existing_entity_ids(session, model, ids)


def _filter_condition(group_by: str, item: UsageFilterItem):
    value = item.identity.value
    current = item.identity.current
    conditions = []
    if group_by == USAGE_GROUP_BY_USER:
        conditions.append(
            _null_safe_column_filter(ModelUsage.user_name, value.user_name)
        )
        if current is None or current.user_id is None:
            conditions.append(ModelUsage.user_id.is_(None))
        else:
            conditions.append(ModelUsage.user_id == current.user_id)
    elif group_by == USAGE_GROUP_BY_ROUTE:
        conditions.append(
            _null_safe_column_filter(ModelUsage.model_route_name, value.route_name)
        )
        if current is None or current.route_id is None:
            conditions.append(ModelUsage.model_route_id.is_(None))
        else:
            conditions.append(ModelUsage.model_route_id == current.route_id)
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


def _can_use_all_scope(user: User, ctx) -> bool:
    """The cross-user (``all``) view is meaningful for platform admin
    and real Org owner only. Others fall back to ``self``.

    Personal Org owner doesn't qualify even though their ``org_role``
    is OWNER: a Personal Org has exactly one member, so ``all`` would
    just be ``self`` — and worse, the org_id filter applied for ``all``
    would hide the user's own usage from Platform-shared models, since
    those rows carry the model owner's owner_principal_id, not the
    Personal Org's.
    """
    if user.is_admin:
        return True
    if ctx is None:
        return False
    if ctx.org_role != OrgRole.OWNER:
        return False
    return not ctx.current_is_personal_scope


def _resolve_effective_scope(user: User, ctx, requested_scope: str) -> str:
    """Map requested scope onto what the caller actually gets.

    - ``self`` always allowed.
    - ``all`` allowed for managers / admin; non-managers asking for
      ``all`` are silently downgraded to ``self`` (the request default
      is ``all``, so a regular user with no explicit scope shouldn't
      hit a 403). Privacy-sensitive group_bys (e.g. ``user``) are
      rejected later in ``_check_permission`` once the effective scope
      is locked.
    """
    if requested_scope == USAGE_SCOPE_SELF:
        return USAGE_SCOPE_SELF
    if not _can_use_all_scope(user, ctx):
        return USAGE_SCOPE_SELF
    return USAGE_SCOPE_ALL


def _self_scope_consumer_condition(user_id: int, org_id: int):
    """Consumer-side scoping for the ``self`` view.

    In personal scope (``org_id`` is the caller's own USER-principal) the
    caller's un-attributed rows (``consumer_principal_id IS NULL``) are
    surfaced alongside their personal-Org rows. Cookie-authed direct
    traffic — e.g. the UI Playground — carries no api_key and no
    ``X-Organization-Id``, so its rows land NULL; without this branch the
    strict ``consumer_principal_id == org_id`` equality silently drops the
    user's own usage from their Usage page. When acting inside a real Org
    the strict equality is kept so personal direct usage doesn't bleed
    into the Org view.
    """
    if org_id == user_id:
        return or_(
            ModelUsage.consumer_principal_id == org_id,
            ModelUsage.consumer_principal_id.is_(None),
        )
    return ModelUsage.consumer_principal_id == org_id


def _apply_usage_scope_and_filters(
    statement: Select,
    *,
    user: User,
    filters,
    scope: str,
    org_id: Optional[int] = None,
) -> Select:
    # ``self`` scope always restricts to the caller's own rows and
    # current Org. ``all`` scope restricts to the current Org unless
    # platform admin is in cross-org "All" mode. The Org dimension is
    # the consumer side (API-key owner), which answers "what did this
    # Org spend" — the only question the Usage page surfaces today. A
    # symmetric provider view (rows whose model / deployment is owned
    # by this Org) is deferred until there's an actual product surface
    # for it.
    if scope == USAGE_SCOPE_SELF:
        statement = statement.where(ModelUsage.user_id == user.id)
        if org_id is not None:
            statement = statement.where(_self_scope_consumer_condition(user.id, org_id))
    elif org_id is not None:
        statement = statement.where(ModelUsage.consumer_principal_id == org_id)

    for group_by, items in [
        (USAGE_GROUP_BY_USER, filters.users),
        (USAGE_GROUP_BY_API_KEY, filters.api_keys),
        (USAGE_GROUP_BY_ROUTE, filters.routes),
    ]:
        if not items:
            continue
        if group_by == USAGE_GROUP_BY_USER and scope == USAGE_SCOPE_SELF:
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


def _option_from_identity(
    group_by: str, identity: UsageIdentity, existing_ids: Optional[set] = None
) -> UsageFilterOption:
    # Match the breakdown row's "Untracked" treatment so the filter option
    # list and the row labels stay consistent. NULL route_id + NULL
    # route_name is structural (pre-upgrade data), not a deletion.
    if (
        group_by == USAGE_GROUP_BY_ROUTE
        and identity.value.route_name is None
        and identity.current is None
    ):
        return UsageFilterOption(
            identity=identity,
            label="Untracked",
            deleted=False,
        )
    deleted = _identity_deleted(group_by, identity, existing_ids)
    return UsageFilterOption(
        identity=identity,
        label=_identity_label(group_by, identity),
        deleted=deleted,
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
    identities = [_row_identity(group_by, row) for row in rows]
    # Resolve which referenced entities still exist so gone ones are tagged
    # ``(Deleted)`` — every id is FK-less now, so it no longer nulls out on
    # delete. ``None`` for dimensions with no backing entity.
    existing_ids = await _existing_ids_for_dimension(
        session,
        group_by,
        [_identity_entity_id(group_by, idt) for idt in identities],
    )
    return [
        _option_from_identity(group_by, identity, existing_ids)
        for identity in identities
    ]


@router.get("/meta", response_model=UsageMetaResponse, response_model_exclude_none=True)
async def get_usage_meta(
    session: SessionDep,
    user: CurrentUserDep,
    ctx: TenantContextDep,
    scope: str = USAGE_SCOPE_ALL,
):
    # Default scope is "all"; downgrades to "self" for non-managers so
    # the page works without an explicit scope param.
    if scope == USAGE_SCOPE_ALL and not _can_use_all_scope(user, ctx):
        scope = USAGE_SCOPE_SELF
    elif scope not in (USAGE_SCOPE_SELF, USAGE_SCOPE_ALL):
        raise InvalidException(message=f"Unsupported scope: {scope}")

    base_statement = _base_statement()
    if scope == USAGE_SCOPE_SELF:
        base_statement = base_statement.where(ModelUsage.user_id == user.id)
        if ctx.current_principal_id is not None:
            base_statement = base_statement.where(
                _self_scope_consumer_condition(user.id, ctx.current_principal_id)
            )
    elif ctx.current_principal_id is not None:
        base_statement = base_statement.where(
            ModelUsage.consumer_principal_id == ctx.current_principal_id
        )

    user_options: List[UsageFilterOption] = []
    if scope == USAGE_SCOPE_ALL:
        user_options = await _get_filter_options(
            session, base_statement=base_statement, group_by=USAGE_GROUP_BY_USER
        )
    api_key_options = await _get_filter_options(
        session, base_statement=base_statement, group_by=USAGE_GROUP_BY_API_KEY
    )
    route_options = await _get_filter_options(
        session, base_statement=base_statement, group_by=USAGE_GROUP_BY_ROUTE
    )

    return UsageMetaResponse(
        metrics=METRIC_OPTIONS,
        granularities=GRANULARITY_OPTIONS,
        group_bys=(
            ADMIN_GROUP_BY_OPTIONS
            if scope == USAGE_SCOPE_ALL
            else SELF_GROUP_BY_OPTIONS
        ),
        filters=UsageFilters(
            users=user_options,
            api_keys=api_key_options,
            routes=route_options,
        ),
    )


def _check_permission(user: User, ctx, request, effective_scope: str) -> None:
    """``mine`` view forbids the user-grouping / user-filter dimensions
    (privacy: a user can only see their own rows). ``org`` view allows
    them since the caller is admin / owner / manager."""
    group_by = request.group_by
    group_bys = group_by if isinstance(group_by, list) else [group_by]
    if effective_scope == USAGE_SCOPE_SELF:
        if USAGE_GROUP_BY_USER in group_bys:
            raise ForbiddenException(message="No permission to group by user")
        if request.filters.users:
            raise ForbiddenException(message="No permission to filter by user")


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
    group_bys: List[str],
    row: Any,
    granularity: str,
    existing_ids_by_dim: Optional[Dict[str, set]] = None,
) -> UsageBreakdownItem:
    existing_ids_by_dim = existing_ids_by_dim or {}
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
    if USAGE_GROUP_BY_USER in group_bys:
        breakdown_item.user = _row_dimension(
            USAGE_GROUP_BY_USER, row, existing_ids_by_dim.get(USAGE_GROUP_BY_USER)
        )
    if USAGE_GROUP_BY_API_KEY in group_bys:
        breakdown_item.api_key = _row_dimension(
            USAGE_GROUP_BY_API_KEY,
            row,
            existing_ids_by_dim.get(USAGE_GROUP_BY_API_KEY),
        )
    if USAGE_GROUP_BY_ROUTE in group_bys:
        breakdown_item.route = _row_dimension(
            USAGE_GROUP_BY_ROUTE, row, existing_ids_by_dim.get(USAGE_GROUP_BY_ROUTE)
        )

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
    elif _single_group_by(group_bys, USAGE_GROUP_BY_ROUTE):
        breakdown_item.models_called = int(
            getattr(row, USAGE_METRIC_MODELS_CALLED, 0) or 0
        )
        breakdown_item.api_keys_used = int(
            getattr(row, USAGE_METRIC_API_KEYS_USED, 0) or 0
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
    ctx: TenantContextDep,
    request: UsageBreakdownRequest,
):
    effective_scope = _resolve_effective_scope(user, ctx, request.scope)
    _check_permission(user, ctx, request, effective_scope)

    metric_columns = _metric_columns()
    base_statement = _apply_usage_scope_and_filters(
        _base_statement(),
        user=user,
        filters=request.filters,
        scope=effective_scope,
        org_id=ctx.current_principal_id,
    )
    if _single_group_by(request.group_by, USAGE_GROUP_BY_API_KEY):
        base_statement = _exclude_incomplete_api_key_identity(base_statement)
    scoped_statement = _date_scoped_statement(
        base_statement, request.start_date, request.end_date
    )
    summary_columns = [metric_columns[item].label(item) for item in metric_columns]
    summary_row = await _get_first_row(
        session, scoped_statement.with_only_columns(*summary_columns)
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
    items_statement = grouped_statement.order_by(*sort_exprs)
    # ``page <= 0`` is the project-wide "no pagination" sentinel (mirrors
    # ActiveRecordMixin.page_query): return every bucket unsliced. The trend
    # chart needs the full date series — a token-sorted page would drop the
    # low-traffic (often most recent) buckets and leave gaps in the chart.
    if request.page > 0:
        items_statement = items_statement.offset(
            (request.page - 1) * request.perPage
        ).limit(request.perPage)

    total = _row_count_value(await _get_first_row(session, count_statement))
    # No-pagination (page <= 0) returns the whole series. ``total`` (the bucket
    # count) is already computed above, so reject an over-large result before
    # fetching its rows — narrowing the range / adding filters beats silently
    # truncating, which would reintroduce the chart-gap bug this path fixes.
    if request.page <= 0 and total > envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS:
        raise InvalidException(
            message=(
                f"Result set too large ({total} buckets, limit "
                f"{envs.USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS}). Narrow the "
                "date range or add filters."
            )
        )
    item_rows = await _get_rows(session, items_statement)

    # Resolve which referenced entities still exist so gone ones are tagged
    # ``(Deleted)`` — every id is FK-less now, so a deleted entity keeps its
    # (dangling) id on the row rather than nulling out.
    existing_ids_by_dim: Dict[str, set] = {}
    for dim, id_attr in _DIMENSION_ROW_ID_ATTR.items():
        if dim in request.group_by:
            existing_ids_by_dim[dim] = await _existing_ids_for_dimension(
                session, dim, [getattr(r, id_attr, None) for r in item_rows]
            )

    return UsageBreakdownResponse(
        summary=_summary_from_row(summary_row),
        group_by=request.group_by,
        granularity=granularity if USAGE_GROUP_BY_DATE in request.group_by else None,
        pagination=Pagination(
            page=request.page,
            perPage=request.perPage,
            total=total,
            totalPage=(
                ceil(total / request.perPage)
                if request.page > 0 and total
                else (1 if total else 0)
            ),
        ),
        items=[
            _build_breakdown_item(
                request.group_by, row, granularity, existing_ids_by_dim
            )
            for row in item_rows
        ],
    )
