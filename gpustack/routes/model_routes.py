import logging
import secrets
from sqlalchemy import true
from sqlalchemy.orm import selectinload
from sqlmodel import col, func, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import Any, Callable, List, Optional, Set, Tuple, Union, Dict
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteCreate,
    ModelRouteUpdate,
    ModelRoutePublic,
    ModelRoutesPublic,
    ModelRouteListParams,
    ModelRouteTarget,
    ModelRouteTargetUpdateItem,
    ModelRouteTargetUpdate,
    ModelRouteTargetPublic,
    ModelRouteTargetsPublic,
    ModelRouteTargetListParams,
    SetFallbackTargetInput,
    ModelAuthorizationList,
    ModelAuthorizationUpdate,
    ModelPrincipalAccess,
    ModelPrincipalRef,
    ModelUserAccessExtended,
    MyModel,
    TargetStateEnum,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.principals import platform_principal_id
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.api.tenant import (
    TenantContext,
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.schemas.users import User
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
)
from gpustack.server.services import (
    ModelRouteService,
    revoke_model_access_cache,
)
from gpustack.routes.model_common import (
    build_category_conditions,
    categories_filter,
)

logger = logging.getLogger(__name__)

router = APIRouter()
target_router = APIRouter()
my_models_router = APIRouter()


@router.get("", response_model=ModelRoutesPublic, response_model_exclude_none=True)
async def get_model_routes(
    ctx: TenantContextDep,
    params: ModelRouteListParams = Depends(),
    name: str = None,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    return await _get_model_routes(
        ctx=ctx,
        params=params,
        name=name,
        search=search,
        categories=categories,
    )


def _model_route_grant_conditions(ctx: TenantContext) -> list:
    """OR-set making a ``ModelRoute`` list query mirror what a member of
    the current Org would see — owner-equality PLUS PUBLIC/AUTHED PLUS
    any grant from ``model_route_principals`` that names the current
    principal. Used in place of ``tenant_list_conditions`` on the
    consumption path (admin act-as ``my-models``); the management
    endpoint (``GET /model-routes``) keeps the narrower owner filter so
    it stays a "manage what you own" view.
    """
    if ctx.current_principal_id is None:
        return []
    grant_exists = (
        select(ModelRoutePrincipalLink.id)
        .where(
            ModelRoutePrincipalLink.route_id == ModelRoute.id,
            ModelRoutePrincipalLink.principal_id == ctx.current_principal_id,
            ModelRoutePrincipalLink.deleted_at.is_(None),
        )
        .exists()
    )
    return [
        or_(
            col(ModelRoute.access_policy).in_(
                (AccessPolicyEnum.PUBLIC, AccessPolicyEnum.AUTHED)
            ),
            ModelRoute.owner_principal_id == ctx.current_principal_id,
            grant_exists,
        )
    ]


async def _model_route_visible_to_ctx(
    session: AsyncSession,
    ctx: TenantContext,
    route: ModelRoute,
) -> bool:
    """Single-row mirror of :func:`_model_route_grant_conditions`."""
    if ctx.current_principal_id is None:
        return True
    if route.access_policy in (AccessPolicyEnum.PUBLIC, AccessPolicyEnum.AUTHED):
        return True
    if route.owner_principal_id == ctx.current_principal_id:
        return True
    stmt = (
        select(ModelRoutePrincipalLink.id)
        .where(
            ModelRoutePrincipalLink.route_id == route.id,
            ModelRoutePrincipalLink.principal_id == ctx.current_principal_id,
            ModelRoutePrincipalLink.deleted_at.is_(None),
        )
        .limit(1)
    )
    return (await session.exec(stmt)).first() is not None


def _my_model_visibility_sql(ctx: TenantContext):
    """SQL predicate partitioning ``MyModel`` rows by the caller's context.

    The view emits one row per (user, route, granting-principal) chain.
    Personal scope shows only USER/GROUP-mediated grants plus PUBLIC /
    AUTHED; Org act-as shows only grants tied to that specific Org plus
    PUBLIC / AUTHED. Returns ``None`` when no filtering applies
    (platform admin in All mode shouldn't reach the MyModel path; for
    safety we treat a missing context as no-op).
    """
    if ctx.current_principal_id is None:
        return None
    if ctx.current_is_personal_scope:
        return or_(
            MyModel.via_principal_id.is_(None),
            MyModel.via_principal_kind.in_(("USER", "GROUP")),
        )
    return or_(
        MyModel.via_principal_id.is_(None),
        MyModel.via_principal_id == ctx.current_principal_id,
    )


def _my_model_visibility_predicate(
    ctx: TenantContext,
) -> Optional[Callable[[Any], bool]]:
    """Python mirror of :func:`_my_model_visibility_sql` for the
    streaming path, which can't push WHERE clauses past the event bus.
    """
    if ctx.current_principal_id is None:
        return None
    if ctx.current_is_personal_scope:

        def _ok(data: Any) -> bool:
            return getattr(data, "via_principal_id", None) is None or getattr(
                data, "via_principal_kind", None
            ) in ("USER", "GROUP")

        return _ok
    org_id = ctx.current_principal_id

    def _ok(data: Any) -> bool:
        return (
            getattr(data, "via_principal_id", None) is None
            or getattr(data, "via_principal_id", None) == org_id
        )

    return _ok


async def _fetch_granted_route_ids(ctx: TenantContext) -> Set[int]:
    """Snapshot the set of route IDs granted to ``current_principal_id``
    via ``model_route_principals``. Used by the streaming path to keep
    the SSE filter sync; the snapshot can drift if grants change while
    a stream is open, but ``ModelRoute`` events don't fire on grant
    edits either, so clients reconnect to refresh.
    """
    async with async_session() as session:
        return set(
            (
                await session.exec(
                    select(ModelRoutePrincipalLink.route_id).where(
                        ModelRoutePrincipalLink.principal_id
                        == ctx.current_principal_id,
                        ModelRoutePrincipalLink.deleted_at.is_(None),
                    )
                )
            ).all()
        )


async def _build_route_grant_stream_filter(
    ctx: Optional[TenantContext],
    target_class,
    include_grants: bool,
) -> Optional[Callable[[Any], bool]]:
    """Streaming counterpart to :func:`_model_route_grant_conditions`.
    Returns ``None`` when no grant filtering applies (non-grants path,
    non-ModelRoute target, or admin "All" mode without a principal).
    """
    if (
        ctx is None
        or target_class is not ModelRoute
        or not include_grants
        or ctx.current_principal_id is None
    ):
        return None
    granted_route_ids = await _fetch_granted_route_ids(ctx)
    return _model_route_grant_predicate(ctx, granted_route_ids)


def _model_route_grant_predicate(
    ctx: TenantContext, granted_route_ids: Set[int]
) -> Callable[[Any], bool]:
    """Python mirror of :func:`_model_route_grant_conditions` for the
    streaming path. ``granted_route_ids`` is snapshotted at stream start
    — ``ModelRoute`` events don't fire on ``model_route_principals``
    changes, so a mid-stream grant edit isn't observable here either
    way; clients reconnect to refresh.
    """
    current_principal_id = ctx.current_principal_id

    def _ok(data: Any) -> bool:
        if getattr(data, "access_policy", None) in (
            AccessPolicyEnum.PUBLIC,
            AccessPolicyEnum.AUTHED,
        ):
            return True
        if getattr(data, "owner_principal_id", None) == current_principal_id:
            return True
        return getattr(data, "id", None) in granted_route_ids

    return _ok


async def _get_model_routes(
    params: ModelRouteListParams,
    name: str = None,
    search: str = None,
    categories: Optional[List[str]] = None,
    user_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
    target_class: Union[ModelRoute, MyModel] = ModelRoute,
    ctx: Optional[TenantContext] = None,
    include_grants: bool = False,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {'deleted_at': None}
    if name:
        fields = {"name": name}

    if user_id is not None:
        fields["user_id"] = user_id
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id

    # Apply tenant scoping to the streaming path too. Skipped for the MyModel
    # view which handles visibility through its own SQL view definition,
    # and for the grants-inclusive consumption path which needs an OR
    # against ``model_route_principals`` that a single field-equality
    # can't express.
    if (
        ctx is not None
        and target_class is ModelRoute
        and not include_grants
        and ctx.current_principal_id is not None
        and "owner_principal_id" not in fields
    ):
        fields["owner_principal_id"] = ctx.current_principal_id

    my_model_sql_filter = None
    my_model_stream_filter: Optional[Callable[[Any], bool]] = None
    if target_class is MyModel and ctx is not None:
        my_model_sql_filter = _my_model_visibility_sql(ctx)
        my_model_stream_filter = _my_model_visibility_predicate(ctx)

    if params.watch:
        # Snapshot grant-mediated route ids only on the streaming path —
        # the paginated branch below pushes the same predicate into SQL
        # via ``_model_route_grant_conditions`` and doesn't need the
        # extra round-trip.
        route_grant_stream_filter = await _build_route_grant_stream_filter(
            ctx, target_class, include_grants
        )

        def _stream_filter(data: Any) -> bool:
            if my_model_stream_filter is not None and not my_model_stream_filter(data):
                return False
            if route_grant_stream_filter is not None and not route_grant_stream_filter(
                data
            ):
                return False
            return categories_filter(data, categories)

        return StreamingResponse(
            target_class.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=_stream_filter,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions: list = []
        # Apply tenant scoping when caller passed a TenantContext. Per-user
        # visibility for ModelRoute is via the model_route_principals table.
        if ctx is not None and target_class is ModelRoute:
            if include_grants:
                extra_conditions.extend(_model_route_grant_conditions(ctx))
            else:
                extra_conditions.extend(tenant_list_conditions(ctx, ModelRoute))
        my_model_sql_filter = _append_my_model_conditions(
            extra_conditions, target_class, my_model_sql_filter
        )
        if my_model_sql_filter is not None:
            extra_conditions.append(my_model_sql_filter)
        if categories:
            conditions = build_category_conditions(session, target_class, categories)
            extra_conditions.append(or_(*conditions))

        return await target_class.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
            extra_conditions=extra_conditions,
        )


def _append_my_model_conditions(extra_conditions, target_class, visibility_filter):
    """Push the MyModel dedup subquery into ``extra_conditions`` and
    return ``None`` to signal the visibility filter has been absorbed
    into the subquery (don't double-apply on the outer query). For
    non-MyModel targets, return ``visibility_filter`` unchanged so the
    caller's downstream branch handles it as before.
    """
    if target_class is not MyModel:
        return visibility_filter
    extra_conditions.append(_my_model_dedup_condition(visibility_filter))
    return None


def _my_model_dedup_condition(visibility_filter):
    """Collapse multi-chain ``(user, route)`` rows down to one in the
    MyModel list path. The view emits one row per (user, route,
    granting-principal) chain by contract; without this, ``COUNT(*)``
    counts chains instead of routes and the page slice can drop chains
    arbitrarily.

    Ranking happens inside the visibility-filtered chain set —
    otherwise ``rn=1`` could land on a chain the caller's scope would
    have dropped, hiding the route entirely. ``MIN(via_principal_id)``
    is used as a deterministic tiebreak; the surviving ``via_*`` is
    arbitrary among the visible chains, matching the agreed semantics
    of "one row per route, via 任取一条". ``_get_model_route`` orders
    by the same key so detail and list agree on which chain to surface.
    """
    ranked = (
        select(
            MyModel.pid,
            func.row_number()
            .over(
                partition_by=[MyModel.id, MyModel.user_id],
                order_by=[col(MyModel.via_principal_id).asc()],
            )
            .label("rn"),
        )
        .where(visibility_filter if visibility_filter is not None else true())
        .subquery()
    )
    return col(MyModel.pid).in_(select(ranked.c.pid).where(ranked.c.rn == 1))


@router.get("/{id}", response_model=ModelRoutePublic, response_model_exclude_none=True)
async def get_model_route(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return await _get_model_route(session=session, id=id, ctx=ctx)


async def _get_model_route(
    session: AsyncSession,
    id: int,
    target_class: Union[ModelRoute, MyModel] = ModelRoute,
    user_id: Optional[int] = None,
    owner_principal_id: Optional[int] = None,
    ctx: Optional[TenantContext] = None,
    include_grants: bool = False,
):
    fields = {"id": id}
    if user_id is not None:
        fields["user_id"] = user_id
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id

    # The MyModel view emits one row per (user, route, granting-principal)
    # chain; ``one_by_fields`` would .first() back an arbitrary row that
    # might be filtered out by the caller's context. Push the same
    # visibility predicate as the list path into the query, and order
    # by ``via_principal_id`` so the chain we surface here matches the
    # one ``_get_model_routes`` picked via ``ROW_NUMBER() ... ORDER BY
    # via_principal_id ASC`` (see :func:`_my_model_dedup_condition`).
    if target_class is MyModel and ctx is not None:
        vis = _my_model_visibility_sql(ctx)
        stmt = select(MyModel)
        for key, value in fields.items():
            stmt = stmt.where(getattr(MyModel, key) == value)
        if vis is not None:
            stmt = stmt.where(vis)
        stmt = stmt.order_by(col(MyModel.via_principal_id).asc())
        existing = (await session.exec(stmt.limit(1))).first()
    else:
        existing = await target_class.one_by_fields(
            session=session,
            fields=fields,
        )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelAccess with id '{id}' not found.")
    if ctx is not None and target_class is ModelRoute:
        if include_grants:
            if not await _model_route_visible_to_ctx(session, ctx, existing):
                raise NotFoundException(f"ModelAccess with id '{id}' not found.")
        else:
            assert_resource_visible(
                ctx,
                existing,
                not_found_message=f"ModelAccess with id '{id}' not found.",
            )
    return existing


@router.post("", response_model=ModelRoutePublic, response_model_exclude_none=True)
async def create_model_route(
    session: SessionDep, ctx: TenantContextDep, input: ModelRouteCreate
):
    # Names are unique within their owning Org. The gateway emits an
    # owner-name prefix as the effective model name for non-platform
    # Orgs, so two Orgs can each have a route called "qwen3-0.6b"
    # without colliding in the AI proxy match rules.
    #
    # Resolve owner from the current Org context, falling back to the
    # platform Org for admin "All" mode. Routes are an Org-owned
    # resource — using the admin's USER-principal here (as
    # ``target_principal_id_for_write`` would) misaligns the route from
    # its targets, since Models default to the platform Org and trip
    # the cross-Org check in ``_assert_target_tenant_aligned``.
    target_org_id = ctx.current_principal_id or platform_principal_id()
    existing = await ModelRoute.one_by_fields(
        session,
        {
            'deleted_at': None,
            "name": input.name,
            "owner_principal_id": target_org_id,
        },
    )
    if existing:
        raise AlreadyExistsException(
            f"ModelRoute with name '{input.name}' already exists."
        )
    source = input.model_dump(exclude={"targets"})
    targets = input.targets or []
    await validate_targets(session, targets, route_owner_principal_id=target_org_id)
    source["targets"] = len(targets)
    source["owner_principal_id"] = target_org_id

    # Multi-tenant default: a non-platform Org's new route is scoped to
    # that Org via ALLOWED_PRINCIPALS with the owning Org auto-granted
    # below — `non_admin_user_models` matches it through the Org grant
    # in `model_route_principals`. The Default (platform) Org keeps
    # AUTHED — admin's shared catalog stays visible to every
    # authenticated user, and existing routes migrated to the platform
    # Org must keep working. Caller's explicit `access_policy` always
    # wins (and then manages its own principal grants via /principals).
    owner_org_id = source.get("owner_principal_id")
    is_platform_org = owner_org_id == platform_principal_id()
    org_scoped_default = (
        not is_platform_org
        and owner_org_id is not None
        and "access_policy" not in input.model_fields_set
    )
    if org_scoped_default:
        source["access_policy"] = AccessPolicyEnum.ALLOWED_PRINCIPALS

    try:
        route: ModelRoute = await ModelRoute.create(
            session=session, source=source, auto_commit=False
        )
        await create_model_route_targets(
            session=session,
            route_id=route.id,
            route_name=route.name,
            targets=targets,
            auto_commit=False,
        )
        # Auto-grant the owning Org so the defaulted ALLOWED_PRINCIPALS
        # route is visible to its members out of the box. Users can add
        # or remove principals afterward via /principals — the Org grant
        # is an ordinary row, not special-cased.
        if org_scoped_default:
            session.add(
                ModelRoutePrincipalLink(
                    route_id=route.id,
                    principal_id=owner_org_id,
                )
            )
        await session.commit()
        await session.refresh(route)
        await revoke_model_access_cache(session=session)
        return route
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(
            f"Failed to create ModelAccess '{input.name}': {e}"
        )


@router.put("/{id}", response_model=ModelRoutePublic, response_model_exclude_none=True)
async def update_model_route(
    id: int,
    session: SessionDep,
    ctx: TenantContextDep,
    input: ModelRouteUpdate,
):
    existing = await ModelRoute.one_by_id(
        session=session,
        id=id,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRoute with id '{id}' not found.")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"ModelRoute with id '{id}' not found.",
    )
    # Names are unique within their owning Org (effective name on the
    # gateway side carries the owner-name prefix for non-platform
    # Orgs).
    duplicated_name = await ModelRoute.one_by_fields(
        session,
        {
            'deleted_at': None,
            "name": input.name,
            "owner_principal_id": existing.owner_principal_id,
        },
    )
    if duplicated_name and duplicated_name.id != id:
        raise AlreadyExistsException(
            f"ModelRoute with name '{input.name}' already exists."
        )
    existing_name = existing.name
    input_name = input.name
    input_data = input.model_dump(exclude={"targets"}, include=input.model_fields_set)
    try:
        if input.targets is not None or input.name != existing.name:
            target_count, _ = await batch_handle_targets(
                session=session,
                route_id=existing.id,
                route_name=existing.name,
                targets=input.targets,
                auto_commit=False,
                new_route_name=input.name if input.name != existing.name else None,
                route_owner_principal_id=existing.owner_principal_id,
            )
            input_data["targets"] = target_count
        await ModelRouteService(session).update(
            existing, source=input_data, auto_commit=False
        )
        await session.commit()
        if existing_name != input_name:
            await revoke_model_access_cache(session=session)
    except Exception as e:
        raise InternalServerErrorException(f"Failed to update ModelRoute '{id}': {e}")
    return await ModelRoute.one_by_id(session=session, id=id)


@router.delete("/{id}")
async def delete_model_route(
    id: int,
    session: SessionDep,
    ctx: TenantContextDep,
):
    existing = await ModelRoute.one_by_id(
        session=session,
        id=id,
        options=[selectinload(ModelRoute.route_targets)],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRoute with id '{id}' not found.")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"ModelRoute with id '{id}' not found.",
    )
    try:
        await revoke_model_access_cache(session=session, model=existing)
        await ModelRouteService(session).delete(existing)
    except Exception as e:
        raise InternalServerErrorException(f"Failed to delete ModelRoute '{id}': {e}")


async def unset_fallback_target(
    session: AsyncSession,
    route_id: int,
    auto_commit: bool = False,
):
    targets = await ModelRouteTarget.all_by_field(
        session=session,
        field="route_id",
        value=route_id,
        for_update=True,
    )
    for target in targets:
        if target.fallback_status_codes is not None and target.deleted_at is None:
            target.fallback_status_codes = None
            await target.update(session=session, auto_commit=auto_commit)


@router.post(
    "/{id}/add-targets",
    response_model=List[ModelRouteTargetPublic],
    response_model_exclude_none=True,
)
async def add_model_route_targets(
    id: int,
    session: SessionDep,
    targets: List[ModelRouteTargetUpdateItem],
):
    route = await ModelRoute.one_by_id(session=session, id=id)
    if not route or route.deleted_at is not None:
        raise NotFoundException(f"ModelRoute with id '{id}' not found.")
    target_count, created_targets = await batch_handle_targets(
        session=session,
        route_id=route.id,
        route_name=route.name,
        new_route_name=None,
        route_owner_principal_id=route.owner_principal_id,
        targets=targets,
        auto_commit=False,
    )
    try:
        route.targets = target_count
        await ModelRouteService(session=session).update(route, auto_commit=True)
        await session.commit()
        for target in created_targets:
            await session.refresh(target)
        return created_targets
    except Exception as e:
        raise InternalServerErrorException(
            f"Failed to add targets to ModelRoute '{id}': {e}"
        )


async def batch_handle_targets(
    session: AsyncSession,
    route_id: int,
    route_name: str,
    targets: List[ModelRouteTargetUpdateItem],
    auto_commit: bool = True,
    new_route_name: Optional[str] = None,
    route_owner_principal_id: Optional[int] = None,
) -> Tuple[int, List[ModelRouteTarget]]:
    existing_targets = await ModelRouteTarget.all_by_field(
        session=session,
        field="route_id",
        value=route_id,
        for_update=True,
    )
    target_count = len(existing_targets)
    existing_target_map = {target.id: target for target in existing_targets}
    invalid_target_ids = [
        target.id
        for target in targets
        if target.id is not None and target.id not in existing_target_map
    ]
    if len(invalid_target_ids) > 0:
        raise NotFoundException(
            f"ModelRouteTargets with ids '{', '.join(map(str, invalid_target_ids))}' not found."
        )
    target_count += len([target for target in targets if target.id is None])

    to_delete_target_ids = [
        target.id
        for target in existing_targets
        if target.id not in [e.id for e in targets if e.id is not None]
    ]
    target_count -= len(to_delete_target_ids)

    fallback_index = await validate_targets(
        session=session,
        targets=targets,
        route_owner_principal_id=route_owner_principal_id,
    )
    if fallback_index is not None:
        fallback_target = targets[fallback_index]
        if fallback_target.id is None:
            await unset_fallback_target(session, route_id, auto_commit=auto_commit)

    targets_to_return = []
    try:
        # Delete
        for target_id in to_delete_target_ids:
            target = existing_target_map[target_id]
            await target.delete(session=session, auto_commit=auto_commit)

        # Update
        updated_targets = await update_model_route_targets(
            session=session,
            targets=targets,
            existing_target_map=existing_target_map,
            new_route_name=new_route_name,
            auto_commit=auto_commit,
        )
        targets_to_return.extend(updated_targets)

        # Create
        created_targets = await create_model_route_targets(
            session=session,
            route_id=route_id,
            route_name=new_route_name or route_name,
            targets=targets,
            auto_commit=auto_commit,
        )
        targets_to_return.extend(created_targets)
    except Exception as e:
        raise InternalServerErrorException(
            f"Failed to batch handle ModelRouteTargets: {e}"
        )

    return target_count, targets_to_return


async def update_model_route_targets(
    session: AsyncSession,
    targets: List[ModelRouteTargetUpdateItem],
    existing_target_map: Dict[int, ModelRouteTarget],
    new_route_name: Optional[str] = None,
    auto_commit: bool = False,
) -> List[ModelRouteTarget]:
    to_update_target_map: Dict[int, ModelRouteTargetUpdateItem] = {
        target.id: target
        for target in targets
        if target.id is not None and target.id in existing_target_map
    }
    targets_to_return = []
    for id, existing_target in existing_target_map.items():
        if new_route_name is None and id not in to_update_target_map:
            continue
        to_compare_fields = {
            "route_name",
            "overridden_model_name",
            "weight",
            "model_id",
            "provider_id",
            "fallback_status_codes",
        }
        existing_dict = existing_target.model_dump(
            include=to_compare_fields, exclude_none=True
        )
        input_target = to_update_target_map.get(id, None)
        input_dict = {**existing_dict}
        if input_target is not None:
            input_dict.update(
                input_target.model_dump(include=to_compare_fields, exclude_none=True)
            )
        if new_route_name is not None:
            input_dict["route_name"] = new_route_name
        update_source = {}
        if existing_dict != input_dict:
            # set state to UNAVAILABLE to force re-validate on next use
            update_source.update(
                {
                    **input_dict,
                    "state": TargetStateEnum.UNAVAILABLE,
                }
            )
        if len(update_source) > 0:
            updated = await existing_target.update(
                session=session, source=update_source, auto_commit=auto_commit
            )
            targets_to_return.append(updated)

    return targets_to_return


async def create_model_route_targets(
    session: AsyncSession,
    route_id: int,
    route_name: str,
    targets: List[ModelRouteTargetUpdateItem],
    auto_commit: bool = True,
) -> List[ModelRouteTarget]:
    created_targets = []
    for target in targets:
        if target.id is not None:
            continue
        route_target = ModelRouteTarget.model_validate(
            {
                **target.model_dump(),
                "route_id": route_id,
                "name": route_name + "-" + secrets.token_hex(5),
                "route_name": route_name,
            }
        )
        if route_target.model_id is not None:
            route_target.state = TargetStateEnum.UNAVAILABLE
        route_target: ModelRouteTarget = await ModelRouteTarget.create(
            session=session, source=route_target, auto_commit=auto_commit
        )
        created_targets.append(route_target)
    if auto_commit:
        await session.commit()
        for target in created_targets:
            await session.refresh(target)
    return created_targets


def _assert_target_tenant_aligned(
    route_owner_principal_id: Optional[int],
    target_owner_principal_id: Optional[int],
    target_kind: str,
    target_id: int,
) -> None:
    """Routes are GPUStack's only cross-tenant sharing surface — a
    deployment (Model / ModelProvider) has no permission primitive of
    its own, so it lives within one Org's boundary and must only be
    referenced from routes in the same Org. Cross-Org targeting that
    bypasses this is a misconfiguration: usage attribution
    (``ModelUsage.owner_principal_id``) is sourced from the model's
    owner, so a cross-Org target silently drifts the row's tenant scope
    away from the route's caller.

    A NULL target owner (e.g. legacy provider rows that predate
    multi-tenancy) is treated as global and allowed everywhere — the
    strict rule kicks in only when the target explicitly carries an
    owner. A route with no owner (also legacy / platform fallback) is
    similarly skipped.
    """
    if route_owner_principal_id is None:
        return
    if target_owner_principal_id is None:
        return
    if target_owner_principal_id == route_owner_principal_id:
        return
    raise InvalidException(
        f"{target_kind} {target_id} belongs to principal "
        f"{target_owner_principal_id}; a route owned by principal "
        f"{route_owner_principal_id} may only target resources in the "
        f"same Org. Cross-Org sharing must be done by exposing this Org's "
        f"own route (via access policy), not by retargeting across Orgs."
    )


async def validate_targets(
    session: SessionDep,
    targets: List[ModelRouteTargetUpdateItem],
    route_owner_principal_id: Optional[int] = None,
) -> Optional[int]:
    fallback_index: Optional[int] = None
    for index, target in enumerate(targets):
        if (
            target.fallback_status_codes is not None
            and len(target.fallback_status_codes) > 0
        ):
            if fallback_index is not None:
                raise InvalidException(
                    "Only one target can be set as fallback for status codes."
                )
            fallback_index = index
        if target.provider_id is not None:
            provider = await ModelProvider.one_by_id(
                session=session, id=target.provider_id
            )
            if provider is None or provider.deleted_at is not None:
                raise NotFoundException(
                    f"ModelProvider with id '{target.provider_id}' not found."
                )
            validate_provider_model_name(provider, target.overridden_model_name)
            _assert_target_tenant_aligned(
                route_owner_principal_id,
                getattr(provider, "owner_principal_id", None),
                "ModelProvider",
                target.provider_id,
            )
        elif target.model_id is not None:
            model = await Model.one_by_id(session=session, id=target.model_id)
            if model is None or model.deleted_at is not None:
                raise NotFoundException(f"Model with id '{target.model_id}' not found.")
            _assert_target_tenant_aligned(
                route_owner_principal_id,
                getattr(model, "owner_principal_id", None),
                "Model",
                target.model_id,
            )
    return fallback_index


def validate_provider_model_name(
    provider: ModelProvider,
    model_name: str,
):
    supported_models = provider.models or []
    model_names = [model.name for model in supported_models]
    if model_name not in model_names:
        raise InvalidException(
            f"overridden_model_name '{model_name}' is not supported by provider '{provider.name}'. Supported models: {', '.join(model_names)}"
        )


@target_router.get(
    "", response_model=ModelRouteTargetsPublic, response_model_exclude_none=True
)
async def get_model_route_targets(
    session: SessionDep,
    params: ModelRouteTargetListParams = Depends(),
    name: str = None,
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {'deleted_at': None}
    if name:
        fields = {"name": name}

    ext_fields = params.model_dump(
        include={
            "route_id",
            "route_name",
            "model_id",
            "provider_id",
        },
        exclude_none=True,
    )
    fields.update(ext_fields)

    if params.watch:
        return StreamingResponse(
            ModelRouteTarget.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await ModelRouteTarget.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )


@target_router.put(
    "/{id}",
    response_model=ModelRouteTargetPublic,
    response_model_exclude_none=True,
)
async def update_model_route_target(
    id: int,
    session: SessionDep,
    input: ModelRouteTargetUpdate,
):
    existing = await ModelRouteTarget.one_by_id(
        session=session,
        id=id,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRouteTarget with id '{id}' not found.")
    # Resolve the owning route's tenant so validate_targets can enforce
    # tenant alignment — a target swap (e.g. pointing at a different
    # model) must still satisfy "route and target share an Org".
    parent_route = await ModelRoute.one_by_id(session=session, id=existing.route_id)
    if parent_route is None:
        raise NotFoundException(f"ModelRoute with id '{existing.route_id}' not found.")
    route_owner_principal_id = parent_route.owner_principal_id
    # don't need to update fallback_status_codes here, handled in set-fallback target
    targets = [
        ModelRouteTargetUpdateItem.model_validate(
            {
                **input.model_dump(),
                "id": id,
                "fallback_status_codes": existing.fallback_status_codes,
            }
        )
    ]
    await validate_targets(
        session, targets, route_owner_principal_id=route_owner_principal_id
    )
    try:
        await update_model_route_targets(
            session=session,
            targets=targets,
            existing_target_map={id: existing},
            auto_commit=True,
        )
    except Exception as e:
        raise InternalServerErrorException(
            f"Failed to update ModelRouteTarget '{id}': {e}"
        )
    return await ModelRouteTarget.one_by_id(session=session, id=id)


@target_router.delete("/{id}")
async def delete_model_route_target(
    id: int,
    session: SessionDep,
):
    existing = await ModelRouteTarget.one_by_id(
        session=session,
        id=id,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRouteTarget with id '{id}' not found.")
    route = existing.model_route
    try:
        await existing.delete(session=session, auto_commit=False)
        if route:
            route.targets = max(0, route.targets - 1)
            await ModelRouteService(session=session).update(route, auto_commit=False)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(
            f"Failed to delete ModelRouteTarget '{id}': {e}"
        )


@target_router.post(
    "/{id}/set-fallback",
    response_model=ModelRouteTargetPublic,
    response_model_exclude_none=True,
)
async def set_fallback_target(
    id: int,
    session: SessionDep,
    input: SetFallbackTargetInput,
):
    existing = await ModelRouteTarget.one_by_id(
        session=session,
        id=id,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRouteTarget with id '{id}' not found.")
    if existing.fallback_status_codes == input.fallback_status_codes:
        return existing
    try:
        if input.fallback_status_codes is not None:
            await unset_fallback_target(session, existing.route_id, auto_commit=False)
        existing.fallback_status_codes = input.fallback_status_codes
        await existing.update(session=session, auto_commit=False)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(
            f"Failed to set fallback status codes for ModelRouteTarget '{id}': {e}"
        )
    return await ModelRouteTarget.one_by_id(session=session, id=id)


async def _list_route_users(session, route_id: int) -> List[ModelUserAccessExtended]:
    """Build the OSS-facing access list for a route.

    User-only ACL rows live in ``model_route_principals`` with a
    ``principal_id`` referencing a USER-kind principal; we hop through
    ``principals`` to ``users`` so the response can carry display-only
    fields (``username`` / ``full_name`` / ``avatar_url``) without an
    extra round trip from the client.
    """
    # USER and Principal are the same table post-consolidation, so the
    # link join can go directly through ``User.id``.
    stmt = (
        select(User, ModelRoutePrincipalLink)
        .join(
            ModelRoutePrincipalLink,
            ModelRoutePrincipalLink.principal_id == User.id,
        )
        .where(
            ModelRoutePrincipalLink.route_id == route_id,
            User.kind == PrincipalType.USER,
        )
    )
    rows = (await session.exec(stmt)).all()
    return [
        ModelUserAccessExtended(
            id=user.id,
            username=user.name,
            full_name=user.display_name,
            avatar_url=user.avatar_url,
        )
        for user, _ in rows
    ]


async def _replace_route_user_principals(
    session, route_id: int, user_ids: List[int]
) -> None:
    """Replace the user-grant rows on a route with exactly ``user_ids``.

    Touches only rows whose principal is USER-kind — org / group
    grants attached via the ``ALLOWED_PRINCIPALS`` flow are left
    alone, even if this endpoint is called on the same route.
    """
    # After identity consolidation a USER's principal id IS the user's
    # id — no lookup needed.
    desired_principal_ids: Set[int] = set(user_ids) if user_ids else set()

    existing_stmt = (
        select(ModelRoutePrincipalLink)
        .join(Principal, Principal.id == ModelRoutePrincipalLink.principal_id)
        .where(
            ModelRoutePrincipalLink.route_id == route_id,
            Principal.kind == PrincipalType.USER,
        )
    )
    existing = list((await session.exec(existing_stmt)).all())
    existing_by_principal = {row.principal_id: row for row in existing}

    for principal_id, row in existing_by_principal.items():
        if principal_id not in desired_principal_ids:
            await session.delete(row)

    for principal_id in desired_principal_ids:
        if principal_id in existing_by_principal:
            continue
        session.add(
            ModelRoutePrincipalLink(
                route_id=route_id,
                principal_id=principal_id,
            )
        )


async def _list_route_principals(session, route_id: int) -> List[ModelPrincipalAccess]:
    """Every principal grant on a route (any kind), with the principal's
    name / display_name joined for display."""
    rows = list(
        (
            await session.exec(
                select(ModelRoutePrincipalLink).where(
                    ModelRoutePrincipalLink.route_id == route_id,
                    ModelRoutePrincipalLink.deleted_at.is_(None),
                )
            )
        ).all()
    )
    if not rows:
        return []
    principal_ids = {r.principal_id for r in rows}
    result = await session.exec(
        select(Principal).where(Principal.id.in_(principal_ids))
    )
    by_id = {p.id: p for p in result.all()}
    out: List[ModelPrincipalAccess] = []
    for r in rows:
        p = by_id.get(r.principal_id)
        out.append(
            ModelPrincipalAccess(
                principal_type=p.kind if p else PrincipalType.USER,
                principal_id=r.principal_id,
                principal_name=p.name if p else None,
                principal_display_name=p.display_name if p else None,
            )
        )
    return out


async def _validate_principals(session, principals: List[ModelPrincipalRef]) -> None:
    """Each ref must name an existing principal whose kind matches. Raises
    InvalidException otherwise. SYSTEM principals fail the kind check (no
    caller asks for kind=SYSTEM in an ACL grant)."""
    for ref in principals:
        target = await Principal.one_by_id(session, ref.principal_id)
        if not target or target.deleted_at is not None:
            raise InvalidException(message=f"Principal {ref.principal_id} not found")
        if target.kind != ref.principal_type:
            raise InvalidException(
                message=(
                    f"Principal {ref.principal_id} is a {target.kind.value}, "
                    f"not a {ref.principal_type.value}"
                )
            )


async def _replace_route_principals(
    session, route_id: int, principal_ids: List[int]
) -> None:
    """Replace the route's entire principal grant set with exactly
    ``principal_ids`` (any kind). Callers validate the refs first."""
    desired: Set[int] = set(principal_ids)

    existing = list(
        (
            await session.exec(
                select(ModelRoutePrincipalLink).where(
                    ModelRoutePrincipalLink.route_id == route_id,
                    ModelRoutePrincipalLink.deleted_at.is_(None),
                )
            )
        ).all()
    )
    existing_by_principal = {row.principal_id: row for row in existing}

    for principal_id, row in existing_by_principal.items():
        if principal_id not in desired:
            await session.delete(row)

    for principal_id in desired:
        if principal_id in existing_by_principal:
            continue
        session.add(
            ModelRoutePrincipalLink(route_id=route_id, principal_id=principal_id)
        )


@router.get("/{id}/access", response_model=ModelAuthorizationList)
async def get_model_authorization_list(session: SessionDep, id: int):
    model: ModelRoute = await ModelRoute.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return ModelAuthorizationList(
        items=await _list_route_users(session, id),
        principals=await _list_route_principals(session, id),
        access_policy=model.access_policy,
    )


@router.post("/{id}/access", response_model=ModelAuthorizationList)
async def add_model_authorization(
    session: SessionDep, id: int, access_request: ModelAuthorizationUpdate
):
    model = await ModelRoute.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    # Two mutually exclusive grant surfaces (else = "don't touch grants",
    # e.g. a plain policy switch):
    #   * ``principals`` (preferred) replaces the FULL grant set, any
    #     kind. An empty list clears all grants.
    #   * ``users`` (deprecated) replaces only USER-kind grants.
    replace_principals = access_request.principals is not None
    replace_users = not replace_principals and access_request.users is not None

    # Validate up front so bad refs surface as 4xx, not the 500 wrapper
    # around the mutation block below.
    requested_user_ids = [u.id for u in (access_request.users or [])]
    if replace_principals:
        await _validate_principals(session, access_request.principals)
    elif replace_users and requested_user_ids:
        users = await User.all_by_fields(
            session=session,
            fields={},
            extra_conditions=[col(User.id).in_(requested_user_ids)],
        )
        existing_user_ids = {u.id for u in users}
        for req_id in requested_user_ids:
            if req_id not in existing_user_ids:
                raise NotFoundException(message=f"User ID {req_id} not found")

    # Cache invalidation. The USER-list path can scope to the affected
    # users (previously + newly granted). Replacing arbitrary principals
    # (org / group) widens visibility unpredictably, so invalidate
    # broadly — as does any access_policy change.
    affected_user_ids: Optional[Set[int]] = None
    cache_model: Optional[ModelRoute] = None
    if replace_users:
        previous_users = await _list_route_users(session, id)
        affected_user_ids = {item.id for item in previous_users} | set(
            requested_user_ids
        )
        cache_model = model

    if (
        access_request.access_policy is not None
        and access_request.access_policy != model.access_policy
    ):
        model.access_policy = access_request.access_policy
        affected_user_ids = None
        cache_model = None

    try:
        if replace_principals:
            await _replace_route_principals(
                session, id, [p.principal_id for p in access_request.principals]
            )
        elif replace_users:
            await _replace_route_user_principals(session, id, requested_user_ids)
        await revoke_model_access_cache(
            session=session,
            model=cache_model,
            extra_user_ids=affected_user_ids,
        )
        await ModelRouteService(session).update(model)
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to add model access: {e}")

    return ModelAuthorizationList(
        items=await _list_route_users(session, id),
        principals=await _list_route_principals(session, id),
        access_policy=model.access_policy,
    )


@my_models_router.get("", response_model=ModelRoutesPublic)
async def get_my_models(
    ctx: TenantContextDep,
    params: ModelRouteListParams = Depends(),
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    """List the model routes available to the calling user.

    Non-admin: read the ``non_admin_user_models`` view (one row per
    user × route × granting-principal) and partition it by the
    request's ``current_principal_id`` — Personal scope sees only
    USER/GROUP-mediated grants plus PUBLIC/AUTHED; Org act-as sees
    only grants tied to that Org plus PUBLIC/AUTHED.

    Platform admin: list the live ``model_routes`` table. With an Org
    context set, the act-as filter matches what a member of that Org
    would see; without one, return everything (the "All" view).
    """
    user = ctx.user
    user_id = None
    target_class = ModelRoute
    if not user.is_admin:
        target_class = MyModel
        user_id = user.id
    # Admin act-as: match what an org member would see (own + PUBLIC /
    # AUTHED + cross-tenant grants), not just routes whose ``owner`` is
    # this org. Admin "All" mode (no current_principal_id) keeps
    # bypassing tenant filters via ``include_grants`` returning [].

    return await _get_model_routes(
        params=params,
        search=search,
        categories=categories,
        target_class=target_class,
        user_id=user_id,
        ctx=ctx,
        include_grants=user.is_admin,
    )


@my_models_router.get("/{id}", response_model=ModelRoutePublic)
async def get_my_model(
    session: SessionDep,
    id: int,
    ctx: TenantContextDep,
):
    user = ctx.user
    user_id = None
    target_class = ModelRoute
    if not user.is_admin:
        target_class = MyModel
        user_id = user.id

    return await _get_model_route(
        session=session,
        id=id,
        user_id=user_id,
        target_class=target_class,
        ctx=ctx,
        include_grants=user.is_admin,
    )
