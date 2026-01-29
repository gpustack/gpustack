import logging
import secrets
from sqlalchemy.orm import selectinload
from sqlmodel import col, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List, Optional, Tuple, Union, Dict
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from gpustack.schemas.model_routes import (
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
    ModelUserAccessExtended,
    MyModel,
    TargetStateEnum,
)
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model
from gpustack.server.deps import SessionDep, CurrentUserDep
from gpustack.schemas.users import User
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
)
from gpustack.server.services import (
    ModelRouteService,
    delete_accessible_model_cache,
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
    session: SessionDep,
    params: ModelRouteListParams = Depends(),
    name: str = None,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    return await _get_model_routes(
        session=session,
        params=params,
        name=name,
        search=search,
        categories=categories,
    )


async def _get_model_routes(
    session: AsyncSession,
    params: ModelRouteListParams,
    name: str = None,
    search: str = None,
    categories: Optional[List[str]] = None,
    user_id: Optional[int] = None,
    target_class: Union[ModelRoute, MyModel] = ModelRoute,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {'deleted_at': None}
    if name:
        fields = {"name": name}

    if user_id is not None:
        fields["user_id"] = user_id

    if params.watch:
        return StreamingResponse(
            target_class.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: categories_filter(data, categories),
            ),
            media_type="text/event-stream",
        )

    extra_conditions = []
    if categories:
        conditions = build_category_conditions(session, Model, categories)
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


@router.get("/{id}", response_model=ModelRoutePublic, response_model_exclude_none=True)
async def get_model_route(
    session: SessionDep,
    id: int,
):
    return await _get_model_route(session=session, id=id)


async def _get_model_route(
    session: AsyncSession,
    id: int,
    target_class: Union[ModelRoute, MyModel] = ModelRoute,
    user_id: Optional[int] = None,
):
    fields = {"id": id}
    if user_id is not None:
        fields["user_id"] = user_id
    existing = await target_class.one_by_fields(
        session=session,
        fields=fields,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelAccess with id '{id}' not found.")
    return existing


@router.post("", response_model=ModelRoutePublic, response_model_exclude_none=True)
async def create_model_route(session: SessionDep, input: ModelRouteCreate):
    existing = await ModelRoute.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(
            f"ModelRoute with name '{input.name}' already exists."
        )
    source = input.model_dump(exclude={"targets"})
    targets = input.targets or []
    await validate_targets(session, targets)
    source["targets"] = len(targets)
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
        await session.commit()
        await session.refresh(route)
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
    input: ModelRouteUpdate,
):
    existing = await ModelRoute.one_by_id(
        session=session,
        id=id,
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRoute with id '{id}' not found.")
    duplicated_name = await ModelRoute.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if duplicated_name and duplicated_name.id != id:
        raise AlreadyExistsException(
            f"ModelRoute with name '{input.name}' already exists."
        )
    input_data = input.model_dump(exclude={"targets"})
    try:
        if input.targets is not None or input.name != existing.name:
            target_count, _ = await batch_handle_targets(
                session=session,
                route_id=existing.id,
                route_name=existing.name,
                targets=input.targets,
                auto_commit=False,
                new_route_name=input.name if input.name != existing.name else None,
            )
            input_data["targets"] = target_count
        await ModelRouteService(session).update(
            existing, source=input_data, auto_commit=False
        )
        await session.commit()
    except Exception as e:
        raise InternalServerErrorException(f"Failed to update ModelRoute '{id}': {e}")
    return await ModelRoute.one_by_id(session=session, id=id)


@router.delete("/{id}")
async def delete_model_route(
    id: int,
    session: SessionDep,
):
    existing = await ModelRoute.one_by_id(
        session=session,
        id=id,
        options=[
            selectinload(ModelRoute.route_targets),
            selectinload(ModelRoute.users),
        ],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(f"ModelRoute with id '{id}' not found.")
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

    fallback_index = await validate_targets(session=session, targets=targets)
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
            "provider_model_name",
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


async def validate_targets(
    session: SessionDep,
    targets: List[ModelRouteTargetUpdateItem],
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
            validate_provider_model_name(provider, target.provider_model_name)
        elif target.model_id is not None:
            model = await Model.one_by_id(session=session, id=target.model_id)
            if model is None or model.deleted_at is not None:
                raise NotFoundException(f"Model with id '{target.model_id}' not found.")
    return fallback_index


def validate_provider_model_name(
    provider: ModelProvider,
    model_name: str,
):
    supported_models = provider.models or []
    model_names = [model.name for model in supported_models]
    if model_name not in model_names:
        raise InvalidException(
            f"provider_model_name '{model_name}' is not supported by provider '{provider.name}'. Supported models: {', '.join(model_names)}"
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
    await validate_targets(session, targets)
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


def model_access_list(model: ModelRoute) -> List[ModelUserAccessExtended]:
    return [
        ModelUserAccessExtended(
            id=access.id,
            username=access.username,
            full_name=access.full_name,
            avatar_url=access.avatar_url,
            # Add more user fields here if needed
        )
        for access in model.users
    ]


@router.get("/{id}/access", response_model=ModelAuthorizationList)
async def get_model_authorization_list(session: SessionDep, id: int):
    model: ModelRoute = await ModelRoute.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")

    return ModelAuthorizationList(items=model_access_list(model))


@router.post("/{id}/access", response_model=ModelAuthorizationList)
async def add_model_authorization(
    session: SessionDep, id: int, access_request: ModelAuthorizationUpdate
):
    model = await ModelRoute.one_by_id(session, id)
    if not model:
        raise NotFoundException(message="Model not found")
    extra_conditions = [
        col(User.id).in_([user.id for user in access_request.users]),
    ]

    affected_user_ids = {user.id for user in model.users}
    cache_model = model

    users = await User.all_by_fields(
        session=session, fields={}, extra_conditions=extra_conditions
    )
    if len(users) != len(access_request.users):
        existing_user_ids = {user.id for user in users}
        for req_user in access_request.users:
            if req_user.id not in existing_user_ids:
                raise NotFoundException(message=f"User ID {req_user.id} not found")

    model.users = list(users)
    if (
        access_request.access_policy is not None
        and access_request.access_policy != model.access_policy
    ):
        model.access_policy = access_request.access_policy
        # if changing to public, need to update all users
        affected_user_ids = None
        cache_model = None
    try:
        await revoke_model_access_cache(
            session=session,
            model=cache_model,
            extra_user_ids=affected_user_ids,
        )
        await ModelRouteService(session).update(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to add model access: {e}")
    await session.refresh(model)
    return ModelAuthorizationList(items=model_access_list(model))


@my_models_router.get("", response_model=ModelRoutesPublic)
async def get_my_models(
    user: CurrentUserDep,
    session: SessionDep,
    params: ModelRouteListParams = Depends(),
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    user_id = None
    target_class = ModelRoute
    if not user.is_admin:
        target_class = MyModel
        user_id = user.id

    return await _get_model_routes(
        session=session,
        params=params,
        search=search,
        categories=categories,
        target_class=target_class,
        user_id=user_id,
    )


@my_models_router.get("/{id}", response_model=ModelRoutePublic)
async def get_my_model(
    session: SessionDep,
    id: int,
    user: CurrentUserDep,
):
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
    )


async def revoke_model_access_cache(
    session: AsyncSession,
    model: Optional[ModelRoute] = None,
    extra_user_ids: Optional[set[int]] = None,
):
    user_ids = set()
    if model is None:
        users = await User.all(session)
        user_ids = {user.id for user in users}
    else:
        user_ids = {user.id for user in model.users}
    if extra_user_ids:
        user_ids.update(extra_user_ids)
    await delete_accessible_model_cache(session, *user_ids)
