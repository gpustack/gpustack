"""Manage principal-based access on a ModelRoute (ALLOWED_PRINCIPALS).

Mounted under /v2/model-routes/{id}/principals on the admin router. The set of
principals (org / group / user) attached to a route only takes effect when
the route's `access_policy` is `ALLOWED_PRINCIPALS`; the legacy `ALLOWED_USERS`
linkage continues to live in `usermodelroutelink` and is independent.
"""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.organizations import Organization
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.user_groups import UserGroup
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep
from gpustack.server.services import revoke_model_access_cache

router = APIRouter()


class PrincipalRef(BaseModel):
    principal_type: PrincipalType
    principal_id: int


class PrincipalView(BaseModel):
    route_id: int
    principal_type: PrincipalType
    principal_id: int

    model_config = {"from_attributes": True}


async def _load_route(session, route_id: int) -> ModelRoute:
    route = await ModelRoute.one_by_id(session, route_id)
    if not route or route.deleted_at is not None:
        raise NotFoundException(message="Model route not found")
    return route


async def _validate_principal(
    session, principal_type: PrincipalType, principal_id: int
) -> None:
    if principal_type == PrincipalType.ORG:
        target = await Organization.one_by_id(session, principal_id)
        if not target or target.deleted_at is not None:
            raise InvalidException(message=f"Organization {principal_id} not found")
    elif principal_type == PrincipalType.GROUP:
        target = await UserGroup.one_by_id(session, principal_id)
        if not target or target.deleted_at is not None:
            raise InvalidException(message=f"User group {principal_id} not found")
    elif principal_type == PrincipalType.USER:
        target = await User.one_by_id(session, principal_id)
        if not target or target.is_system or target.deleted_at is not None:
            raise InvalidException(message=f"User {principal_id} not found")


@router.get("/{id}/principals", response_model=List[PrincipalView])
async def list_route_principals(session: SessionDep, id: int):
    await _load_route(session, id)
    stmt = select(ModelRoutePrincipalLink).where(ModelRoutePrincipalLink.route_id == id)
    return list((await session.exec(stmt)).all())


@router.post("/{id}/principals", response_model=PrincipalView)
async def add_route_principal(session: SessionDep, id: int, body: PrincipalRef):
    await _load_route(session, id)
    await _validate_principal(session, body.principal_type, body.principal_id)

    existing_stmt = select(ModelRoutePrincipalLink).where(
        ModelRoutePrincipalLink.route_id == id,
        ModelRoutePrincipalLink.principal_type == body.principal_type,
        ModelRoutePrincipalLink.principal_id == body.principal_id,
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(message="Principal already attached to route")

    try:
        link = ModelRoutePrincipalLink(
            route_id=id,
            principal_type=body.principal_type,
            principal_id=body.principal_id,
        )
        session.add(link)
        await session.commit()
        await session.refresh(link)
        # Visibility may have widened; bust the access cache for the route.
        # Pass model=None to broadly invalidate accessible-model caches; the
        # set of affected users for an org/group principal can't be derived
        # cheaply from `route` alone, so we err on the side of correctness.
        await revoke_model_access_cache(session=session)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add principal: {e}")
    return link


@router.delete("/{id}/principals/{principal_type}/{principal_id}")
async def remove_route_principal(
    session: SessionDep,
    id: int,
    principal_type: PrincipalType,
    principal_id: int,
):
    await _load_route(session, id)

    stmt = select(ModelRoutePrincipalLink).where(
        ModelRoutePrincipalLink.route_id == id,
        ModelRoutePrincipalLink.principal_type == principal_type,
        ModelRoutePrincipalLink.principal_id == principal_id,
    )
    link = (await session.exec(stmt)).first()
    if not link:
        raise NotFoundException(message="Principal not attached to route")

    try:
        await session.delete(link)
        await session.commit()
        # Pass model=None to broadly invalidate accessible-model caches; the
        # set of affected users for an org/group principal can't be derived
        # cheaply from `route` alone, so we err on the side of correctness.
        await revoke_model_access_cache(session=session)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove principal: {e}")
