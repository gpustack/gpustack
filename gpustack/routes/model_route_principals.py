"""Manage principal-based access on a ModelRoute (ALLOWED_PRINCIPALS).

Mounted under /v2/model-routes/{id}/principals on the admin router.
The set of principals (USER / ORG / GROUP) attached to a route only
takes effect when the route's ``access_policy`` is ``ALLOWED_PRINCIPALS``.

Storage: each principal is one ``model_route_principals`` row with a
single ``principal_id`` FK. Kind is read from the joined ``principals``
row. The API surface keeps the legacy ``(principal_type, principal_id)``
shape, where ``principal_id`` here is the principals.id of the target
(USER / ORG / GROUP principal).
"""

from typing import List, Optional

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
from gpustack.schemas.principals import Principal, PrincipalType
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
    # Resolved at read time from the joined principals row's
    # ``display_name`` column so the client can render a label without
    # an extra lookup.
    principal_display_name: Optional[str] = None


async def _load_route(session, route_id: int) -> ModelRoute:
    route = await ModelRoute.one_by_id(session, route_id)
    if not route or route.deleted_at is not None:
        raise NotFoundException(message="Model route not found")
    return route


async def _validate_principal(
    session, principal_type: PrincipalType, principal_id: int
) -> Principal:
    target = await Principal.one_by_id(session, principal_id)
    if not target or target.deleted_at is not None:
        raise InvalidException(message=f"Principal {principal_id} not found")
    if target.kind != principal_type:
        raise InvalidException(
            message=(
                f"Principal {principal_id} is a {target.kind.value}, "
                f"not a {principal_type.value}"
            )
        )
    # Caller-requested kind is constrained to USER / GROUP / ORG by
    # the API surface; SYSTEM principals are rejected by the
    # ``target.kind != principal_type`` mismatch above (no API path
    # asks for kind=SYSTEM in an ACL grant).
    return target


def _row_to_view(
    row: ModelRoutePrincipalLink,
    kind: PrincipalType,
    display_name: Optional[str] = None,
) -> PrincipalView:
    return PrincipalView(
        route_id=row.route_id,
        principal_type=kind,
        principal_id=row.principal_id,
        principal_display_name=display_name,
    )


async def _resolve_views(
    session, rows: List[ModelRoutePrincipalLink]
) -> List[PrincipalView]:
    principal_ids = {r.principal_id for r in rows}
    by_id: dict[int, Principal] = {}
    if principal_ids:
        result = await session.exec(
            select(Principal).where(Principal.id.in_(principal_ids))
        )
        by_id = {p.id: p for p in result.all()}
    return [
        _row_to_view(
            r,
            (
                by_id[r.principal_id].kind
                if r.principal_id in by_id
                else PrincipalType.USER
            ),
            by_id[r.principal_id].display_name if r.principal_id in by_id else None,
        )
        for r in rows
    ]


@router.get("/{id}/principals", response_model=List[PrincipalView])
async def list_route_principals(session: SessionDep, id: int):
    await _load_route(session, id)
    stmt = select(ModelRoutePrincipalLink).where(ModelRoutePrincipalLink.route_id == id)
    rows = list((await session.exec(stmt)).all())
    return await _resolve_views(session, rows)


@router.post("/{id}/principals", response_model=PrincipalView)
async def add_route_principal(session: SessionDep, id: int, body: PrincipalRef):
    await _load_route(session, id)
    target = await _validate_principal(session, body.principal_type, body.principal_id)

    existing_stmt = select(ModelRoutePrincipalLink).where(
        ModelRoutePrincipalLink.route_id == id,
        ModelRoutePrincipalLink.principal_id == body.principal_id,
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(message="Principal already attached to route")

    try:
        link = ModelRoutePrincipalLink(
            route_id=id,
            principal_id=body.principal_id,
        )
        session.add(link)
        await session.commit()
        await session.refresh(link)
        # Visibility may have widened; bust the access cache for the
        # route. Pass model=None to broadly invalidate accessible-model
        # caches; the set of affected users for an org/group principal
        # can't be derived cheaply from ``route`` alone, so we err on
        # the side of correctness.
        await revoke_model_access_cache(session=session)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add principal: {e}")
    return _row_to_view(link, target.kind)


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
        ModelRoutePrincipalLink.principal_id == principal_id,
    )
    link = (await session.exec(stmt)).first()
    if not link:
        raise NotFoundException(message="Principal not attached to route")

    try:
        await session.delete(link)
        await session.commit()
        await revoke_model_access_cache(session=session)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove principal: {e}")
