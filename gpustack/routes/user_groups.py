"""UserGroup management — Org owner+ or platform admin.

Groups are GROUP-kind ``Principal`` rows whose ``parent_principal_id``
points at their owning ORG-principal. Group memberships live in the
unified ``principal_memberships`` table with ``role=NULL`` (groups
don't have role tiers).
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
)
from gpustack.schemas.user_groups import (
    UserGroupCreate,
    UserGroupListParams,
    UserGroupMembershipPublic,
    UserGroupPublic,
    UserGroupUpdate,
    UserGroupsPublic,
)
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class GroupMembershipCreate(BaseModel):
    user_id: int


def _can_manage_groups(ctx, org_id: int) -> bool:
    if ctx.is_platform_admin:
        return True
    if ctx.current_principal_id != org_id:
        return False
    return ctx.org_role == OrgRole.OWNER


async def _load_org(session, org_id: int) -> Principal:
    org = await Principal.one_by_id(session, org_id)
    if not org or org.deleted_at is not None or org.kind != PrincipalType.ORG:
        raise NotFoundException(message="Organization not found")
    return org


async def _load_group(session, org_id: int, group_id: int) -> Principal:
    group = await Principal.one_by_id(session, group_id)
    if (
        not group
        or group.deleted_at is not None
        or group.kind != PrincipalType.GROUP
        or group.parent_principal_id != org_id
    ):
        raise NotFoundException(message="Group not found")
    return group


def _group_to_public(group: Principal) -> UserGroupPublic:
    return UserGroupPublic.from_principal(group)


# ---- groups ----------------------------------------------------------------


@router.get("/organizations/{org_id}/groups", response_model=UserGroupsPublic)
async def list_groups(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    params: UserGroupListParams = Depends(),
    search: Optional[str] = None,
):
    await _load_org(session, org_id)
    if not ctx.is_platform_admin and ctx.current_principal_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")

    fuzzy_fields = {"name": search} if search else {}
    page = await Principal.paginated_by_query(
        session=session,
        fields={
            "kind": PrincipalType.GROUP,
            "parent_principal_id": org_id,
            "deleted_at": None,
        },
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )
    page.items = [_group_to_public(g) for g in page.items]
    return page


@router.post("/organizations/{org_id}/groups", response_model=UserGroupPublic)
async def create_group(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    body: UserGroupCreate,
):
    await _load_org(session, org_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    existing = await Principal.one_by_fields(
        session,
        {
            "kind": PrincipalType.GROUP,
            "parent_principal_id": org_id,
            "name": body.name,
            "deleted_at": None,
        },
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Group '{body.name}' already exists in this organization"
        )

    try:
        group = Principal(
            kind=PrincipalType.GROUP,
            parent_principal_id=org_id,
            name=body.name,
            description=body.description,
        )
        created = await Principal.create(session, group)
    except Exception as e:
        raise InvalidException(message=f"Failed to create group: {e}")
    return _group_to_public(created)


@router.put("/organizations/{org_id}/groups/{group_id}", response_model=UserGroupPublic)
async def update_group(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    body: UserGroupUpdate,
):
    group = await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    try:
        await group.update(session, body.model_dump(exclude_unset=True))
    except Exception as e:
        raise InvalidException(message=f"Failed to update group: {e}")
    return _group_to_public(group)


@router.delete("/organizations/{org_id}/groups/{group_id}")
async def delete_group(
    session: SessionDep, ctx: TenantContextDep, org_id: int, group_id: int
):
    group = await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    try:
        await group.delete(session)
    except Exception as e:
        raise InvalidException(message=f"Failed to delete group: {e}")


# ---- group members ---------------------------------------------------------


async def _resolve_user(session, user_id: int) -> Optional[User]:
    user = await User.one_by_id(session, user_id)
    if not user or user.is_system or user.deleted_at is not None:
        return None
    return user


@router.get(
    "/organizations/{org_id}/groups/{group_id}/members",
    response_model=List[UserGroupMembershipPublic],
)
async def list_group_members(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
):
    await _load_group(session, org_id, group_id)
    if not ctx.is_platform_admin and ctx.current_principal_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")

    stmt = select(PrincipalMembership).where(
        PrincipalMembership.parent_principal_id == group_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    rows = list((await session.exec(stmt)).all())
    member_ids = {r.member_principal_id for r in rows}
    user_by_principal: dict[int, User] = {}
    if member_ids:
        result = await session.exec(
            select(User).where(User.principal_id.in_(member_ids))
        )
        user_by_principal = {u.principal_id: u for u in result.all()}
    out: List[UserGroupMembershipPublic] = []
    for r in rows:
        u = user_by_principal.get(r.member_principal_id)
        out.append(
            UserGroupMembershipPublic(
                user_id=getattr(u, "id", 0),
                group_id=group_id,
                created_at=r.created_at,
                username=getattr(u, "username", None),
                full_name=getattr(u, "full_name", None),
            )
        )
    return out


@router.post(
    "/organizations/{org_id}/groups/{group_id}/members",
    response_model=UserGroupMembershipPublic,
)
async def add_group_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    body: GroupMembershipCreate,
):
    await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    user = await _resolve_user(session, body.user_id)
    if not user:
        raise NotFoundException(message="User not found")

    # User must be an active member of the group's org first.
    org_membership_stmt = select(PrincipalMembership.id).where(
        PrincipalMembership.parent_principal_id == org_id,
        PrincipalMembership.member_principal_id == user.principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    if (await session.exec(org_membership_stmt)).first() is None:
        raise InvalidException(
            message=(
                f"User {body.user_id} is not a member of " f"organization {org_id}"
            )
        )

    existing_stmt = select(PrincipalMembership).where(
        PrincipalMembership.parent_principal_id == group_id,
        PrincipalMembership.member_principal_id == user.principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(
            message=f"User {body.user_id} is already in group {group_id}"
        )

    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        link = PrincipalMembership(
            parent_principal_id=group_id,
            member_principal_id=user.principal_id,
            role=None,
            created_at=now,
            updated_at=now,
        )
        session.add(link)
        await session.commit()
        await session.refresh(link)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add group member: {e}")
    return UserGroupMembershipPublic(
        user_id=user.id,
        group_id=group_id,
        created_at=link.created_at,
        username=user.username,
        full_name=user.full_name,
    )


@router.delete("/organizations/{org_id}/groups/{group_id}/members/{user_id}")
async def remove_group_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    user_id: int,
):
    await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    user = await _resolve_user(session, user_id)
    if not user:
        raise NotFoundException(message="Group membership not found")

    stmt = select(PrincipalMembership).where(
        PrincipalMembership.parent_principal_id == group_id,
        PrincipalMembership.member_principal_id == user.principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    link = (await session.exec(stmt)).first()
    if not link:
        raise NotFoundException(message="Group membership not found")

    try:
        await link.delete(session, soft=True)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove group member: {e}")
