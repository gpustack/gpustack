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
    user_ids: List[int]


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
    response_model=List[UserGroupMembershipPublic],
)
async def add_group_members(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    body: GroupMembershipCreate,
):
    """Add one or more users to a group in a single request.

    All-or-nothing: validate every user up front (exists, is an Org
    member, not already in the group); if any check fails, raise and
    write nothing. Otherwise all rows insert in a single transaction.
    """
    await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    if not body.user_ids:
        raise InvalidException(message="user_ids must not be empty")

    user_ids = list(dict.fromkeys(body.user_ids))

    rows = (await session.exec(select(User).where(User.id.in_(user_ids)))).all()
    users_by_id: dict[int, User] = {
        u.id: u for u in rows if not u.is_system and u.deleted_at is None
    }
    missing = [uid for uid in user_ids if uid not in users_by_id]
    if missing:
        raise NotFoundException(message=f"User(s) not found: {missing}")

    principal_ids = [users_by_id[uid].principal_id for uid in user_ids]

    # Every user must already be an active org member.
    org_member_principals = set(
        (
            await session.exec(
                select(PrincipalMembership.member_principal_id).where(
                    PrincipalMembership.parent_principal_id == org_id,
                    PrincipalMembership.member_principal_id.in_(principal_ids),
                    PrincipalMembership.deleted_at.is_(None),
                )
            )
        ).all()
    )
    non_members = [
        uid
        for uid in user_ids
        if users_by_id[uid].principal_id not in org_member_principals
    ]
    if non_members:
        raise InvalidException(
            message=f"User(s) not in organization {org_id}: {non_members}"
        )

    # Bulk-load existing memberships (including soft-deleted, so we can
    # resurrect them instead of producing duplicate rows). Matches the
    # add_org_members soft-delete handling so a user removed from a
    # group and re-added keeps a single timeline row.
    existing_rows = (
        await session.exec(
            select(PrincipalMembership).where(
                PrincipalMembership.parent_principal_id == group_id,
                PrincipalMembership.member_principal_id.in_(principal_ids),
            )
        )
    ).all()
    existing_by_principal: dict[int, PrincipalMembership] = {
        m.member_principal_id: m for m in existing_rows
    }
    duplicates = [
        uid
        for uid in user_ids
        if (m := existing_by_principal.get(users_by_id[uid].principal_id)) is not None
        and m.deleted_at is None
    ]
    if duplicates:
        raise AlreadyExistsException(
            message=f"User(s) already in group {group_id}: {duplicates}"
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stored_pairs: List[tuple[User, PrincipalMembership]] = []
    try:
        for uid in user_ids:
            user = users_by_id[uid]
            existing = existing_by_principal.get(user.principal_id)
            if existing is not None:
                # Soft-deleted row → resurrect so the membership timeline
                # stays on a single row.
                existing.deleted_at = None
                existing.updated_at = now
                session.add(existing)
                stored_pairs.append((user, existing))
            else:
                link = PrincipalMembership(
                    parent_principal_id=group_id,
                    member_principal_id=user.principal_id,
                    role=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(link)
                stored_pairs.append((user, link))
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add group members: {e}")

    return [
        UserGroupMembershipPublic(
            user_id=user.id,
            group_id=group_id,
            created_at=link.created_at,
            username=user.username,
            full_name=user.full_name,
        )
        for user, link in stored_pairs
    ]


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
