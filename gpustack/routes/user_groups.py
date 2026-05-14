"""UserGroup management — platform admin only.

Groups are GROUP-kind ``Principal`` rows. They are peer-level
principals: no structural parent, no Org affiliation baked into the
row. A Group may be a member of zero or more Orgs via rows in
``principal_memberships`` with ``parent=Org, member=Group`` — managed
through the ``organization_members`` routes, not here.

User membership in a Group lives in ``principal_memberships`` too
(``parent=Group, member=User``, ``role=NULL``).
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import require_platform_admin
from gpustack.schemas.principals import (
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


async def _load_group(session, group_id: int) -> Principal:
    group = await Principal.one_by_id(session, group_id)
    if not group or group.deleted_at is not None or group.kind != PrincipalType.GROUP:
        raise NotFoundException(message="Group not found")
    return group


async def _bulk_member_counts(session, group_ids: List[int]) -> Dict[int, int]:
    """Active user-count per group, in one COUNT GROUP BY query.

    Single query regardless of page size — avoids an N+1 over
    ``/groups/{id}/members`` from the UI. Groups absent from the
    result map have zero active members; callers default to 0.
    """
    if not group_ids:
        return {}
    stmt = (
        select(
            PrincipalMembership.parent_principal_id,
            func.count(PrincipalMembership.id),
        )
        .where(
            PrincipalMembership.parent_principal_id.in_(group_ids),
            PrincipalMembership.deleted_at.is_(None),
        )
        .group_by(PrincipalMembership.parent_principal_id)
    )
    return {row[0]: row[1] for row in (await session.exec(stmt)).all()}


# ---- groups (platform-admin CRUD) ------------------------------------------


@router.get("/groups", response_model=UserGroupsPublic)
async def list_groups(
    session: SessionDep,
    ctx: TenantContextDep,
    params: UserGroupListParams = Depends(),
    search: Optional[str] = None,
):
    """List all groups. Visible to any authenticated user — Group names
    are global identifiers in the system; rendering an Org's
    group-memberships needs to be able to enumerate candidate groups.
    """
    fuzzy_fields = {"name": search} if search else {}
    page = await Principal.paginated_by_query(
        session=session,
        fields={
            "kind": PrincipalType.GROUP,
            "deleted_at": None,
        },
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )
    counts = await _bulk_member_counts(session, [g.id for g in page.items])
    page.items = [
        UserGroupPublic.from_principal(g, member_count=counts.get(g.id, 0))
        for g in page.items
    ]
    return page


@router.get("/groups/{group_id}", response_model=UserGroupPublic)
async def get_group(session: SessionDep, ctx: TenantContextDep, group_id: int):
    group = await _load_group(session, group_id)
    counts = await _bulk_member_counts(session, [group.id])
    return UserGroupPublic.from_principal(group, member_count=counts.get(group.id, 0))


@router.post(
    "/groups",
    response_model=UserGroupPublic,
    dependencies=[Depends(require_platform_admin)],
)
async def create_group(session: SessionDep, body: UserGroupCreate):
    existing = await Principal.one_by_fields(
        session,
        {
            "kind": PrincipalType.GROUP,
            "name": body.name,
            "deleted_at": None,
        },
    )
    if existing:
        raise AlreadyExistsException(message=f"Group '{body.name}' already exists")

    try:
        group = Principal(
            kind=PrincipalType.GROUP,
            name=body.name,
            description=body.description,
        )
        created = await Principal.create(session, group)
    except Exception as e:
        raise InvalidException(message=f"Failed to create group: {e}")
    # New group has no members yet; skip the COUNT round trip.
    return UserGroupPublic.from_principal(created, member_count=0)


@router.put(
    "/groups/{group_id}",
    response_model=UserGroupPublic,
    dependencies=[Depends(require_platform_admin)],
)
async def update_group(session: SessionDep, group_id: int, body: UserGroupUpdate):
    group = await _load_group(session, group_id)

    if body.name != group.name:
        clash = await Principal.one_by_fields(
            session,
            {
                "kind": PrincipalType.GROUP,
                "name": body.name,
                "deleted_at": None,
            },
        )
        if clash and clash.id != group.id:
            raise AlreadyExistsException(message=f"Group '{body.name}' already exists")

    try:
        await group.update(session, body.model_dump(exclude_unset=True))
    except Exception as e:
        raise InvalidException(message=f"Failed to update group: {e}")
    counts = await _bulk_member_counts(session, [group.id])
    return UserGroupPublic.from_principal(group, member_count=counts.get(group.id, 0))


@router.delete(
    "/groups/{group_id}",
    dependencies=[Depends(require_platform_admin)],
)
async def delete_group(session: SessionDep, group_id: int):
    group = await _load_group(session, group_id)
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
    "/groups/{group_id}/members",
    response_model=List[UserGroupMembershipPublic],
)
async def list_group_members(session: SessionDep, ctx: TenantContextDep, group_id: int):
    await _load_group(session, group_id)

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
    "/groups/{group_id}/members",
    response_model=List[UserGroupMembershipPublic],
    dependencies=[Depends(require_platform_admin)],
)
async def add_group_members(
    session: SessionDep,
    group_id: int,
    body: GroupMembershipCreate,
):
    """Add one or more users to a group in a single request.

    All-or-nothing: validate every user up front; if any check fails,
    raise and write nothing. Otherwise all rows insert in a single
    transaction.
    """
    await _load_group(session, group_id)

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


@router.delete(
    "/groups/{group_id}/members/{user_id}",
    dependencies=[Depends(require_platform_admin)],
)
async def remove_group_member(
    session: SessionDep,
    group_id: int,
    user_id: int,
):
    await _load_group(session, group_id)

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
