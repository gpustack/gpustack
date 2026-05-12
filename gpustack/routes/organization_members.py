"""Organization membership management.

These routes are nested under /organizations/{org_id}/members. Both the
platform admin and any Org owner can manage memberships. The last owner
of an Org cannot be demoted or removed — that would leave the Org
without anyone able to manage members or infra.

Storage is the unified ``principal_memberships`` table. Each row links
two principals: the parent (an ORG-principal here, but the same table
also carries GROUP memberships) and the member (a USER-principal). The
URL path's ``user_id`` is the legacy ``users.id``, which we resolve to
the user's ``principal_id`` for storage.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.organizations import OrganizationMembershipPublic
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
)
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class MembershipCreate(BaseModel):
    user_ids: List[int]
    role: OrgRole = OrgRole.MEMBER


class MembershipUpdate(BaseModel):
    role: OrgRole


def _can_manage(ctx, org_id: int) -> bool:
    """Platform admin can manage any Org's memberships; an Org owner
    can only manage their own Org. The role check is bound to the
    target Org from the URL path — ``ctx.org_role`` reflects the
    caller's *current* Org context, which may not match the path when
    a savvy client crafts the URL directly. Anchoring on ``org_id``
    closes that cross-Org escalation.
    """
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


async def _resolve_user(session, user_id: int) -> Optional[User]:
    user = await User.one_by_id(session, user_id)
    if not user or user.is_system or user.deleted_at is not None:
        return None
    return user


async def _list_memberships(
    session, org_principal_id: int
) -> List[PrincipalMembership]:
    stmt = select(PrincipalMembership).where(
        PrincipalMembership.parent_principal_id == org_principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    return list((await session.exec(stmt)).all())


async def _find_membership(
    session,
    org_principal_id: int,
    member_principal_id: int,
    *,
    include_deleted: bool = False,
) -> Optional[PrincipalMembership]:
    """Return the (optionally soft-deleted) membership row.

    Used by the add path with ``include_deleted=True`` so a soft-deleted
    row can be resurrected instead of producing a duplicate.
    """
    conditions = [
        PrincipalMembership.parent_principal_id == org_principal_id,
        PrincipalMembership.member_principal_id == member_principal_id,
    ]
    if not include_deleted:
        conditions.append(PrincipalMembership.deleted_at.is_(None))
    stmt = select(PrincipalMembership).where(*conditions)
    return (await session.exec(stmt)).first()


async def _has_other_owner(
    session, org_principal_id: int, exclude_member_principal_id: int
) -> bool:
    stmt = select(PrincipalMembership.id).where(
        PrincipalMembership.parent_principal_id == org_principal_id,
        PrincipalMembership.role == OrgRole.OWNER,
        PrincipalMembership.member_principal_id != exclude_member_principal_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    return (await session.exec(stmt)).first() is not None


async def _enrich_with_user_labels(
    session,
    org_principal_id: int,
    rows: List[PrincipalMembership],
) -> List[OrganizationMembershipPublic]:
    """Bulk-resolve username / full_name / users.id for each membership.

    Membership rows reference users via ``member_principal_id``
    (= ``users.principal_id``). We join back to ``users`` so the
    response can carry the legacy ``user_id`` (= ``users.id``) plus
    display labels, in a single query — no per-row round trip from
    the client.
    """
    member_ids = {r.member_principal_id for r in rows}
    user_by_principal: dict[int, User] = {}
    if member_ids:
        result = await session.exec(
            select(User).where(User.principal_id.in_(member_ids))
        )
        user_by_principal = {u.principal_id: u for u in result.all()}
    out: List[OrganizationMembershipPublic] = []
    for r in rows:
        u = user_by_principal.get(r.member_principal_id)
        out.append(
            OrganizationMembershipPublic(
                user_id=getattr(u, "id", 0),
                organization_id=org_principal_id,
                role=r.role,
                created_at=r.created_at,
                username=getattr(u, "username", None),
                full_name=getattr(u, "full_name", None),
            )
        )
    return out


@router.get(
    "/organizations/{org_id}/members",
    response_model=List[OrganizationMembershipPublic],
)
async def list_org_members(session: SessionDep, ctx: TenantContextDep, org_id: int):
    await _load_org(session, org_id)
    if not ctx.is_platform_admin and ctx.current_principal_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")
    rows = await _list_memberships(session, org_id)
    return await _enrich_with_user_labels(session, org_id, rows)


@router.post(
    "/organizations/{org_id}/members",
    response_model=List[OrganizationMembershipPublic],
)
async def add_org_members(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    body: MembershipCreate,
):
    """Add one or more members in a single request.

    All-or-nothing: validate every user up front; if any are missing or
    already members, raise and write nothing. On success, every row is
    inserted in a single transaction.
    """
    org = await _load_org(session, org_id)

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to add member")

    if not body.user_ids:
        raise InvalidException(message="user_ids must not be empty")

    # Preserve order, drop duplicates the client sent.
    user_ids = list(dict.fromkeys(body.user_ids))

    # Bulk-resolve users in one query. Filter out system / soft-deleted
    # rows the same way _resolve_user does.
    rows = (await session.exec(select(User).where(User.id.in_(user_ids)))).all()
    users_by_id: dict[int, User] = {
        u.id: u for u in rows if not u.is_system and u.deleted_at is None
    }
    missing = [uid for uid in user_ids if uid not in users_by_id]
    if missing:
        raise NotFoundException(message=f"User(s) not found: {missing}")

    # Bulk-load existing memberships (including soft-deleted, so we can
    # resurrect them instead of producing duplicate rows).
    principal_ids = [users_by_id[uid].principal_id for uid in user_ids]
    existing_rows = (
        await session.exec(
            select(PrincipalMembership).where(
                PrincipalMembership.parent_principal_id == org.id,
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
            message=f"User(s) already members of organization {org.id}: {duplicates}"
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stored_pairs: List[tuple[User, PrincipalMembership]] = []
    try:
        for uid in user_ids:
            user = users_by_id[uid]
            existing = existing_by_principal.get(user.principal_id)
            if existing is not None:
                # Resurrect a soft-deleted row so the membership timeline
                # stays on a single row.
                existing.deleted_at = None
                existing.role = body.role
                existing.updated_at = now
                session.add(existing)
                stored_pairs.append((user, existing))
            else:
                row = PrincipalMembership(
                    parent_principal_id=org.id,
                    member_principal_id=user.principal_id,
                    role=body.role,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                stored_pairs.append((user, row))
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add members: {e}")

    return [
        OrganizationMembershipPublic(
            user_id=user.id,
            organization_id=org.id,
            role=row.role,
            created_at=row.created_at,
            username=user.username,
            full_name=user.full_name,
        )
        for user, row in stored_pairs
    ]


@router.put(
    "/organizations/{org_id}/members/{user_id}",
    response_model=OrganizationMembershipPublic,
)
async def update_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    user_id: int,
    body: MembershipUpdate,
):
    await _load_org(session, org_id)
    user = await _resolve_user(session, user_id)
    if not user:
        raise NotFoundException(message="Membership not found")

    membership = await _find_membership(session, org_id, user.principal_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to change role")

    if membership.role == OrgRole.OWNER and body.role != OrgRole.OWNER:
        if not await _has_other_owner(
            session, org_id, exclude_member_principal_id=user.principal_id
        ):
            raise ConflictException(
                message="Cannot demote the only owner of this organization"
            )

    try:
        membership.role = body.role
        session.add(membership)
        await session.commit()
        await session.refresh(membership)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to update member: {e}")

    return OrganizationMembershipPublic(
        user_id=user.id,
        organization_id=org_id,
        role=membership.role,
        created_at=membership.created_at,
        username=user.username,
        full_name=user.full_name,
    )


@router.delete("/organizations/{org_id}/members/{user_id}")
async def remove_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    user_id: int,
):
    await _load_org(session, org_id)
    user = await _resolve_user(session, user_id)
    if not user:
        raise NotFoundException(message="Membership not found")

    membership = await _find_membership(session, org_id, user.principal_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to remove member")

    if membership.role == OrgRole.OWNER:
        if not await _has_other_owner(
            session, org_id, exclude_member_principal_id=user.principal_id
        ):
            raise ConflictException(
                message="Cannot remove the only owner of this organization"
            )

    try:
        await membership.delete(session, soft=True)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove member: {e}")
