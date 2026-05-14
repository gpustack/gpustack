"""Organization membership management.

Routes live under ``/organizations/{org_id}/members``. Both the
platform admin and any Org owner can manage memberships. The last
OWNER of an Org cannot be demoted or removed — that protection
counts both USER-owners and GROUP-owners (active members of a
GROUP-owner can act as Org admin via transitive role resolution).

Storage is the unified ``principal_memberships`` table. Each row
links two principals: the parent (an ORG-principal here) and the
member (a USER or GROUP principal). Group-members confer ``role`` on
every active user inside the Group. The URL path's ``principal_id``
is the principal id of the member — for USER members it equals
``users.principal_id``; for GROUP members it's the group's
principal id.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import and_, exists, or_
from sqlalchemy.orm import aliased
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
    principal_ids: List[int]
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


async def _resolve_member_principal(session, principal_id: int) -> Optional[Principal]:
    """Return a principal eligible to be an Org member.

    Only USER and GROUP kinds can join an Org. ORG-of-ORG is rejected
    here even though the table would technically accept it.
    """
    p = await Principal.one_by_id(session, principal_id)
    if (
        not p
        or p.deleted_at is not None
        or p.kind not in (PrincipalType.USER, PrincipalType.GROUP)
    ):
        return None
    return p


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
    """True if at least one OWNER-membership other than ``exclude``
    can actually act as an owner of this Org.

    "Can act as owner" means:

    - USER-owner: the user-principal itself is not soft-deleted.
    - GROUP-owner: the group-principal is not soft-deleted *and* has
      at least one active user-member. A soft-deleted Group, or a
      Group with no active users, confers OWNER on nobody — counting
      it would let the operator demote the last real owner and leave
      the Org effectively ownerless.
    """
    gm = aliased(PrincipalMembership)
    stmt = (
        select(PrincipalMembership.id)
        .join(Principal, Principal.id == PrincipalMembership.member_principal_id)
        .where(
            PrincipalMembership.parent_principal_id == org_principal_id,
            PrincipalMembership.role == OrgRole.OWNER,
            PrincipalMembership.member_principal_id != exclude_member_principal_id,
            PrincipalMembership.deleted_at.is_(None),
            Principal.deleted_at.is_(None),
            or_(
                Principal.kind == PrincipalType.USER,
                and_(
                    Principal.kind == PrincipalType.GROUP,
                    exists(
                        select(gm.id).where(
                            gm.parent_principal_id == Principal.id,
                            gm.deleted_at.is_(None),
                        )
                    ),
                ),
            ),
        )
        .limit(1)
    )
    return (await session.exec(stmt)).first() is not None


async def _enrich_with_labels(
    session,
    org_principal_id: int,
    rows: List[PrincipalMembership],
) -> List[OrganizationMembershipPublic]:
    """Bulk-resolve display labels for each membership row.

    Principal-level fields come off ``principals``. ``full_name`` is
    looked up on ``users`` for USER-kind members only — a single
    bounded query, scoped to just the user-principals in this page.
    Drops away once identity consolidation moves full_name onto the
    principal row.
    """
    member_ids = {r.member_principal_id for r in rows}
    if not member_ids:
        return []

    principal_by_id: dict[int, Principal] = {
        p.id: p
        for p in (
            await session.exec(
                select(Principal).where(
                    Principal.id.in_(member_ids),
                    Principal.deleted_at.is_(None),
                )
            )
        ).all()
    }

    user_principal_ids = [
        pid for pid, p in principal_by_id.items() if p.kind == PrincipalType.USER
    ]
    full_name_by_principal: dict[int, Optional[str]] = {}
    if user_principal_ids:
        full_name_by_principal = {
            u.principal_id: u.full_name
            for u in (
                await session.exec(
                    select(User).where(User.principal_id.in_(user_principal_ids))
                )
            ).all()
        }

    out: List[OrganizationMembershipPublic] = []
    for r in rows:
        p = principal_by_id.get(r.member_principal_id)
        if p is None:
            continue
        out.append(
            _to_public(
                p,
                r.role,
                r.created_at,
                org_principal_id,
                full_name=full_name_by_principal.get(p.id),
            )
        )
    return out


def _to_public(
    p: Principal,
    role: Optional[OrgRole],
    created_at: datetime,
    organization_id: int,
    *,
    full_name: Optional[str] = None,
) -> OrganizationMembershipPublic:
    return OrganizationMembershipPublic(
        principal_id=p.id,
        principal_kind=p.kind,
        principal_name=p.name,
        principal_description=p.description,
        full_name=full_name,
        organization_id=organization_id,
        role=role,
        created_at=created_at,
    )


@router.get(
    "/organizations/{org_id}/members",
    response_model=List[OrganizationMembershipPublic],
)
async def list_org_members(session: SessionDep, ctx: TenantContextDep, org_id: int):
    """List all members of an Org — Users and Groups together."""
    await _load_org(session, org_id)
    if not ctx.is_platform_admin and ctx.current_principal_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")
    stmt = select(PrincipalMembership).where(
        PrincipalMembership.parent_principal_id == org_id,
        PrincipalMembership.deleted_at.is_(None),
    )
    rows = list((await session.exec(stmt)).all())
    return await _enrich_with_labels(session, org_id, rows)


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
    """Add one or more principals (Users or Groups) as Org members.

    All-or-nothing: validate every principal up front; if any are
    missing, of the wrong kind, or already members, raise and write
    nothing. On success, every row is inserted in a single
    transaction. Soft-deleted rows are resurrected so the membership
    timeline stays on a single row.
    """
    org = await _load_org(session, org_id)

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to add member")

    if not body.principal_ids:
        raise InvalidException(message="principal_ids must not be empty")

    principal_ids = list(dict.fromkeys(body.principal_ids))

    candidates = (
        await session.exec(
            select(Principal).where(
                Principal.id.in_(principal_ids),
                Principal.deleted_at.is_(None),
            )
        )
    ).all()
    principal_by_id: dict[int, Principal] = {
        p.id: p
        for p in candidates
        if p.kind in (PrincipalType.USER, PrincipalType.GROUP)
    }
    missing = [pid for pid in principal_ids if pid not in principal_by_id]
    if missing:
        raise NotFoundException(message=f"Principal(s) not found: {missing}")

    # USER-kind: filter out system users (workers / cluster service
    # accounts must not be enrolled as Org members). System-user
    # detection still lives on the ``users`` table, so a single
    # bounded query — only over the USER-kind principals we're about
    # to insert — gates them. We reuse the ``users`` rows we pulled
    # to fill ``full_name`` on the response.
    user_principal_ids = [
        pid for pid, p in principal_by_id.items() if p.kind == PrincipalType.USER
    ]
    users_by_principal: dict[int, User] = {}
    if user_principal_ids:
        users_by_principal = {
            u.principal_id: u
            for u in (
                await session.exec(
                    select(User).where(User.principal_id.in_(user_principal_ids))
                )
            ).all()
        }
        bad = [
            pid
            for pid in user_principal_ids
            if (u := users_by_principal.get(pid)) is None
            or u.is_system
            or u.deleted_at is not None
        ]
        if bad:
            raise NotFoundException(message=f"Principal(s) not eligible: {bad}")

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
        pid
        for pid in principal_ids
        if (m := existing_by_principal.get(pid)) is not None and m.deleted_at is None
    ]
    if duplicates:
        raise AlreadyExistsException(
            message=f"Principal(s) already members of organization {org.id}: {duplicates}"
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    stored: List[tuple[Principal, PrincipalMembership]] = []
    try:
        for pid in principal_ids:
            p = principal_by_id[pid]
            existing = existing_by_principal.get(pid)
            if existing is not None:
                existing.deleted_at = None
                existing.role = body.role
                existing.updated_at = now
                session.add(existing)
                stored.append((p, existing))
            else:
                row = PrincipalMembership(
                    parent_principal_id=org.id,
                    member_principal_id=pid,
                    role=body.role,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                stored.append((p, row))
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add members: {e}")

    return [
        _to_public(
            p,
            row.role,
            row.created_at,
            org.id,
            full_name=getattr(users_by_principal.get(p.id), "full_name", None),
        )
        for p, row in stored
    ]


@router.put(
    "/organizations/{org_id}/members/{principal_id}",
    response_model=OrganizationMembershipPublic,
)
async def update_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    principal_id: int,
    body: MembershipUpdate,
):
    await _load_org(session, org_id)
    p = await _resolve_member_principal(session, principal_id)
    if not p:
        raise NotFoundException(message="Membership not found")

    membership = await _find_membership(session, org_id, principal_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to change role")

    if membership.role == OrgRole.OWNER and body.role != OrgRole.OWNER:
        if not await _has_other_owner(
            session, org_id, exclude_member_principal_id=principal_id
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

    full_name: Optional[str] = None
    if p.kind == PrincipalType.USER:
        u = await User.one_by_field(session, "principal_id", p.id)
        full_name = getattr(u, "full_name", None)
    return _to_public(
        p, membership.role, membership.created_at, org_id, full_name=full_name
    )


@router.delete("/organizations/{org_id}/members/{principal_id}")
async def remove_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    principal_id: int,
):
    await _load_org(session, org_id)
    p = await _resolve_member_principal(session, principal_id)
    if not p:
        raise NotFoundException(message="Membership not found")

    membership = await _find_membership(session, org_id, principal_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to remove member")

    if membership.role == OrgRole.OWNER:
        if not await _has_other_owner(
            session, org_id, exclude_member_principal_id=principal_id
        ):
            raise ConflictException(
                message="Cannot remove the only owner of this organization"
            )

    try:
        await membership.delete(session, soft=True)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove member: {e}")
