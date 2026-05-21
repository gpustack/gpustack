"""Unit tests for P2 (Org / Group / Membership / cluster_access) route logic.

The codebase pattern for tests in this repo is mock-based; we follow that
here. Coverage focuses on the authorization branches and the corner cases
that aren't trivially expressible in declarative SQL constraints.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.routes import (
    cluster_access as cluster_access_route,
    dashboard as dashboard_route,
    organization_members,
    organizations as organizations_route,
    user_groups as user_groups_route,
)
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
)


def _ctx(
    *,
    user_id: int = 1,
    user_principal_id: int = 100,
    is_admin: bool = False,
    current_principal_id: int | None = None,
    org_role: OrgRole | None = None,
):
    ctx = MagicMock()
    ctx.user = MagicMock()
    ctx.user.id = user_id
    ctx.user.principal_id = user_principal_id
    ctx.is_platform_admin = is_admin
    ctx.current_principal_id = current_principal_id
    ctx.org_role = org_role
    return ctx


def _principal(
    id: int = 10,
    kind: PrincipalType = PrincipalType.ORG,
    display_name: str = "Acme",
    name: str | None = "acme",
):
    p = MagicMock(spec=Principal)
    p.id = id
    p.kind = kind
    p.name = name
    p.display_name = display_name
    p.description = None
    p.deleted_at = None
    p.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    p.updated_at = p.created_at
    return p


def _session_returning(*results):
    """Make a mock async session whose successive .exec() return the queued results."""
    session = MagicMock()
    queue = []
    for value in results:
        result = MagicMock()
        if isinstance(value, list):
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=value)
            result.scalars = MagicMock(return_value=scalars)
            result.first = MagicMock(return_value=value[0] if value else None)
            result.all = MagicMock(return_value=value)
        else:
            result.scalar_one_or_none = MagicMock(return_value=value)
            result.first = MagicMock(return_value=value)
        queue.append(result)
    session.exec = AsyncMock(side_effect=queue)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    return session


# ---- _can_manage (organization_members) ------------------------------------


def test_can_manage_platform_admin_always():
    ctx = _ctx(is_admin=True)
    assert organization_members._can_manage(ctx, 1) is True
    assert organization_members._can_manage(ctx, 99) is True


def test_can_manage_admin_in_org_can_manage():
    ctx = _ctx(current_principal_id=10, org_role=OrgRole.OWNER)
    assert organization_members._can_manage(ctx, 10) is True


def test_can_manage_admin_cannot_manage_other_org():
    ctx = _ctx(current_principal_id=10, org_role=OrgRole.OWNER)
    assert organization_members._can_manage(ctx, 99) is False


def test_can_manage_member_cannot_manage():
    ctx = _ctx(current_principal_id=10, org_role=OrgRole.MEMBER)
    assert organization_members._can_manage(ctx, 10) is False


# ---- organizations route ---------------------------------------------------


@pytest.mark.asyncio
async def test_create_organization_rejects_duplicate_name():
    # Non-GROUP principal already holding the name → reject.
    session = _session_returning(_principal(display_name="Existing", name="acme"))
    with pytest.raises(AlreadyExistsException):
        await organizations_route.create_organization(
            session=session,
            org_in=organizations_route.OrganizationCreate(
                display_name="Acme", name="acme"
            ),
        )


@pytest.mark.asyncio
async def test_delete_platform_org_blocked(monkeypatch):
    platform = _principal(id=1, display_name="Platform", name="default")
    monkeypatch.setattr(
        organizations_route.Principal,
        "one_by_id",
        AsyncMock(return_value=platform),
    )
    with pytest.raises(ConflictException):
        await organizations_route.delete_organization(session=MagicMock(), id=1)


@pytest.mark.asyncio
async def test_delete_org_blocked_when_resources_exist(monkeypatch):
    org = _principal(id=2, display_name="Acme", name="acme")
    monkeypatch.setattr(
        organizations_route.Principal,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    monkeypatch.setattr(
        organizations_route,
        "_has_resources",
        AsyncMock(return_value=["models", "api_keys"]),
    )
    with pytest.raises(ConflictException) as excinfo:
        await organizations_route.delete_organization(session=MagicMock(), id=2)
    assert "models" in excinfo.value.message


# ---- organization_members route -------------------------------------------


def _patch_org_and_member(monkeypatch, org, member_principal):
    """Both ``_load_org`` and ``_resolve_member_principal`` use
    ``Principal.one_by_id`` — order matches the route handler's calls.
    """
    monkeypatch.setattr(
        organization_members.Principal,
        "one_by_id",
        AsyncMock(side_effect=[org, member_principal]),
    )


@pytest.mark.asyncio
async def test_remove_only_owner_blocked(monkeypatch):
    org = _principal(id=10, display_name="Acme", name="acme")
    member = _principal(
        id=200, kind=PrincipalType.USER, display_name="user-2", name="user-2"
    )
    membership = MagicMock(spec=PrincipalMembership)
    membership.parent_principal_id = 10
    membership.member_principal_id = 200
    membership.role = OrgRole.OWNER
    membership.deleted_at = None
    _patch_org_and_member(monkeypatch, org, member)
    monkeypatch.setattr(
        organization_members,
        "_find_membership",
        AsyncMock(return_value=membership),
    )
    monkeypatch.setattr(
        organization_members,
        "_has_other_owner",
        AsyncMock(return_value=False),
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(ConflictException):
        await organization_members.remove_org_member(
            session=MagicMock(), ctx=ctx, org_id=10, principal_id=200
        )


@pytest.mark.asyncio
async def test_demote_only_owner_blocked(monkeypatch):
    org = _principal(id=10, display_name="Acme", name="acme")
    member = _principal(
        id=200, kind=PrincipalType.USER, display_name="user-2", name="user-2"
    )
    membership = MagicMock(spec=PrincipalMembership)
    membership.parent_principal_id = 10
    membership.member_principal_id = 200
    membership.role = OrgRole.OWNER
    membership.deleted_at = None
    _patch_org_and_member(monkeypatch, org, member)
    monkeypatch.setattr(
        organization_members,
        "_find_membership",
        AsyncMock(return_value=membership),
    )
    monkeypatch.setattr(
        organization_members,
        "_has_other_owner",
        AsyncMock(return_value=False),
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(ConflictException):
        await organization_members.update_org_member(
            session=MagicMock(),
            ctx=ctx,
            org_id=10,
            principal_id=200,
            body=organization_members.MembershipUpdate(role=OrgRole.MEMBER),
        )


@pytest.mark.asyncio
async def test_remove_only_owner_blocked_for_group_owner(monkeypatch):
    """Removing a GROUP-OWNER should be blocked just like a USER-OWNER
    when it's the last owner — the unified API treats both kinds
    symmetrically.
    """
    org = _principal(id=10, display_name="Acme", name="acme")
    group = _principal(
        id=300, kind=PrincipalType.GROUP, display_name=None, name="gpu-admins"
    )
    membership = MagicMock(spec=PrincipalMembership)
    membership.parent_principal_id = 10
    membership.member_principal_id = 300
    membership.role = OrgRole.OWNER
    membership.deleted_at = None
    _patch_org_and_member(monkeypatch, org, group)
    monkeypatch.setattr(
        organization_members,
        "_find_membership",
        AsyncMock(return_value=membership),
    )
    monkeypatch.setattr(
        organization_members,
        "_has_other_owner",
        AsyncMock(return_value=False),
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(ConflictException):
        await organization_members.remove_org_member(
            session=MagicMock(), ctx=ctx, org_id=10, principal_id=300
        )


@pytest.mark.asyncio
async def test_has_other_owner_recognises_user_owner():
    """A live USER-OWNER counts as another owner — the simple case."""
    session = _session_returning([1])  # the single-row "found" sentinel
    assert (
        await organization_members._has_other_owner(
            session, org_principal_id=10, exclude_member_principal_id=99
        )
        is True
    )


@pytest.mark.asyncio
async def test_has_other_owner_rejects_empty_group_owner():
    """A GROUP-OWNER whose group has no active user-members confers
    OWNER on nobody, so the guard must treat it as "no other owner".
    The query filters such groups out via the EXISTS subquery; an
    empty result set proves that filter fires.
    """
    session = _session_returning([])  # no rows survive the EXISTS filter
    assert (
        await organization_members._has_other_owner(
            session, org_principal_id=10, exclude_member_principal_id=99
        )
        is False
    )


# ---- cluster_access route --------------------------------------------------


@pytest.mark.asyncio
async def test_grant_cluster_access_validates_principal(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    monkeypatch.setattr(
        cluster_access_route.Principal,
        "one_by_id",
        AsyncMock(return_value=None),  # principal does not exist
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(InvalidException):
        await cluster_access_route.grant_cluster_access(
            session=MagicMock(),
            ctx=ctx,
            cluster_id=1,
            body=cluster_access_route.ClusterAccessGrant(
                principal_type=PrincipalType.ORG, principal_id=999
            ),
        )


@pytest.mark.asyncio
async def test_grant_cluster_access_rejects_duplicate(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    org = _principal(id=2, kind=PrincipalType.ORG, display_name="Acme", name="acme")
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    monkeypatch.setattr(
        cluster_access_route.Principal,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    session = _session_returning([MagicMock()])  # exec returns existing row
    ctx = _ctx(is_admin=True)
    with pytest.raises(AlreadyExistsException):
        await cluster_access_route.grant_cluster_access(
            session=session,
            ctx=ctx,
            cluster_id=1,
            body=cluster_access_route.ClusterAccessGrant(
                principal_type=PrincipalType.ORG, principal_id=2
            ),
        )


@pytest.mark.asyncio
async def test_revoke_cluster_access_404_when_missing(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    session = _session_returning(None)
    ctx = _ctx(is_admin=True)
    with pytest.raises(NotFoundException):
        await cluster_access_route.revoke_cluster_access(
            session=session,
            ctx=ctx,
            cluster_id=1,
            principal_id=42,
        )


# ---- user_groups route -----------------------------------------------------


@pytest.mark.asyncio
async def test_create_group_rejects_duplicate_name(monkeypatch):
    existing = _principal(
        id=5,
        kind=PrincipalType.GROUP,
        display_name=None,
        name="team-a",
    )
    monkeypatch.setattr(
        user_groups_route.Principal,
        "one_by_fields",
        AsyncMock(return_value=existing),
    )
    with pytest.raises(AlreadyExistsException):
        await user_groups_route.create_group(
            session=MagicMock(),
            body=user_groups_route.UserGroupCreate(name="team-a"),
        )


@pytest.mark.asyncio
async def test_add_group_members_rejects_missing_user(monkeypatch):
    group = _principal(
        id=5,
        kind=PrincipalType.GROUP,
        display_name=None,
        name="team-a",
    )
    monkeypatch.setattr(
        user_groups_route.Principal,
        "one_by_id",
        AsyncMock(return_value=group),
    )
    # Bulk user resolve returns []  → user 99 missing → NotFoundException.
    session = _session_returning([])
    with pytest.raises(NotFoundException):
        await user_groups_route.add_group_members(
            session=session,
            group_id=5,
            body=user_groups_route.GroupMembershipCreate(user_ids=[99]),
        )


# ---- _resolve_dashboard_scope ----------------------------------------------


def test_dashboard_scope_admin_all_mode_is_unscoped():
    # Platform admin without a pinned Org: aggregate everything, same as
    # before the multi-tenant refactor.
    ctx = _ctx(is_admin=True, current_principal_id=None)
    assert dashboard_route._resolve_dashboard_scope(ctx) is None


def test_dashboard_scope_admin_acting_as_org():
    # Platform admin acting inside an Org context: scope to that org.
    ctx = _ctx(is_admin=True, current_principal_id=10)
    assert dashboard_route._resolve_dashboard_scope(ctx) == 10


def test_dashboard_scope_org_owner_sees_their_org():
    ctx = _ctx(current_principal_id=10, org_role=OrgRole.OWNER)
    assert dashboard_route._resolve_dashboard_scope(ctx) == 10


def test_dashboard_scope_org_member_blocked():
    ctx = _ctx(current_principal_id=10, org_role=OrgRole.MEMBER)
    with pytest.raises(ForbiddenException):
        dashboard_route._resolve_dashboard_scope(ctx)


def test_dashboard_scope_non_admin_without_org_blocked():
    # Logged-in but no Org context and no admin flag — e.g. Personal
    # scope (org_role is never OWNER for the user's own USER-principal).
    ctx = _ctx(current_principal_id=None, org_role=None)
    with pytest.raises(ForbiddenException):
        dashboard_route._resolve_dashboard_scope(ctx)
