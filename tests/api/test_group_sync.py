"""Unit tests for IdP group-sync.

The sync helper lives on the services layer and operates on the
``principal_memberships`` table; tests mock the AsyncSession so we
can exercise the diff/add/remove decision tree without standing up a
real DB.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from gpustack.api.exceptions import InternalServerErrorException
from gpustack.routes.auth import _coerce_group_claim
from gpustack.schemas.principals import (
    Principal,
    PrincipalMembership,
    PrincipalType,
)
from gpustack.schemas.users import AuthProviderEnum
from gpustack.server.services import sync_user_group_memberships


# ---- _coerce_group_claim ---------------------------------------------------


def test_coerce_none_is_empty():
    assert _coerce_group_claim(None) == []


def test_coerce_empty_string_is_empty():
    assert _coerce_group_claim("") == []


def test_coerce_single_string_wraps():
    assert _coerce_group_claim("engineering") == ["engineering"]


def test_coerce_csv_string_splits():
    assert _coerce_group_claim("a,b, c ") == ["a", "b", "c"]


def test_coerce_list_passthrough():
    assert _coerce_group_claim(["x", "y"]) == ["x", "y"]


def test_coerce_list_drops_blanks_and_trims():
    assert _coerce_group_claim(["alpha", "", "  ", "beta "]) == ["alpha", "beta"]


def test_coerce_non_string_member_skipped():
    # Some IdPs return numeric group ids; we drop those (Group.name is
    # a string identifier).
    assert _coerce_group_claim(["alpha", 42, None, "beta"]) == ["alpha", "beta"]


def test_coerce_non_string_scalar_is_empty():
    # Same rule as list members: a bare non-string scalar is dropped,
    # not stringified. Group identifiers are required to come in as
    # strings from the IdP — auto-coercing ``42`` to ``"42"`` would
    # silently turn a misconfigured claim into a phantom Group name.
    assert _coerce_group_claim(42) == []
    assert _coerce_group_claim(True) == []


# ---- sync_user_group_memberships -------------------------------------------


def _principal(id: int, name: str) -> Principal:
    """Make a Group-principal Mock that quacks like a real row."""
    p = MagicMock(spec=Principal)
    p.id = id
    p.kind = PrincipalType.GROUP
    p.name = name
    p.deleted_at = None
    return p


def _membership(
    parent_id: int,
    member_id: int,
    source: AuthProviderEnum = AuthProviderEnum.OIDC,
    deleted: bool = False,
) -> PrincipalMembership:
    m = MagicMock(spec=PrincipalMembership)
    m.parent_principal_id = parent_id
    m.member_principal_id = member_id
    m.source = source
    m.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None) if deleted else None
    return m


def _session(exec_results, *, flush_raises=None):
    """Mock AsyncSession where successive ``exec`` calls return queued results.

    Each item is a list of rows (.all() returns it). ``first()``
    returns the first row of the list (or None when empty), matching
    real SQLModel result behavior.

    ``flush_raises``: optional list, one entry per ``flush`` call. A
    non-None entry is raised as the side effect of that call. Lets a
    test simulate an ``IntegrityError`` from the partial unique index.

    Also mocks ``begin_nested`` as an async context manager so the
    SAVEPOINT-wrapped insert path in ``sync_user_group_memberships``
    can be exercised. ``__aexit__`` returns False so any exception
    raised inside the ``async with`` block propagates — matching real
    Session behavior where SAVEPOINT rollback happens but the
    exception still surfaces.
    """
    session = MagicMock()
    results = []
    for value in exec_results:
        result = MagicMock()
        result.all = MagicMock(return_value=value)
        result.first = MagicMock(return_value=value[0] if value else None)
        results.append(result)
    session.exec = AsyncMock(side_effect=results)
    session.add = MagicMock()
    if flush_raises is None:
        session.flush = AsyncMock()
    else:
        session.flush = AsyncMock(side_effect=flush_raises)

    nested_ctx = MagicMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_ctx)
    return session


@pytest.mark.asyncio
async def test_sync_rejects_local_provider():
    """Sync is for IdP providers only — Local would be a code mistake."""
    session = _session([])
    with pytest.raises(InternalServerErrorException):
        await sync_user_group_memberships(
            session,
            user_principal_id=100,
            provider=AuthProviderEnum.Local,
            group_names=["x"],
        )


@pytest.mark.asyncio
async def test_sync_no_groups_no_existing_is_noop():
    """Empty claim and no prior memberships: nothing inserted, nothing flagged."""
    # exec calls: groups-by-name (skipped when names empty), then
    # owned-rows lookup.
    session = _session([[]])  # owned_rows is empty
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=[],
    )
    session.add.assert_not_called()
    session.flush.assert_not_called()


@pytest.mark.asyncio
async def test_sync_adds_membership_against_existing_group(monkeypatch):
    """User joins a Group that already exists in the system."""
    grp = _principal(5, "engineering")
    session = _session(
        [
            [grp],  # existing groups matching name
            [],  # no owned rows yet
        ]
    )
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=["engineering"],
    )
    # Exactly one new PrincipalMembership added.
    added = [c.args[0] for c in session.add.call_args_list]
    assert len(added) == 1
    new_row = added[0]
    assert isinstance(new_row, PrincipalMembership)
    assert new_row.parent_principal_id == 5
    assert new_row.member_principal_id == 100
    assert new_row.source == AuthProviderEnum.OIDC


@pytest.mark.asyncio
async def test_sync_auto_creates_unknown_group():
    """IdP pushes a name we've never seen → create the Group on the fly."""
    session = _session(
        [
            [],  # no existing groups match
            [],  # no owned memberships
        ]
    )
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=["brand-new"],
    )
    added_kinds = [type(c.args[0]).__name__ for c in session.add.call_args_list]
    # First add is the new Principal (Group), then the membership.
    assert added_kinds.count("Principal") == 1
    assert added_kinds.count("PrincipalMembership") == 1


@pytest.mark.asyncio
async def test_sync_removes_membership_dropped_by_idp():
    """Group not in claim but previously OIDC-synced → soft-delete it."""
    stale = _membership(parent_id=5, member_id=100, source=AuthProviderEnum.OIDC)
    # Empty claim → service skips the groups-by-name query entirely;
    # only the owned-rows query runs.
    session = _session([[stale]])
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=[],
    )
    # stale membership was soft-deleted (deleted_at populated).
    assert stale.deleted_at is not None


@pytest.mark.asyncio
async def test_sync_does_not_touch_local_memberships():
    """Admin-added (source=Local) memberships are owned by the admin, not
    the IdP — sync queries are scoped to ``source=provider`` so the Local
    row never enters the diff. Verify by passing an empty owned-rows
    list and checking nothing the function flags would even reach a
    Local-sourced row (this is structurally enforced by the WHERE
    clause; we just exercise the path).
    """
    session = _session([[]])  # owned rows scoped to OIDC = empty
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=[],
    )
    # No add / flush / mutation — the Local-sourced row exists in the
    # DB but the sync function never sees it.
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_sync_revives_soft_deleted_row():
    """User dropped from a group then re-added — revive the audit row
    instead of inserting a duplicate."""
    grp = _principal(5, "engineering")
    revivable = _membership(
        parent_id=5,
        member_id=100,
        source=AuthProviderEnum.OIDC,
        deleted=True,
    )
    session = _session(
        [
            [grp],  # group still exists
            [revivable],  # previously soft-deleted OIDC row
        ]
    )
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=["engineering"],
    )
    # Row was revived (deleted_at cleared), not a fresh insert.
    assert revivable.deleted_at is None
    inserted = [
        c
        for c in session.add.call_args_list
        if isinstance(c.args[0], PrincipalMembership) and c.args[0] is not revivable
    ]
    assert inserted == []


@pytest.mark.asyncio
async def test_sync_race_recovers_when_concurrent_insert_wins():
    """Two concurrent first-time logins push the same new group name.
    One transaction inserts, the other's flush raises IntegrityError
    inside the SAVEPOINT; the loser re-fetches the winning row and
    proceeds. The sync must not surface the IntegrityError to the
    caller.
    """
    winner = _principal(7, "engineering")
    session = _session(
        [
            [],  # initial existing-by-name lookup: empty (race not yet resolved)
            [winner],  # SAVEPOINT-recovery re-fetch finds the winner's row
            [],  # owned_rows for this user: empty
        ],
        flush_raises=[IntegrityError("dup", None, Exception())],
    )
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=["engineering"],
    )
    # Membership row inserted against the winner's group id.
    inserted = [
        c.args[0]
        for c in session.add.call_args_list
        if isinstance(c.args[0], PrincipalMembership)
    ]
    assert len(inserted) == 1
    assert inserted[0].parent_principal_id == 7


@pytest.mark.asyncio
async def test_sync_race_surfaces_when_refetch_finds_nothing():
    """Pathological case: IntegrityError raised but no row visible on
    re-read. We surface the original error rather than silently
    skipping the group — the caller's transaction needs to roll back.
    """
    session = _session(
        [
            [],  # initial existing-by-name lookup: empty
            [],  # re-fetch after IntegrityError finds nothing
        ],
        flush_raises=[IntegrityError("dup", None, Exception())],
    )
    with pytest.raises(IntegrityError):
        await sync_user_group_memberships(
            session,
            user_principal_id=100,
            provider=AuthProviderEnum.OIDC,
            group_names=["mystery"],
        )


@pytest.mark.asyncio
async def test_sync_dedupes_input_names():
    """Same group name appearing twice in the claim should not produce
    two memberships."""
    grp = _principal(5, "engineering")
    session = _session(
        [
            [grp],  # only one group exists, even though name appears twice
            [],  # no owned rows
        ]
    )
    await sync_user_group_memberships(
        session,
        user_principal_id=100,
        provider=AuthProviderEnum.OIDC,
        group_names=["engineering", "engineering"],
    )
    inserted = [
        c.args[0]
        for c in session.add.call_args_list
        if isinstance(c.args[0], PrincipalMembership)
    ]
    assert len(inserted) == 1
