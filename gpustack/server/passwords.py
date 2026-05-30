"""Password credential helpers.

Encapsulates the ``user_passwords`` row + hashing. Routes / services
call the four helpers below instead of touching the password row
directly so the storage shape (algorithm marker, soft-delete column)
stays internal to this module.

Login credentials moved off the Principal row in the multi-tenancy
foundation migration; see :mod:`gpustack.schemas.user_passwords` for
the storage contract.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.user_passwords import (
    PasswordAlgorithm,
    UserPassword,
)
from gpustack.security import get_secret_hash, verify_hashed_secret


async def _get(session: AsyncSession, principal_id: int) -> Optional[UserPassword]:
    """Return the active password row for a principal, or None if none.

    Soft-deleted rows are excluded so a deleted-then-recreated password
    doesn't accidentally surface the old hash.
    """
    return await UserPassword.one_by_fields(
        session,
        {"owner_principal_id": principal_id, "deleted_at": None},
    )


async def set_password(
    session: AsyncSession,
    principal_id: int,
    plain: str,
    *,
    require_password_change: bool = False,
    auto_commit: bool = True,
) -> UserPassword:
    """Set (or replace) the principal's password.

    If the principal already has a row, the hash + require_change flag
    are updated in place — keeps the row's id stable so anything that
    referenced it (foreign keys, audit links) survives the change. New
    principal: insert a row.
    """
    hashed = get_secret_hash(plain)
    existing = await _get(session, principal_id)
    if existing is not None:
        existing.algorithm = PasswordAlgorithm.ARGON2
        existing.hashed_secret = hashed
        existing.require_password_change = require_password_change
        await existing.save(session, auto_commit=auto_commit)
        return existing
    row = UserPassword(
        owner_principal_id=principal_id,
        algorithm=PasswordAlgorithm.ARGON2,
        hashed_secret=hashed,
        require_password_change=require_password_change,
    )
    return await UserPassword.create(session, row, auto_commit=auto_commit)


async def verify_password(
    session: AsyncSession,
    principal_id: int,
    plain: str,
) -> bool:
    """Constant-time-ish password check. Returns False on any failure
    path (no row, wrong hash, unknown algorithm) so callers can treat
    the result uniformly without leaking which leg failed.
    """
    row = await _get(session, principal_id)
    if row is None:
        return False
    # Algorithm dispatch — today only argon2 is implemented; the marker
    # exists so legacy hashes from a future migration can verify with
    # their original comparator before being lazy-rehashed.
    if row.algorithm != PasswordAlgorithm.ARGON2:
        return False
    return verify_hashed_secret(row.hashed_secret, plain)


async def change_password(
    session: AsyncSession,
    principal_id: int,
    current: str,
    new: str,
    *,
    auto_commit: bool = True,
) -> bool:
    """Verify the current password and rotate to ``new`` on success.

    Returns False without writing anything if the current password
    doesn't verify. Clears ``require_password_change`` on success.
    """
    if not await verify_password(session, principal_id, current):
        return False
    await set_password(
        session,
        principal_id,
        new,
        require_password_change=False,
        auto_commit=auto_commit,
    )
    return True


async def is_password_change_required(session: AsyncSession, principal_id: int) -> bool:
    """Whether the principal must change password on next interaction.

    Returns False when there's no password row at all — those are SSO
    or system accounts that don't have a local password to rotate.
    """
    row = await _get(session, principal_id)
    return bool(row and row.require_password_change)


async def clear_require_password_change(
    session: AsyncSession,
    principal_id: int,
    *,
    auto_commit: bool = True,
) -> None:
    """Mark the require-change flag cleared. No-op when there's no row."""
    row = await _get(session, principal_id)
    if row is None or not row.require_password_change:
        return
    row.require_password_change = False
    row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await row.save(session, auto_commit=auto_commit)