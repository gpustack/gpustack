"""Credential schema + password-credential helpers.

The ``credentials`` table holds two flavors of authentication material:

- **Asymmetric keypairs** (``credential_type`` of SSH / CA / X509) —
  ``public_key`` and base64-encoded private key in ``encoded_secret``,
  plus ``options`` carrying the key spec (``algorithm`` / ``length``).
  Owned by the principal that owns the worker / cluster using them.
- **Password hashes** (``credential_type=PASSWORD``) — bcrypt hash in
  ``encoded_secret``, ``public_key`` is NULL, and ``options`` carries
  ``{"require_password_change": bool}``. Owned by the USER-principal
  whose login they authenticate.

A single principal has at most one active PASSWORD row at a time;
rows are soft-deleted via ``deleted_at``. The helpers at the bottom of
this module wrap the lookup / create / update / verify dance so
callers don't have to think about row identity or soft-deletion
semantics.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel
from sqlalchemy import and_
import sqlalchemy as sa
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    JSON,
    SQLModel,
    String,
    Text,
    select,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin
from gpustack.security import get_secret_hash, verify_hashed_secret


class CredentialType(str, Enum):
    # SSH / CA / X509 hold asymmetric keypairs (public_key + encoded_secret).
    # PASSWORD holds a bcrypt hash in encoded_secret with public_key NULL.
    #
    # NOTE: SQLAlchemy/SQLModel stores Enum-typed columns using the
    # member *name* (uppercase), not the value. The DB-level
    # ``credentialtype`` enum therefore contains ``SSH/CA/X509/PASSWORD``;
    # only the lowercase ``.value`` form appears in Pydantic JSON output.
    SSH = "ssh"
    CA = "ca"
    X509 = "x509"
    PASSWORD = "password"


class SSHKeyOptions(BaseModel):
    algorithm: str = Field(default="RSA")
    length: int = Field(default=2048)


class CredentialBase(SQLModel):
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    credential_type: CredentialType = Field(default=CredentialType.SSH)
    # The principal that owns this credential. SSH rows are owned by
    # whichever principal owns the worker / cluster that uses them
    # (typically populated from `worker.owner_principal_id`); PASSWORD
    # rows are owned by the USER principal whose login they
    # authenticate.
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
    )
    # PEM public key for asymmetric credentials; NULL for PASSWORD.
    public_key: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Generic "secret payload": base64-encoded private key for SSH/CA/X509,
    # bcrypt hash for PASSWORD.
    encoded_secret: str = Field(default="", sa_column=Column(Text, nullable=False))
    # JSON bag for credential-type-specific options.
    #   SSH:      {"algorithm": "...", "length": ...}
    #   PASSWORD: {"require_password_change": bool}
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )


class Credential(CredentialBase, BaseModelMixin, table=True):
    __tablename__ = "credentials"
    __table_args__ = (sa.Index("idx_credentials_external_id", "external_id"),)
    id: Optional[int] = Field(default=None, primary_key=True)

    @property
    def ssh_key_options(self) -> Optional[SSHKeyOptions]:
        """Back-compat accessor for SSH rows that historically used a
        typed Pydantic model on this column. Returns None for non-SSH
        rows or when `options` is unset."""
        if self.credential_type != CredentialType.SSH or not self.options:
            return None
        return SSHKeyOptions(**self.options)


# ---- PASSWORD-credential helpers ----------------------------------------
#
# Helpers below operate on PASSWORD-typed rows owned by a USER principal.
# Schema reminders:
# - One active PASSWORD row per ``owner_principal_id``; ``deleted_at``
#   retires old rows. Only ``deleted_at IS NULL`` rows count as current.
# - The bcrypt hash lives in ``encoded_secret``; ``public_key`` is NULL.
# - ``options`` carries ``require_password_change`` (bool).


async def get_password_credential(
    session: AsyncSession, principal_id: int
) -> Optional[Credential]:
    """Fetch the active PASSWORD credential for a principal, or None."""
    stmt = select(Credential).where(
        and_(
            Credential.owner_principal_id == principal_id,
            Credential.credential_type == CredentialType.PASSWORD,
            Credential.deleted_at.is_(None),
        )
    )
    return (await session.exec(stmt)).first()


async def verify_password(
    session: AsyncSession, principal_id: int, raw_password: str
) -> bool:
    """Constant-time check that ``raw_password`` matches the stored hash."""
    credential = await get_password_credential(session, principal_id)
    if credential is None or not credential.encoded_secret:
        return False
    return verify_hashed_secret(credential.encoded_secret, raw_password)


async def set_password(
    session: AsyncSession,
    principal_id: int,
    raw_password: str,
    *,
    require_password_change: bool = False,
    auto_commit: bool = False,
) -> Credential:
    """Set (create or update in place) the PASSWORD credential.

    In-place update so password history is not retained on this row —
    if a future PR adds "no-reuse-of-last-N" policy, switch to insert
    new + soft-delete old (and add a partial unique index, see
    discussion in the design doc).
    """
    credential = await get_password_credential(session, principal_id)
    hashed = get_secret_hash(raw_password)
    options = {"require_password_change": bool(require_password_change)}
    if credential is None:
        credential = Credential(
            credential_type=CredentialType.PASSWORD,
            owner_principal_id=principal_id,
            public_key=None,
            encoded_secret=hashed,
            options=options,
        )
        await Credential.create(session, credential, auto_commit=auto_commit)
    else:
        await credential.update(
            session,
            {"encoded_secret": hashed, "options": options},
            auto_commit=auto_commit,
        )
    return credential


async def clear_require_password_change(
    session: AsyncSession,
    principal_id: int,
    *,
    auto_commit: bool = False,
) -> None:
    """Flip ``require_password_change`` off on the current PASSWORD row."""
    credential = await get_password_credential(session, principal_id)
    if credential is None:
        return
    options = dict(credential.options or {})
    if not options.get("require_password_change"):
        return
    options["require_password_change"] = False
    await credential.update(session, {"options": options}, auto_commit=auto_commit)


def require_password_change(credential: Optional[Credential]) -> bool:
    """Read the ``require_password_change`` flag without another DB hit."""
    if credential is None or not credential.options:
        return False
    return bool(credential.options.get("require_password_change", False))
