"""UserPassword — login-credential storage decoupled from Principal.

A USER's hashed password used to live inline on ``principals`` (legacy
``users``) as ``hashed_password`` + ``require_password_change``. Pulling
it out into its own row keeps Principal as pure identity and leaves room
to evolve algorithms (currently bcrypt, future pbkdf2 / argon2) without
breaking existing rows — the chosen algorithm is recorded on each row
via :attr:`algorithm` and read at verify time.

One active row per USER principal — enforced at the application layer
by :mod:`gpustack.server.passwords`, which checks for an existing active
row before inserting and updates the hash in place when one is found.
No DB-level UNIQUE on ``owner_principal_id``: that would collide with
the soft-delete pattern (rows kept with ``deleted_at`` set), and a
partial unique index scoped to ``deleted_at IS NULL`` needs dialect-
specific syntax that doesn't translate cleanly across PG / MySQL /
OceanBase / openGauss.

Not modeled as a credential-vault that also holds SSH keys / CAs —
those have completely different access patterns (SSH keys are written
once and read by GPU-instance provisioning; passwords are read on
every login) and lifecycles. The SSH ``credentials`` table stays
SSH-only.
"""

from enum import Enum
from typing import Optional

from sqlalchemy import Enum as SQLEnum
from sqlmodel import Column, Field, ForeignKey, Integer, SQLModel

from gpustack.mixins import BaseModelMixin


class PasswordAlgorithm(str, Enum):
    """Algorithm used to hash the stored secret.

    Stored on each row so legacy hashes can coexist with newer ones if
    the project ever swaps in a different hasher — the verify path
    reads this marker to pick the right comparator and can lazy-rehash
    on next successful login without a schema break.

    Follows :class:`PrincipalType` / :class:`OrgRole` convention:
    member ``name`` is uppercase (this is what SQLAlchemy sends to
    the DB by default, and what the DB enum literal must match),
    ``value`` is lowercase (Pydantic / wire / display).
    """

    ARGON2 = "argon2"


class UserPasswordBase(SQLModel):
    owner_principal_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    algorithm: PasswordAlgorithm = Field(
        default=PasswordAlgorithm.ARGON2,
        sa_column=Column(
            SQLEnum(PasswordAlgorithm),
            nullable=False,
            # ``.name`` (not ``.value``) — the DB enum literal is the
            # uppercase Python name; ``.value`` is the lowercase
            # Pydantic / display form.
            server_default=PasswordAlgorithm.ARGON2.name,
        ),
    )
    hashed_secret: str = Field(nullable=False)
    # Admin reset / bootstrap flow marks the next login as
    # forced-change; the change-password route clears it on success.
    require_password_change: bool = Field(default=False, nullable=False)


class UserPassword(UserPasswordBase, BaseModelMixin, table=True):
    __tablename__ = "user_passwords"

    id: Optional[int] = Field(default=None, primary_key=True)
