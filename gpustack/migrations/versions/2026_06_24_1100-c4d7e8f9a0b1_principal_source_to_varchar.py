"""principal.source enum -> VARCHAR(64)

Opens up ``principals.source`` (and ``principal_memberships.source``) for
provider values beyond the original ``Local`` / ``OIDC`` / ``SAML`` set —
e.g. ``CAS`` and any other SSO kind added later — without a schema
migration per provider. The ``AuthProviderEnum`` Python class stays as a
constants reference for the well-known values; only the DB-level
enumeration goes away.

Revision ID: c4d7e8f9a0b1
Revises: b2c3d4e5f6a7
Create Date: 2026-06-24 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4d7e8f9a0b1'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = ('principals', 'principal_memberships')
_ENUM_NAME = 'authproviderenum'
_ENUM_VALUES = ('Local', 'OIDC', 'SAML')


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == 'postgresql':
        # ALTER COLUMN TYPE can't see through the default, so drop the
        # default, swap the type, then put the default back. After both
        # columns are off the enum type, drop it.
        for table in _TABLES:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source DROP DEFAULT"
            )
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source "
                f"TYPE VARCHAR(64) USING source::text"
            )
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source SET DEFAULT 'Local'"
            )
        op.execute(f"DROP TYPE IF EXISTS {_ENUM_NAME}")
    elif dialect == 'mysql':
        existing_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)
        for table in _TABLES:
            op.alter_column(
                table,
                'source',
                existing_type=existing_enum,
                type_=sa.String(length=64),
                existing_nullable=False,
                existing_server_default='Local',
            )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Scrub any rows whose ``source`` is outside the original
    # {Local, OIDC, SAML} set (e.g. ``CAS`` rows created on the new
    # schema) before narrowing the column back to ``authproviderenum``
    # — without this, the type cast crashes the downgrade. Reverting
    # to ``Local`` preserves the user row (versus deleting); the user
    # loses SSO-source attribution but their grants / memberships
    # survive a possible re-upgrade.
    values_sql = ", ".join(f"'{v}'" for v in _ENUM_VALUES)
    for table in _TABLES:
        op.execute(
            f"UPDATE {table} SET source = 'Local' "
            f"WHERE source NOT IN ({values_sql})"
        )

    if dialect == 'postgresql':
        op.execute(f"CREATE TYPE {_ENUM_NAME} AS ENUM ({values_sql})")
        for table in _TABLES:
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source DROP DEFAULT"
            )
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source "
                f"TYPE {_ENUM_NAME} USING source::text::{_ENUM_NAME}"
            )
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN source SET DEFAULT 'Local'"
            )
    elif dialect == 'mysql':
        new_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)
        for table in _TABLES:
            op.alter_column(
                table,
                'source',
                existing_type=sa.String(length=64),
                type_=new_enum,
                existing_nullable=False,
                existing_server_default='Local',
            )
