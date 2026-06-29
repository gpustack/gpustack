"""v2.2.1 database changes

Bundles the pre-release schema tweaks for v2.2.1:

1. ``principals.source`` (and ``principal_memberships.source``) goes
   from the ``authproviderenum`` enum to ``VARCHAR(64)``, opening it
   up to provider values beyond the original ``Local`` / ``OIDC`` /
   ``SAML`` set (e.g. ``CAS`` and any other SSO kind added later)
   without a schema migration per provider. The ``AuthProviderEnum``
   Python class stays as a constants reference for the well-known
   values; only the DB-level enumeration goes away.

2. ``api_keys.owner_principal_id`` becomes nullable. An admin-created
   API key with no Org context needs to fall through to
   ``bypass_tenant_filter`` exactly like an admin cookie session, and
   that resolver branch only fires when the key carries a NULL owner.
   The previous platform-Org fallback left USER-personal and cross-Org
   resources unreachable.

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

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id',
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # NULL ``api_keys.owner_principal_id`` rows on the new schema
    # represent admin "All" mode keys. The pre-migration column was
    # NOT NULL with no usable DB-level default
    # (``default_factory=_platform_principal_id`` is Python-side and
    # resolves at app startup, not migration time), so backfill the
    # NULL rows to the platform Org before re-asserting NOT NULL.
    # The platform principal is seeded by the multi-tenancy
    # foundation migration and outlives this one. ``kind`` is the
    # ``principaltype`` enum on PG — cast to text so the equality is
    # against the plain ``'org'`` literal in both dialects.
    kind_expr = 'kind::text' if dialect == 'postgresql' else 'kind'
    platform_org_id = bind.execute(
        sa.text(
            f"SELECT id FROM principals "
            f"WHERE {kind_expr} = 'org' AND name = 'default' "
            f"AND deleted_at IS NULL"
        )
    ).scalar()
    if platform_org_id is not None:
        bind.execute(
            sa.text(
                "UPDATE api_keys SET owner_principal_id = :pid "
                "WHERE owner_principal_id IS NULL"
            ),
            {"pid": platform_org_id},
        )

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column(
            'owner_principal_id',
            existing_type=sa.Integer(),
            nullable=False,
        )

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
