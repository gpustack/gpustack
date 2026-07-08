"""v2.2.2 database changes

Bundles the pre-release schema tweaks for v2.2.2:

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

3. Backfill ``metered_usage.sku_count`` from the snapshotted shape.
   ``sku_count`` is the per-instance unit multiplier — GPU card count, or CPU
   base-flavor unit count (e.g. 3 for a 3c6g instance on a 1c2g flavor). Older
   metering wrote ``gpu_count or 1``, so every CPU row landed ``1`` regardless
   of size. The Instance Types breakdown groups by ``(sku, sku_count)``, so
   historical CPU rows would otherwise collapse all sizes into one row.
   Recompute it from each row's ``dimensions`` (live table + archive), in
   batches and Python-side so it is dialect-agnostic. GPU rows already stored
   the count, so in practice only CPU rows change.

4. Drop every foreign key on ``model_usages``, making the table fully FK-less
   (mirroring ``model_usage_details`` / ``metered_usage``). A usage row is an
   attribution / audit record that must outlive the entities it references;
   the previous ``ON DELETE SET NULL`` erased which user / tenant / model /
   route / key the usage belonged to on parent delete. Ids are now kept
   (dangling) and the read path resolves existence live, tagging gone
   entities ``(Deleted)``. Targets PostgreSQL and MySQL (SQLite unsupported).

Revision ID: c4d7e8f9a0b1
Revises: b2c3d4e5f6a7
Create Date: 2026-06-24 11:00:00.000000

"""
import json
from typing import Optional, Sequence, Union

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


# --- sku_count backfill (part 3) ---
_UPTIME_METER = 'instance.uptime'
_METERED_TABLES = ('metered_usage', 'metered_usage_archive')
_BATCH = 1000


# --- model_usages FK drop (part 4) ---
# (column, referred table) for every FK the table used to carry, all
# ``ON DELETE SET NULL``. Used to recreate them on downgrade.
_MODEL_USAGES_FKS = (
    ('user_id', 'principals'),
    ('owner_principal_id', 'principals'),
    ('consumer_principal_id', 'principals'),
    ('model_id', 'models'),
    ('model_route_id', 'model_routes'),
    ('provider_id', 'model_providers'),
    ('api_key_id', 'api_keys'),
)


def _coerce_dims(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _resolve_sku_count(dims: dict) -> int:
    """Unit multiplier from a stored ``dimensions`` blob — GPU card count, else
    CPU base-flavor unit count (mirrors
    gpustack.server.resource_usage_collector._resolve_sku_count; inlined so the
    migration carries no app-import dependency)."""

    def _int(v) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    gpu_count = _int(dims.get('gpu_count')) or 0
    if gpu_count > 0:
        return gpu_count
    for total_key, unit_key in (
        ('cpu_milli', 'unit_cpu_milli'),
        ('memory_mib', 'unit_memory_mib'),
    ):
        total = _int(dims.get(total_key))
        unit = _int(dims.get(unit_key))
        if total and unit and unit > 0:
            return max(1, round(total / unit))
    return 1


def _backfill_sku_count(bind) -> None:
    for table in _METERED_TABLES:
        if not sa.inspect(bind).has_table(table):
            continue
        rows = bind.execute(
            sa.text(
                f"SELECT id, sku_count, dimensions FROM {table} "
                f"WHERE meter_key = :m"
            ),
            {"m": _UPTIME_METER},
        ).fetchall()

        updates = [
            {"id": r[0], "c": _resolve_sku_count(_coerce_dims(r[2]))}
            for r in rows
        ]
        updates = [u for u, r in zip(updates, rows) if u["c"] != r[1]]

        stmt = sa.text(f"UPDATE {table} SET sku_count = :c WHERE id = :id")
        for i in range(0, len(updates), _BATCH):
            bind.execute(stmt, updates[i : i + _BATCH])

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
        
    # Part 3: backfill sku_count (all dialects).
    _backfill_sku_count(bind)

    # Part 4: drop every FK on model_usages — the table becomes fully FK-less.
    # Constraint names are resolved by reflection (the ``user_id`` FK kept its
    # pre-rename name ``fk_model_usages_user_id_users`` after users→principals,
    # while the rest follow ``fk_model_usages_<col>_<referred>``).
    for fk in sa.inspect(bind).get_foreign_keys('model_usages'):
        if fk.get('name'):
            op.drop_constraint(fk['name'], 'model_usages', type_='foreignkey')


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Revert part 4: recreate the model_usages foreign keys as ON DELETE SET
    # NULL, pointing at the (post-rename) parent tables.
    #
    # While FK-less (the upgraded state), deleting a parent left its id dangling
    # on model_usages. PostgreSQL / MySQL validate existing rows when a FK is
    # added, so ADD CONSTRAINT would fail on those orphaned references — NULL
    # them out first (exactly what ON DELETE SET NULL would have done) so the
    # constraint can be created.
    for column, referred in _MODEL_USAGES_FKS:
        bind.execute(
            sa.text(
                f"UPDATE model_usages SET {column} = NULL "
                f"WHERE {column} IS NOT NULL "
                f"AND {column} NOT IN (SELECT id FROM {referred})"
            )
        )
        op.create_foreign_key(
            f'fk_model_usages_{column}_{referred}',
            'model_usages',
            referred,
            [column],
            ['id'],
            ondelete='SET NULL',
        )

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
            
    # Part 3 (sku_count backfill) is an irreversible data fix — the original
    # (mostly ``1``) values are not recoverable and were wrong anyway, so the
    # downgrade only reverts the principal.source column type.
