"""metered usage and usage events

Revision ID: b2c3d4e5f6a7
Revises: 61929acb0676
Create Date: 2026-05-29 10:00:00.000000

Creates the tables behind the unified resource metering framework:

* ``metered_usage``          — unified hourly rollup of time-based resources
                               (GPU/CPU instances, persistent volumes)
* ``metered_usage_archive``  — cold archive of ``metered_usage``
* ``resource_events``           — event-level lifecycle/audit log (hot)
* ``resource_events_archive``   — cold archive of ``resource_events``

``metered_usage`` buckets by the hour (``bucket_start``); coarser granularities
are derived at query time. owner/creator/cluster id columns are FK-less plain
integers — this is an audit/billing row, so deleting a principal / cluster must
NOT null out the attribution (``SET NULL`` would erase "who pays" and make the
row untraceable). ids stay as reported; the ``*_name`` snapshots keep rows
human-readable after the parent is gone. ``resource_id`` is likewise FK-less
(polymorphic across instance / volume id spaces). The natural key
``(meter_key, resource_id, bucket_start)`` is all non-null so the collector's
idempotent upsert works even when ``owner_principal_id`` is NULL
(K8s-direct-created resources).

The ``*_archive`` tables mirror their hot table's column layout exactly (for
positional ``INSERT ... SELECT``); their id columns are non-autoincrement and
their id columns are FK-less. No price columns: the metering layer is
pricing-agnostic.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from gpustack.schemas.common import UTCDateTime

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = '61929acb0676'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Column factories — hot + archive must stay column-identical for the bulk
# INSERT ... SELECT archival path.
# ---------------------------------------------------------------------------

def _metered_usage_columns() -> list:
    return [
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),

        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('owner_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('consumer_principal_id', sa.Integer(), nullable=True),
        sa.Column('consumer_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('creator_id', sa.Integer(), nullable=True),
        sa.Column('creator_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('cluster_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),

        sa.Column('meter_key', sa.String(length=64), nullable=False),
        sa.Column('resource_type', sa.String(length=32), nullable=False),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('resource_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column('resource_display_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),

        sa.Column('sku', sqlmodel.sql.sqltypes.AutoString(length=128), nullable=True),
        sa.Column('sku_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('dimensions', gpustack.schemas.common.JSON(), nullable=True),

        sa.Column('bucket_start', UTCDateTime(), nullable=False),
        sa.Column('quantity', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('unit', sa.String(length=32), nullable=False),

        sa.Column('settled_until', UTCDateTime(), nullable=True),
        sa.Column('sealed_at', UTCDateTime(), nullable=True),

        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
    ]


def _create_metered_usage_indexes(table_name: str) -> None:
    op.create_index(f'ix_{table_name}_bucket_start',
                    table_name, ['bucket_start'], unique=False)
    op.create_index(f'ix_{table_name}_consumer_principal_id_bucket',
                    table_name, ['consumer_principal_id', 'bucket_start'], unique=False)
    op.create_index(f'ix_{table_name}_resource_type_meter_bucket',
                    table_name, ['resource_type', 'meter_key', 'bucket_start'], unique=False)
    op.create_index(f'ix_{table_name}_sku_bucket',
                    table_name, ['sku', 'bucket_start'], unique=False)
    op.create_index(f'ix_{table_name}_creator_id_bucket',
                    table_name, ['creator_id', 'bucket_start'], unique=False)


def _resource_event_columns() -> list:
    return [
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('occurred_at', UTCDateTime(), nullable=False),

        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('owner_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('consumer_principal_id', sa.Integer(), nullable=True),
        sa.Column('consumer_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column('creator_id', sa.Integer(), nullable=True),
        sa.Column('creator_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),

        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('cluster_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),

        sa.Column('resource_type', sa.String(length=32), nullable=False),
        sa.Column('resource_id', sa.Integer(), nullable=True),
        sa.Column('resource_name', sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),

        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('event_message', sqlmodel.sql.sqltypes.AutoString(length=1024), nullable=True),
        sa.Column('phase', sqlmodel.sql.sqltypes.AutoString(length=64), nullable=True),

        sa.Column('spec_snapshot', gpustack.schemas.common.JSON(), nullable=True),

        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
    ]


def _create_resource_event_indexes(table_name: str) -> None:
    op.create_index(
        f'ix_{table_name}_occurred_at', table_name, ['occurred_at'], unique=False
    )
    op.create_index(
        f'ix_{table_name}_owner_principal_id', table_name, ['owner_principal_id'], unique=False
    )
    op.create_index(
        f'ix_{table_name}_creator_id', table_name, ['creator_id'], unique=False
    )
    op.create_index(
        f'ix_{table_name}_resource_lookup',
        table_name,
        ['resource_type', 'resource_id', 'occurred_at'],
        unique=False,
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # ----- metered_usage (hot) ---------------------------------------------
    op.create_table(
        'metered_usage',
        *_metered_usage_columns(),
        # FK-less owner/creator/cluster ids — audit/billing rows must outlive
        # parent deletion (a SET NULL would erase the payer attribution).
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('meter_key', 'resource_id', 'bucket_start',
                            name='uq_metered_usage'),
    )
    _create_metered_usage_indexes('metered_usage')

    # ----- metered_usage_archive (cold) ------------------------------------
    mu_archive_cols = _metered_usage_columns()
    mu_archive_cols[0] = sa.Column('id', sa.Integer(), nullable=False, autoincrement=False)
    op.create_table(
        'metered_usage_archive',
        *mu_archive_cols,
        sa.PrimaryKeyConstraint('id'),
    )
    _create_metered_usage_indexes('metered_usage_archive')

    # ----- resource_events (hot) ----------------------------------------------
    op.create_table(
        'resource_events',
        *_resource_event_columns(),
        sa.PrimaryKeyConstraint('id'),
    )
    _create_resource_event_indexes('resource_events')

    # ----- resource_events_archive (cold) -------------------------------------
    ue_archive_cols = _resource_event_columns()
    ue_archive_cols[0] = sa.Column('id', sa.Integer(), nullable=False, autoincrement=False)
    op.create_table(
        'resource_events_archive',
        *ue_archive_cols,
        sa.PrimaryKeyConstraint('id'),
    )
    _create_resource_event_indexes('resource_events_archive')


def downgrade() -> None:
    for t in ('resource_events_archive', 'resource_events'):
        op.drop_index(f'ix_{t}_resource_lookup', table_name=t)
        op.drop_index(f'ix_{t}_creator_id', table_name=t)
        op.drop_index(f'ix_{t}_owner_principal_id', table_name=t)
        op.drop_index(f'ix_{t}_occurred_at', table_name=t)
        op.drop_table(t)

    for t in ('metered_usage_archive', 'metered_usage'):
        op.drop_index(f'ix_{t}_creator_id_bucket', table_name=t)
        op.drop_index(f'ix_{t}_sku_bucket', table_name=t)
        op.drop_index(f'ix_{t}_resource_type_meter_bucket', table_name=t)
        op.drop_index(f'ix_{t}_consumer_principal_id_bucket', table_name=t)
        op.drop_index(f'ix_{t}_bucket_start', table_name=t)
        op.drop_table(t)
