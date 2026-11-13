"""v2.3.1 database changes.

Persist model launch configuration and index instance membership lookups.
Meter image generation in images and pixels on the model usage tables.

Revision ID: c8d3e6f1a902
Revises: a7e2c9d4f681
Create Date: 2026-10-30 16:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


revision: str = 'c8d3e6f1a902'
down_revision: Union[str, None] = 'a7e2c9d4f681'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables that carry per-request or rolled-up model usage quantities.
_USAGE_TABLES = (
    'model_usages',
    'model_usage_details',
    'model_usage_details_archive',
)
_IMAGE_USAGE_COLUMNS = ('image_count', 'image_pixel_count')


def upgrade() -> None:
    if table_exists('model_instances') and not column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.add_column(
            'model_instances',
            sa.Column('model_config_snapshot', sa.JSON(), nullable=True),
        )

    _upgrade_model_instance_lookup_indexes()
    _upgrade_image_usage_columns()


def downgrade() -> None:
    _downgrade_image_usage_columns()
    _downgrade_model_instance_lookup_indexes()
    if table_exists('model_instances') and column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.drop_column('model_instances', 'model_config_snapshot')


_MODEL_INSTANCE_LOOKUP_FIELDS = ('cluster_id', 'model_id')


def _upgrade_model_instance_lookup_indexes() -> None:
    if not table_exists('model_instances'):
        return
    indexes = sa.inspect(op.get_bind()).get_indexes('model_instances')
    for field in _MODEL_INSTANCE_LOOKUP_FIELDS:
        # MySQL-compatible foreign keys can already own a supporting index.
        # Keep it instead of replacing it with a duplicate that downgrade
        # could not drop while the foreign key still depends on it.
        if not any(ix['column_names'][:1] == [field] for ix in indexes):
            op.create_index(f'ix_model_instances_{field}', 'model_instances', [field])


def _downgrade_model_instance_lookup_indexes() -> None:
    if not table_exists('model_instances'):
        return
    indexes = sa.inspect(op.get_bind()).get_indexes('model_instances')
    names = {ix['name'] for ix in indexes}
    foreign_keys = (
        sa.inspect(op.get_bind()).get_foreign_keys('model_instances')
        if op.get_bind().dialect.name == 'mysql'
        else []
    )
    for field in _MODEL_INSTANCE_LOOKUP_FIELDS:
        name = f'ix_model_instances_{field}'
        if name in names:
            # MySQL/OceanBase cannot drop the last supporting FK index. Keep
            # that index when necessary; it changes no data or constraints.
            if any(fk['constrained_columns'] == [field] for fk in foreign_keys) and not any(
                ix['name'] != name and ix['column_names'][:1] == [field]
                for ix in indexes
            ):
                continue
            op.drop_index(name, table_name='model_instances')


def _upgrade_image_usage_columns() -> None:
    for table in _USAGE_TABLES:
        if not table_exists(table):
            continue
        for column in _IMAGE_USAGE_COLUMNS:
            if column_exists(table, column):
                continue
            # ``server_default`` backfills existing rows to 0 on every
            # supported database without a separate UPDATE.
            op.add_column(
                table,
                sa.Column(
                    column, sa.BigInteger(), nullable=False, server_default='0'
                ),
            )


def _downgrade_image_usage_columns() -> None:
    for table in _USAGE_TABLES:
        if not table_exists(table):
            continue
        for column in _IMAGE_USAGE_COLUMNS:
            if column_exists(table, column):
                op.drop_column(table, column)
