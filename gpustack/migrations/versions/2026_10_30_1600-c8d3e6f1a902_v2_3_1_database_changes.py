"""v2.3.1 database changes.

Persist model configuration used to launch each instance.
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


def downgrade() -> None:
    for table in _USAGE_TABLES:
        if not table_exists(table):
            continue
        for column in _IMAGE_USAGE_COLUMNS:
            if column_exists(table, column):
                op.drop_column(table, column)

    if table_exists('model_instances') and column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.drop_column('model_instances', 'model_config_snapshot')
