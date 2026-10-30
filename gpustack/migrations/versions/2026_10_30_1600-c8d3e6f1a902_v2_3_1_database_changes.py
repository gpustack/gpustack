"""v2.3.1 database changes.

Persist model configuration used to launch each instance.

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


def upgrade() -> None:
    if table_exists('model_instances') and not column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.add_column(
            'model_instances',
            sa.Column('model_config_snapshot', sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if table_exists('model_instances') and column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.drop_column('model_instances', 'model_config_snapshot')
