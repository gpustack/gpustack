"""v2.3.1 database changes.

Persist model configuration used to launch each instance and site branding.

Revision ID: c8d3e6f1a902
Revises: a7e2c9d4f681
Create Date: 2026-10-30 16:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from gpustack.schemas.common import UTCDateTime

from gpustack.migrations.utils import column_exists, table_exists


revision: str = 'c8d3e6f1a902'
down_revision: Union[str, None] = 'a7e2c9d4f681'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not table_exists('site_settings'):
        op.create_table(
            'site_settings',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=False),
            sa.Column('site_name', sa.String(128), nullable=False),
            sa.Column('primary_color', sa.String(7), nullable=False),
            sa.Column('logo', sa.Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=True),
            sa.Column('logo_dark', sa.Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=True),
            sa.Column('mini_logo', sa.Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=True),
            sa.Column('mini_logo_dark', sa.Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=True),
            sa.Column('favicon', sa.Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=True),
            sa.Column('footer_name', sa.String(128), nullable=True),
            sa.Column('footer_url', sa.String(2048), nullable=True),
            sa.Column('updated_at', UTCDateTime(), nullable=True),
            sa.CheckConstraint('id = 1', name='ck_site_settings_singleton'),
        )
    if table_exists('model_instances') and not column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.add_column(
            'model_instances',
            sa.Column('model_config_snapshot', sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if table_exists('site_settings'):
        op.drop_table('site_settings')
    if table_exists('model_instances') and column_exists(
        'model_instances', 'model_config_snapshot'
    ):
        op.drop_column('model_instances', 'model_config_snapshot')
