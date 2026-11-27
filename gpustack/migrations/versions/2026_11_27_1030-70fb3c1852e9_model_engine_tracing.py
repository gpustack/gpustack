"""Add a per-deployment engine tracing preference.

Revision ID: 70fb3c1852e9
Revises: f8c4a291bd65
"""

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists

revision = '70fb3c1852e9'
down_revision = 'f8c4a291bd65'
branch_labels = None
depends_on = None


def upgrade() -> None:
    if table_exists('models') and not column_exists('models', 'enable_engine_tracing'):
        op.add_column('models', sa.Column('enable_engine_tracing', sa.Boolean(), nullable=True))


def downgrade() -> None:
    if table_exists('models') and column_exists('models', 'enable_engine_tracing'):
        op.drop_column('models', 'enable_engine_tracing')
