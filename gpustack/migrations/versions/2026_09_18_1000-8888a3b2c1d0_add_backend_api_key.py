"""add backend_api_key

Revision ID: 8888a3b2c1d0
Revises: f1a2b3c4d5e6
Create Date: 2026-09-18 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.migrations.utils import column_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = '8888a3b2c1d0'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if table_exists('models'):
        if not column_exists('models', 'backend_api_key'):
            op.add_column('models', sa.Column('backend_api_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    if table_exists('inference_backends'):
        if not column_exists('inference_backends', 'api_key_parameter'):
            op.add_column('inference_backends', sa.Column('api_key_parameter', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    if table_exists('models'):
        if column_exists('models', 'backend_api_key'):
            op.drop_column('models', 'backend_api_key')

    if table_exists('inference_backends'):
        if column_exists('inference_backends', 'api_key_parameter'):
            op.drop_column('inference_backends', 'api_key_parameter')
