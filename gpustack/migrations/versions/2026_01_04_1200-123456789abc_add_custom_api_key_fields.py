"""add custom api key fields

Revision ID: 123456789abc
Revises: 2aed534bd7b2
Create Date: 2026-01-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '123456789abc'
down_revision: Union[str, None] = '2aed534bd7b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add is_custom column to api_keys table
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_custom', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('custom_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    # Remove custom API key columns
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_column('custom_key')
        batch_op.drop_column('is_custom')
