"""add backend_source and readme to inference_backends table

Revision ID: 1a2b3c4d5e6f
Revises: 53667f33f000
Create Date: 2026-01-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = '1a2b3c4d5e6f'
down_revision: Union[str, None] = '53667f33f000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.add_column(sa.Column('backend_source', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('enabled', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('icon', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('default_environment', sa.JSON(), nullable=True))
        # Change description column from String(255) to Text to support longer descriptions
        batch_op.alter_column('description',
                              existing_type=sa.String(length=255),
                              type_=sa.Text(),
                              existing_nullable=True)


def downgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.drop_column('backend_source')
        batch_op.drop_column('enabled')
        batch_op.drop_column('icon')
        batch_op.drop_column('default_environment')
        # Revert description column back to String(255)
        batch_op.alter_column('description',
                              existing_type=sa.Text(),
                              type_=sa.String(length=255),
                              existing_nullable=True)
