"""add worker version

Revision ID: 8bf38a6bb3b5
Revises: 53667f33f000
Create Date: 2026-03-10 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8bf38a6bb3b5'
down_revision: Union[str, None] = '53667f33f000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('worker_version', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('worker_git_commit', sa.String(40), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('worker_git_commit')
        batch_op.drop_column('worker_version')
