"""update cpu_offloading to nullable

Revision ID: 89cb8df41bf0
Revises: 2ea2c247a117
Create Date: 2025-11-06 18:39:40.615588

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '89cb8df41bf0'
down_revision: Union[str, None] = '2ea2c247a117'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('cpu_offloading',
               existing_type=sa.BOOLEAN(),
               nullable=True,
               existing_server_default=sa.text('true'))

def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('cpu_offloading',
               existing_type=sa.BOOLEAN(),
               nullable=False,
               existing_server_default=sa.text('true'))
