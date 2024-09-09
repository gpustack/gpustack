"""update memory and gpu_memory to ram and vram

Revision ID: 599e39158abe
Revises: de42569c03ba
Create Date: 2024-09-09 16:25:29.948918

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '599e39158abe'
down_revision: Union[str, None] = 'de42569c03ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('system_loads', 'memory', new_column_name='ram')
    op.alter_column('system_loads', 'gpu_memory', new_column_name='vram')


def downgrade() -> None:
    op.alter_column('system_loads', 'ram', new_column_name='memory')
    op.alter_column('system_loads', 'vram', new_column_name='gpu_memory')
