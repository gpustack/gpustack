"""update partial offload to cpu offload

Revision ID: dc3de970595b
Revises: 599e39158abe
Create Date: 2024-09-10 10:32:59.985761

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'dc3de970595b'
down_revision: Union[str, None] = '599e39158abe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('models', 'partial_offload', new_column_name='cpu_offloading')


def downgrade() -> None:
    op.alter_column('models', 'cpu_offloading', new_column_name='partial_offload')
