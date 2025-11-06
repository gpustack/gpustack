"""Server Proxy Improvement

Revision ID: 7737c7eba9c4
Revises: eca16ce6dedd
Create Date: 2025-10-23 17:36:07.871032

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '7737c7eba9c4'
down_revision: Union[str, None] = 'eca16ce6dedd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    with op.batch_alter_table("clusters", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('gateway_endpoint', sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column('reported_gateway_endpoint', sa.String(length=255), nullable=True)
        )

def downgrade() -> None:
    with op.batch_alter_table("clusters", schema=None) as batch_op:
        batch_op.drop_column('gateway_endpoint')
        batch_op.drop_column('reported_gateway_endpoint')
