"""add worker ifname

Revision ID: 35e8ebcae667
Revises: nrxtab43e8j8
Create Date: 2025-10-22 17:49:55.394206

"""
from typing import Sequence, Union

import sqlmodel
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '35e8ebcae667'
down_revision: Union[str, None] = 'nrxtab43e8j8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('worker_ifname', sqlmodel.sql.sqltypes.AutoString(), nullable=True, server_default=""))
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ifname', sqlmodel.sql.sqltypes.AutoString(), nullable=True, server_default=""))


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('worker_ifname')
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('ifname')
