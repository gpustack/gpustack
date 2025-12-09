"""v2.0.2 database changes

Revision ID: 2aed534bd7b2
Revises: e30134bd18dc
Create Date: 2025-12-08 16:23:31.709376

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '2aed534bd7b2'
down_revision: Union[str, None] = 'e30134bd18dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    model_instance_proxy_mode = sa.Enum(
        'WORKER',
        'DIRECT',
        'DELEGATED',
        name='modelinstanceproxymodeenum',
    )
    model_instance_proxy_mode.create(op.get_bind(), checkfirst=True)
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('proxy_model', model_instance_proxy_mode, nullable=True))


def downgrade() -> None:
    model_instance_proxy_mode = sa.Enum(
        'WORKER',
        'DIRECT',
        'DELEGATED',
        name='modelinstanceproxymodeenum',
    )
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('proxy_model')
    model_instance_proxy_mode.drop(op.get_bind(), checkfirst=True)
