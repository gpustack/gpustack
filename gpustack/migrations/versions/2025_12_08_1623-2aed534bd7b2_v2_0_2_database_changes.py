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
from gpustack.schemas.stmt import model_user_after_drop_view_stmt
from gpustack.schemas.common import SQLAlchemyJSON


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
        batch_op.add_column(sa.Column('proxy_mode', model_instance_proxy_mode, nullable=True))
        batch_op.add_column(sa.Column('advertise_address', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('worker_advertise_address', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.add_column(sa.Column('server_url', sa.String(length=2048), nullable=True))
        batch_op.add_column(sa.Column('worker_config',  SQLAlchemyJSON(), nullable=True))

    op.execute(model_user_after_drop_view_stmt)

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('run_command', type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)

    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.alter_column('default_run_command', type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)



def downgrade() -> None:
    model_instance_proxy_mode = sa.Enum(
        'WORKER',
        'DIRECT',
        'DELEGATED',
        name='modelinstanceproxymodeenum',
    )
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('proxy_mode')
        batch_op.drop_column('advertise_address')
    model_instance_proxy_mode.drop(op.get_bind(), checkfirst=True)

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('worker_advertise_address')

    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.drop_column('server_url')
        batch_op.drop_column('worker_config')

    op.execute(model_user_after_drop_view_stmt)

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('run_command', type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)

    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.alter_column('default_run_command', type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)
