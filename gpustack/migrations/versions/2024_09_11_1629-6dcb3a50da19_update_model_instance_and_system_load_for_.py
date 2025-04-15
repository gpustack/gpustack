"""update model, instance, and system load for distributed scheduling

Revision ID: 6dcb3a50da19
Revises: 8277680cfcb7
Create Date: 2024-09-11 16:29:50.615356

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '6dcb3a50da19'
down_revision: Union[str, None] = '8277680cfcb7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

placement_strategy_enum = sa.Enum('SPREAD', 'BINPACK', name='placementstrategyenum')

def upgrade() -> None:
    # system_loads
    with op.batch_alter_table('system_loads') as batch_op:
        batch_op.alter_column('memory', new_column_name='ram', existing_type=sa.Float)
        batch_op.alter_column('gpu_memory', new_column_name='vram', existing_type=sa.Float)

    # models
    bind = op.get_bind()
    if bind.dialect.name in ('postgresql', 'mysql'):
        placement_strategy_enum.create(bind, checkfirst=True)
    with op.batch_alter_table('models') as batch_op:
        batch_op.add_column(sa.Column('placement_strategy', placement_strategy_enum, nullable=False, server_default='SPREAD'))
        batch_op.add_column(sa.Column('cpu_offloading', sa.Boolean(), nullable=False, server_default="1"))
        batch_op.add_column(sa.Column(
            'distributed_inference_across_workers', sa.Boolean(), nullable=False, server_default="1"))
        batch_op.add_column(sa.Column('worker_selector', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('gpu_selector',
            gpustack.schemas.common.JSON(), nullable=True))

    # model_instances
    op.execute('Delete from model_instances')
    with op.batch_alter_table('model_instances') as batch_op:
        batch_op.add_column(sa.Column('distributed_servers',
            gpustack.schemas.common.JSON(), nullable=True))
        batch_op.add_column(sa.Column('gpu_indexes', sa.JSON(), nullable=True))
        batch_op.drop_column('gpu_index')


def downgrade() -> None:
    # system_loads
    with op.batch_alter_table('system_loads') as batch_op:
        batch_op.alter_column('ram', new_column_name='memory')
        batch_op.alter_column('vram', new_column_name='gpu_memory')

    # models
    with op.batch_alter_table('models') as batch_op:
        batch_op.drop_column('gpu_selector')
        batch_op.drop_column('worker_selector')
        batch_op.drop_column('distributed_inference_across_workers')
        batch_op.drop_column('cpu_offloading')
        batch_op.drop_column('placement_strategy')

    # model_instances
    op.execute('Delete from model_instances')
    with op.batch_alter_table('model_instances') as batch_op:
        batch_op.drop_column('distributed_servers')
        batch_op.drop_column('gpu_indexes')
        batch_op.add_column(sa.Column(
            'gpu_index', sa.INTEGER(), nullable=True))
