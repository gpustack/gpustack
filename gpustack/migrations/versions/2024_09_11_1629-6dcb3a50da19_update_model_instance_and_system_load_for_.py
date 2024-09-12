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


def upgrade() -> None:
    # system_loads
    op.alter_column('system_loads', 'memory', new_column_name='ram')
    op.alter_column('system_loads', 'gpu_memory', new_column_name='vram')

    # models
    op.add_column('models', sa.Column('placement_strategy', sa.Enum(
        'SPREAD', 'BINPACK', name='placementstrategyenum'), nullable=True))
    op.execute('UPDATE models SET placement_strategy = "SPREAD"')
    with op.batch_alter_table('models') as batch_op:
        batch_op.alter_column('placement_strategy', nullable=False)

    op.add_column('models', sa.Column('cpu_offloading', sa.Boolean(), nullable=True))
    op.execute('UPDATE models SET cpu_offloading = True')
    with op.batch_alter_table('models') as batch_op:
        batch_op.alter_column('cpu_offloading', nullable=False)

    op.add_column('models', sa.Column(
        'distributed_inference_across_workers', sa.Boolean(), nullable=True))
    op.execute('UPDATE models SET distributed_inference_across_workers = False')
    with op.batch_alter_table('models') as batch_op:
        batch_op.alter_column('distributed_inference_across_workers', nullable=False)

    op.add_column('models', sa.Column('worker_selector', sa.JSON(), nullable=True))
    op.add_column('models', sa.Column('gpu_selector',
                  gpustack.schemas.common.JSON(), nullable=True))

    # model_instances
    op.add_column('model_instances', sa.Column('distributed_servers',
                  gpustack.schemas.common.JSON(), nullable=True))

    op.add_column('model_instances', sa.Column('gpu_indexes', sa.JSON(), nullable=True))
    conn = op.get_bind()
    model_instances = conn.execute(
        sa.text("SELECT id, gpu_index, computed_resource_claim FROM model_instances")).fetchall()

    for instance in model_instances:
        gpu_array = []
        if instance.gpu_index is not None:
            gpu_array = [instance.gpu_index]

        claim = json.loads(instance.computed_resource_claim)
        if "gpu_memory" in claim and isinstance(claim["gpu_memory"], int):
            claim["vram"] = {0: claim["gpu_memory"]}

        if "memory" in claim and isinstance(claim["memory"], int):
            claim["ram"] = claim["memory"]

        conn.execute(
            sa.text("UPDATE model_instances SET gpu_indexes = :gpu_indexes, computed_resource_claim = :computed_resource_claim  WHERE id = :id"),
            {"gpu_indexes": json.dumps(gpu_array), "computed_resource_claim": json.dumps(
                claim), "id": instance.id}
        )
    op.drop_column('model_instances', 'gpu_index')


def downgrade() -> None:
    # system_loads
    op.alter_column('system_loads', 'ram', new_column_name='memory')
    op.alter_column('system_loads', 'vram', new_column_name='gpu_memory')

    # models
    op.drop_column('models', 'gpu_selector')
    op.drop_column('models', 'worker_selector')
    op.drop_column('models', 'distributed_inference_across_workers')
    op.drop_column('models', 'cpu_offloading')
    op.drop_column('models', 'placement_strategy')

    # model_instances
    op.execute('Delete from model_instances')
    op.drop_column('model_instances', 'distributed_servers')
    op.drop_column('model_instances', 'gpu_indexes')
    op.add_column('model_instances', sa.Column(
        'gpu_index', sa.INTEGER(), nullable=True))
