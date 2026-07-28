"""v2.3.0 database changes

Bundles the pre-release schema changes for v2.3.0:

1. GPU instance types: a ``gpu_instance_types`` table holding the per-cluster
   catalog of offerable types, plus ``gpu_instances.type_snapshot`` recording
   the type an instance was created from, so later edits to the catalog don't
   retroactively change existing instances.

2. ``models.scaling_schedule`` for scheduled scaling: a per-model cron
   timetable that drives the model's replica count. The column stores the
   serialized ``ScalingSchedule`` (enabled flag, ``baseline_replicas``, and the
   list of ``start_cron`` + ``duration_seconds`` + ``replicas`` window rules);
   NULL means no schedule is configured.

Revision ID: 367a3982fcde
Revises: c4d7e8f9a0b1
Create Date: 2026-07-15 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack

# revision identifiers, used by Alembic.
revision: str = '367a3982fcde'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'gpu_instance_types',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('spec', gpustack.schemas.common.JSON(), nullable=False),
        sa.Column('status', gpustack.schemas.common.JSON(), nullable=True),
        sa.Column('snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(['cluster_id'], ['clusters.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('snapshot', name='uq_gpu_instance_type_snapshot'),
    )

    with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'type_snapshot', sqlmodel.sql.sqltypes.AutoString(), nullable=True
            )
        )

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('scaling_schedule', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('scaling_schedule')

    with op.batch_alter_table('gpu_instances', schema=None) as batch_op:
        batch_op.drop_column('type_snapshot')

    op.drop_table('gpu_instance_types')
