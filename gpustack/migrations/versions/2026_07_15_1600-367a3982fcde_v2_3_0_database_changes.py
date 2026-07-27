"""v2.3.0 database changes

Adds the ``scaling_schedule`` JSON column to ``models`` for scheduled scaling:
a per-model cron timetable that drives the model's replica count. The column
stores the serialized ``ScalingSchedule`` (enabled flag, ``baseline_replicas``,
and the list of ``start_cron`` + ``duration_seconds`` + ``replicas`` window
rules); NULL means no schedule is configured.

Revision ID: 367a3982fcde
Revises: b3e4a57bee80
Create Date: 2026-07-15 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '367a3982fcde'
down_revision: Union[str, None] = 'b3e4a57bee80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('scaling_schedule', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('scaling_schedule')
