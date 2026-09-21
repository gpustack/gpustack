"""Preserve the first failure reason on a model instance

Adds ``model_instances.first_failure_message`` and
``model_instances.first_failure_at``: the failure that started the current
unhealthy streak, kept across the automatic restarts that clear
``state_message``. Without them the recovery destroys the reason for the failure
it recovered from, so a crash loop leaves only the latest secondary error behind
(issue #6019).

Both nullable and additive, and nothing reads them unless the worker writes
them, so an older version pointed at the same database keeps working.

The subordinate-worker equivalents need no schema change: they live inside the
existing ``model_instances.distributed_servers`` JSON column.

Revision ID: 9f4c1ab7d203
Revises: 367a3982fcde
Create Date: 2026-08-11 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime
from gpustack.migrations.utils import column_exists

# revision identifiers, used by Alembic.
revision: str = '9f4c1ab7d203'
down_revision: Union[str, None] = '367a3982fcde'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Guarded per column, matching the v2.3.0 bundle: a development database
    that already carries one of these can migrate forward instead of rebuild."""
    # The guard sits outside the batch context on purpose: SQLite batch mode
    # rebuilds the table from the reflected definition, so reflecting from
    # inside a live batch block is not the same question as asking first.
    if not column_exists('model_instances', 'first_failure_message'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('first_failure_message', sa.Text(), nullable=True)
            )

    if not column_exists('model_instances', 'first_failure_at'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('first_failure_at', UTCDateTime(), nullable=True)
            )


def downgrade() -> None:
    if column_exists('model_instances', 'first_failure_at'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.drop_column('first_failure_at')

    if column_exists('model_instances', 'first_failure_message'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.drop_column('first_failure_message')
