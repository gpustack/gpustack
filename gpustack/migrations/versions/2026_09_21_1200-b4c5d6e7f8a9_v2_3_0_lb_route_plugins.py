"""v2.3.0 LB route plugins schema

Storage for the LB route plugin framework:

1. ``model_route_targets.max_running_requests`` — the LB base
   capability's per-target knob (concurrency soft-cap; NULL = no
   filtering). LB stores on the route/target rows themselves.

2. ``model_route_capability_policies`` — the capability route plugins'
   shared storage: one row per (capability, route). The capability
   column distinguishes the plugin (session-affinity, least-load, the
   enterprise prefix-affinity), ``weight`` is the finisher's
   weighted-sum contribution as a first-class column, and ``config``
   carries the plugin-specific fields only. Lives in the central chain
   rather than startup ``create_all`` so every deployment gets it from
   the migration alone.

Revision ID: b4c5d6e7f8a9
Revises: f1a2b3c4d5e6
Create Date: 2026-09-21 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists
import gpustack.schemas.common


# revision identifiers, used by Alembic.
revision: str = 'b4c5d6e7f8a9'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not table_exists('model_route_capability_policies'):
        op.create_table(
            'model_route_capability_policies',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('route_id', sa.Integer(), nullable=False),
            sa.Column('capability', sa.String(length=64), nullable=False),
            sa.Column('weight', sa.Float(), nullable=True),
            sa.Column('config', sa.JSON(), nullable=False),
            sa.Column(
                'created_at', gpustack.schemas.common.UTCDateTime(), nullable=False
            ),
            sa.Column(
                'updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False
            ),
            sa.Column(
                'deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.ForeignKeyConstraint(
                ['route_id'], ['model_routes.id'], ondelete='CASCADE'
            ),
            sa.UniqueConstraint(
                'capability',
                'route_id',
                name='uix_capability_policy_capability_route',
            ),
        )
    if table_exists('model_route_targets'):
        if not column_exists('model_route_targets', 'max_running_requests'):
            with op.batch_alter_table(
                'model_route_targets', schema=None
            ) as batch_op:
                batch_op.add_column(
                    sa.Column('max_running_requests', sa.Integer(), nullable=True)
                )


def downgrade() -> None:
    if table_exists('model_route_capability_policies'):
        op.drop_table('model_route_capability_policies')
    if table_exists('model_route_targets'):
        if column_exists('model_route_targets', 'max_running_requests'):
            with op.batch_alter_table(
                'model_route_targets', schema=None
            ) as batch_op:
                batch_op.drop_column('max_running_requests')
