"""Persist startup peers for static PD routers.

The snapshot records member identity, addresses and bootstrap ports so router
membership changes can be detected after server and worker restarts.

Revision ID: a7e2c9d4f681
Revises: f4a5b6c7d8e9
Create Date: 2026-10-13 16:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


revision: str = 'a7e2c9d4f681'
down_revision: Union[str, None] = 'f4a5b6c7d8e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if table_exists('model_instances') and not column_exists(
        'model_instances', 'router_peer_snapshot'
    ):
        op.add_column(
            'model_instances',
            sa.Column('router_peer_snapshot', sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if table_exists('model_instances') and column_exists(
        'model_instances', 'router_peer_snapshot'
    ):
        op.drop_column('model_instances', 'router_peer_snapshot')
