"""Persist KV transfer interfaces for workers.

This standalone revision ships in v2.3.x. Later release bundles must retain
this revision and its parent to support upgrades from v2.3.x databases.

Revision ID: 921586f4a09b
Revises: c8d3e6f1a902
Create Date: 2026-11-20 02:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists

revision: str = '921586f4a09b'
down_revision: Union[str, None] = 'c8d3e6f1a902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if table_exists('workers') and not column_exists('workers', 'kv_transfer_ifname'):
        op.add_column('workers', sa.Column('kv_transfer_ifname', sa.Text(), nullable=True))


def downgrade() -> None:
    if table_exists('workers') and column_exists('workers', 'kv_transfer_ifname'):
        op.drop_column('workers', 'kv_transfer_ifname')
