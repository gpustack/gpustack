"""v2.1.0 database migration

Revision ID: 6c59ed2c8603
Revises: 2aed534bd7b2
Create Date: 2026-01-04 14:19:49.920230

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack.utils.sql_enum as sql_enum


# revision identifiers, used by Alembic.
revision: str = '6c59ed2c8603'
down_revision: Union[str, None] = '2aed534bd7b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

cluster_state_enum = sa.Enum(
    'PROVISIONING',
    'PROVISIONED',
    'READY',
    name='clusterstateenum',
)

cluster_state_to_add = ['PENDING']


def upgrade() -> None:
    sql_enum.add_enum_values(
        {'clusters': 'state'},
        cluster_state_enum,
        *cluster_state_to_add,
    )


def downgrade() -> None:
    sql_enum.remove_enum_values(
        {'clusters': ('state', 'PROVISIONING')},
        cluster_state_enum,
        *cluster_state_to_add,
    )
