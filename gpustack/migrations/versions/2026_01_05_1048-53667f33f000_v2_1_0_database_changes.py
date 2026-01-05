"""v2.1.0 database changes

Revision ID: 53667f33f000
Revises: 2aed534bd7b2
Create Date: 2026-01-05 10:48:18.831340

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql
import gpustack.utils.sql_enum as sql_enum

# revision identifiers, used by Alembic.
revision: str = '53667f33f000'
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
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('api_detected_backend_version', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('gpu_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        
    sql_enum.add_enum_values(
        {'clusters': 'state'},
        cluster_state_enum,
        *cluster_state_to_add,
    )


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('api_detected_backend_version')
        batch_op.drop_column('gpu_type')

    sql_enum.remove_enum_values(
        {'clusters': ('state', 'PROVISIONING')},
        cluster_state_enum,
        *cluster_state_to_add,
    )
