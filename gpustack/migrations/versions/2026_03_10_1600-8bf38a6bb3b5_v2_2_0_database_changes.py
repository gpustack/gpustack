"""v2.2.0 add worker version

Revision ID: 8bf38a6bb3b5
Revises: 8ad0f94c92e8
Create Date: 2026-03-10 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import gpustack
import gpustack.utils.sql_enum as sql_enum

# revision identifiers, used by Alembic.
revision: str = '8bf38a6bb3b5'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

model_instance_proxy_mode = sa.Enum(
    'WORKER',
    'DIRECT',
    'DELEGATED',
    name='modelinstanceproxymodeenum',
)
proxy_mode_to_add = ['TUNNEL']


def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('worker_version', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('proxy_address', sa.String(255), nullable=True))
    
    sql_enum.add_enum_values(
        {'workers': 'proxy_mode'},
        model_instance_proxy_mode,
        *proxy_mode_to_add,
    )

    ### k8s volume mount
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.add_column(sa.Column('k8s_volume_mounts', gpustack.schemas.common.JSON(), nullable=True))
    ### end
    

def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('worker_version')
        batch_op.drop_column('proxy_address')

    sql_enum.remove_enum_values(
        {'workers': ('proxy_mode', 'WORKER')},
        model_instance_proxy_mode,
        *proxy_mode_to_add,
    )

    ### k8s volume mount
    with op.batch_alter_table('clusters', schema=None) as batch_op:
        batch_op.drop_column('k8s_volume_mounts')
    ### end
