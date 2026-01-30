"""v2.1.0 database changes

Revision ID: 53667f33f000
Revises: 2aed534bd7b2
Create Date: 2026-01-05 10:48:18.831340

"""
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import json
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

UPGRADE_GPU_TYPE_MAPPING = {
    "npu": "cann",
    "dcu": "dtk",
    "mlu": "neuware",
}

DOWNGRADE_GPU_TYPE_MAPPING = {v: k for k, v in UPGRADE_GPU_TYPE_MAPPING.items()}

def upgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('api_detected_backend_version', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('gpu_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        
    sql_enum.add_enum_values(
        {'clusters': 'state'},
        cluster_state_enum,
        *cluster_state_to_add,
    )
       
    _migrate_model_gpu_selector(UPGRADE_GPU_TYPE_MAPPING)
    
    # Create benchmarks table
    op.create_table('benchmarks',
    sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
    sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
    sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
    sa.Column('raw_metrics', sa.JSON(), nullable=True),
    sa.Column('requests_per_second_mean', sa.Float(), nullable=True),
    sa.Column('request_latency_mean', sa.Float(), nullable=True),
    sa.Column('time_per_output_token_mean', sa.Float(), nullable=True),
    sa.Column('inter_token_latency_mean', sa.Float(), nullable=True),
    sa.Column('time_to_first_token_mean', sa.Float(), nullable=True),
    sa.Column('tokens_per_second_mean', sa.Float(), nullable=True),
    sa.Column('output_tokens_per_second_mean', sa.Float(), nullable=True),
    sa.Column('input_tokens_per_second_mean', sa.Float(), nullable=True),
    sa.Column('snapshot', gpustack.schemas.common.JSON(), nullable=True),
    sa.Column('gpu_summary', sa.Text(), nullable=True),
    sa.Column('gpu_vendor_summary', sa.Text(), nullable=True),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('profile', sqlmodel.sql.sqltypes.AutoString(), nullable=True, server_default="Custom"),
    sa.Column('dataset_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('dataset_input_tokens', sa.Integer(), nullable=True),
    sa.Column('dataset_output_tokens', sa.Integer(), nullable=True),
    sa.Column('dataset_seed', sa.Integer(), nullable=True, server_default="42"),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('cluster_id', sa.Integer(), nullable=False),
    sa.Column('model_id', sa.Integer(), nullable=True),
    sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('model_instance_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('request_rate', sa.Integer(), nullable=False),
    sa.Column('total_requests', sa.Integer(), nullable=True),
    sa.Column('state', sa.Enum('PENDING', 'RUNNING', 'QUEUED', 'STOPPED', 'ERROR', 'UNREACHABLE', 'COMPLETED', name='benchmarkstateenum'), nullable=False),
    sa.Column('state_message', sa.Text(), nullable=True),
    sa.Column('progress', sa.Float(), nullable=True),
    sa.Column('worker_id', sa.Integer(), nullable=True),
    sa.Column('pid', sa.Integer(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('benchmarks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_benchmarks_name'), ['name'], unique=True)


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('api_detected_backend_version')
        batch_op.drop_column('gpu_type')

    sql_enum.remove_enum_values(
        {'clusters': ('state', 'PROVISIONING')},
        cluster_state_enum,
        *cluster_state_to_add,
    )
    
    _migrate_model_gpu_selector(DOWNGRADE_GPU_TYPE_MAPPING)
        
    with op.batch_alter_table('benchmarks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_benchmarks_name'))
    op.drop_table('benchmarks')

def _migrate_model_gpu_selector(gpu_type_map: dict[str, str]) -> None:
    conn = op.get_bind()

    select_stmt = sa.text("SELECT id, gpu_selector FROM models WHERE gpu_selector IS NOT NULL")
    model_instances = conn.execute(select_stmt).fetchall()

    for row in model_instances:
        instance_id = row[0]
        gpu_selector = row[1]
        if not gpu_selector:
            continue
        
        if isinstance(gpu_selector, str):
            try:
                gpu_selector = json.loads(gpu_selector)
            except json.JSONDecodeError:
                continue

        if gpu_selector is None:
            continue
        
        gpu_ids = gpu_selector.get("gpu_ids")
        if not isinstance(gpu_ids, list):
            continue
        
        new_gpu_ids = []
        changed = False
        
        for gpu_id in gpu_ids:
            if not isinstance(gpu_id, str):
                new_gpu_ids.append(gpu_id)
                continue
            
            parts = gpu_id.split(":")
            if len(parts) != 3:
                new_gpu_ids.append(gpu_id)
                continue
            
            worker, gpu_type, index = parts
            new_gpu_type = gpu_type_map.get(gpu_type, gpu_type)
            if new_gpu_type != gpu_type:
                changed = True
                
            new_gpu_ids.append(f"{worker}:{new_gpu_type}:{index}")

        if not changed:
            continue
        
        gpu_selector["gpu_ids"] = new_gpu_ids
        conn.execute(
            sa.text(
                "UPDATE models SET gpu_selector = :gpu_selector WHERE id = :id"
            ),
            {"gpu_selector": json.dumps(gpu_selector), "id": instance_id}
        )    
            
