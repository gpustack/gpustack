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
from gpustack.migrations.utils import table_exists
import gpustack.utils.sql_enum as sql_enum
from gpustack.schemas.stmt import model_user_after_drop_view_stmt

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

access_policy_enum = sa.Enum(
    'PUBLIC',
    'AUTHED',
    'ALLOWED_USERS',
    name='accesspolicyenum'
).with_variant(
    postgresql.ENUM(
        'PUBLIC',
        'AUTHED',
        'ALLOWED_USERS',
        name='accesspolicyenum',
        create_type=False,
    ), "postgresql"
)

UPGRADE_GPU_TYPE_MAPPING = {
    "npu": "cann",
    "dcu": "dtk",
    "mlu": "neuware",
}

DOWNGRADE_GPU_TYPE_MAPPING = {v: k for k, v in UPGRADE_GPU_TYPE_MAPPING.items()}

sqlite_now = "datetime('now')"

def now_func():
    return sa.func.now() if op.get_bind().dialect.name != 'sqlite' else sa.text(sqlite_now)


def upgrade() -> None:
    public_maas_integration_upgrade()
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
    public_maas_integration_downgrade()
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
            
def public_maas_integration_upgrade():
    # Create model_providers table
    if not table_exists('model_providers'):
        op.create_table(
            'model_providers',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('name', sqlmodel.sql.sqltypes.AutoString() , nullable=False, unique=True, index=True),
            sa.Column('description', sqlmodel.sql.sqltypes.AutoString() , nullable=True),
            sa.Column('api_tokens', sa.JSON(), nullable=False, server_default='[]'),
            sa.Column('timeout', sa.Integer(), nullable=False, default=120),
            sa.Column('config', sa.JSON(), nullable=False),
            sa.Column('models', sa.JSON(), nullable=True),
            sa.Column('proxy_url', sqlmodel.sql.sqltypes.AutoString() , nullable=True),
            sa.Column('proxy_timeout', sa.Integer(), nullable=True),
            sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
            sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
            sa.Column('deleted_at', sa.DateTime(), nullable=True),
        )

    # Create targetstateenum type
    target_state_enum = sa.Enum(
        'ACTIVE', 'UNAVAILABLE', name='targetstateenum',
    )

    if not table_exists('model_routes'):
        # Create model_routes table
        op.create_table(
            'model_routes',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('name', sqlmodel.sql.sqltypes.AutoString() , nullable=False),
            sa.Column('description', sqlmodel.sql.sqltypes.AutoString() , nullable=True),
            sa.Column('categories', sa.JSON(), nullable=False, default='[]'),
            sa.Column('meta', sa.JSON(), nullable=True, default='{}'),
            sa.Column('created_by_model', sa.Boolean(), nullable=False, default=False),
            sa.Column('targets', sa.Integer(), nullable=False, default=0),
            sa.Column('ready_targets', sa.Integer(), nullable=False, default=0),
            sa.Column('access_policy', access_policy_enum, nullable=False, server_default='AUTHED'),
            sa.Column('generic_proxy', sa.Boolean(), nullable=True, server_default=sa.sql.expression.false()),
            sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
            sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
            sa.Column('deleted_at', sa.DateTime(), nullable=True),

        )
        op.create_index(op.f('ix_model_routes_name'), 'model_routes', ['name'], unique=True)

    if not table_exists('model_route_targets'):
        # Create model_route_targets table
        op.create_table(
            'model_route_targets',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('name', sqlmodel.sql.sqltypes.AutoString() , nullable=False),
            sa.Column('route_id', sa.Integer(), sa.ForeignKey('model_routes.id', ondelete='CASCADE'), nullable=False),
            sa.Column('route_name', sqlmodel.sql.sqltypes.AutoString() , nullable=False),
            sa.Column('provider_id', sa.Integer(), sa.ForeignKey('model_providers.id', ondelete='CASCADE'), nullable=True),
            sa.Column('provider_model_name', sqlmodel.sql.sqltypes.AutoString() , nullable=True),
            sa.Column('model_id', sa.Integer(), sa.ForeignKey('models.id', ondelete='CASCADE'), nullable=True),
            sa.Column('weight', sa.Integer(), nullable=False, default=100),
            sa.Column('fallback_status_codes', sa.JSON(), nullable=True),
            sa.Column('state', target_state_enum, nullable=False, server_default='ACTIVE'),
            sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
            sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
            sa.Column('deleted_at', sa.DateTime(), nullable=True),
        )
        # create index for route_name
        op.create_index(op.f('ix_model_route_targets_route_name'), 'model_route_targets', ['route_name'], unique=False)

    if not table_exists('usermodelroutelink'):
        op.create_table('usermodelroutelink',
            sa.Column('route_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(['route_id'], ['model_routes.id'], name='fk_route_user_link_routes', ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_route_user_link_users', ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('route_id', 'user_id')
        )

    # migrate create model route data from models
    conn = op.get_bind()
    conn.execute(sa.text("""
    insert into model_routes (name, description, categories, meta, created_by_model, targets, ready_targets, access_policy, generic_proxy)
    select name, description, categories, meta, true, 1, 0, access_policy, generic_proxy from models;
    """))
    # make target state default to UNAVAILABLE and let the controller update to ACTIVE after deployment is ready
    conn.execute(sa.text("""
    insert into model_route_targets (name, route_id, model_id, route_name, weight, state)
        select concat(m.name, '-deployment') as name,
            ma.id as route_id,
            m.id as model_id,
            m.name as route_name,
            100 as weight,
            'UNAVAILABLE' as state
        from models m
        join model_routes ma on ma.name = m.name;
    """))

    # migrate modeluserlink to usermodelroutelink
    conn.execute(sa.text("""
    insert into usermodelroutelink(route_id, user_id)
        select mae.route_id, user_id from
            modeluserlink as link
            join model_route_targets mae 
            on link.model_id = mae.model_id;
    """))

    # drop non_admin_user_models view if exists
    op.execute(model_user_after_drop_view_stmt)

    with op.batch_alter_table('model_usages', schema=None) as batch_op:
        batch_op.alter_column('model_id', existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column('provider_id', sa.Integer(), nullable=True))
    
    op.create_foreign_key(
        'fk_model_usages_provider_id_model_providers',
        'model_usages',
        'model_providers',
        ['provider_id'],
        ['id'],
        ondelete='SET NULL'
    )
    
    conn.execute(sa.text("""
    update model_usages mu set
        model_name = m.name,
        model_id = m.id
    from models m where mu.model_id = m.id;
    """))

    with op.batch_alter_table('model_usages', schema=None) as batch_op:
        batch_op.alter_column('model_name', existing_type=sqlmodel.sql.sqltypes.AutoString(), nullable=False)

def public_maas_integration_downgrade():
    conn = op.get_bind()
    
    op.drop_constraint('fk_model_usages_provider_id_model_providers', 'model_usages', type_='foreignkey', if_exists=True)
    # remove rows which model_id is null in model_usages
    conn.execute(sa.text("""
    delete from model_usages where model_id is null;
    """)) 

    with op.batch_alter_table('model_usages', schema=None) as batch_op:
        batch_op.drop_column('model_name')
        batch_op.drop_column('provider_id')
        batch_op.alter_column('model_id', existing_type=sa.Integer(), nullable=False)
    # drop non_admin_user_models view if exists
    op.execute(model_user_after_drop_view_stmt)

    op.drop_table('usermodelroutelink', if_exists=True)
    # Drop model_route_targets table
    op.drop_index(op.f('ix_model_route_targets_route_name'), table_name='model_route_targets', if_exists=True)
    op.drop_table('model_route_targets', if_exists=True)
    # Drop model_routes table
    op.drop_index(op.f('ix_model_routes_name'), table_name='model_routes', if_exists=True)
    op.drop_table('model_routes', if_exists=True)
    try:
        # Drop targetstateenum type
        sa.Enum(name='targetstateenum').drop(op.get_bind(), checkfirst=True)
    except Exception:
        pass
    # Drop model_providers table
    op.drop_table('model_providers', if_exists=True)
