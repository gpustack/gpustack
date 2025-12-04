"""v2.0 database migration

Revision ID: 924c9a0b4c13
Revises: d19176de3b74
Create Date: 2025-09-08 14:54:02.843848

"""
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
import secrets
from gpustack.schemas.stmt import (
    worker_after_drop_view_stmt_sqlite,
    worker_after_drop_view_stmt_mysql,
    worker_after_drop_view_stmt_postgres,
)


# revision identifiers, used by Alembic.
revision: str = '924c9a0b4c13'
down_revision: Union[str, None] = 'd19176de3b74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



sqlite_now = "datetime('now')"

def now_func():
    return sa.func.now() if op.get_bind().dialect.name != 'sqlite' else sa.text(sqlite_now)

def utc_now_sql_literal():
    """
    Return the current UTC time as a naive datetime SQL literal string,
    e.g. '2025-11-17 12:33:15.123456'
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return f"'{now.isoformat(sep=' ')}'"

WORKER_STATE_ADDITIONAL_VALUES = ['PENDING', 'PROVISIONING', 'INITIALIZING', 'DELETING', "ERROR"]

def upgrade() -> None:
    cluster_state_enum = sa.Enum(
        'PROVISIONING',
        'PROVISIONED',
        'READY',
        name='clusterstateenum',
    )
    cluster_provider_enum = sa.Enum(
        'Docker',
        'Kubernetes',
        'DigitalOcean',
        name='clusterprovider',
    )
    op.create_table(
        'cloud_credentials',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('provider', cluster_provider_enum, nullable=False, default='DigitalOcean'),
        sa.Column('key', sa.String(length=255), nullable=True),
        sa.Column('secret', sa.String(length=255), nullable=True),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('idx_cloud_credentials_deleted_at_created_at', 'cloud_credentials', ['deleted_at', 'created_at'])

    op.create_table(
        'clusters',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('provider', cluster_provider_enum, nullable=False, default='Docker'),
        sa.Column('credential_id', sa.Integer(), sa.ForeignKey('cloud_credentials.id'), nullable=True),
        sa.Column('region', sa.String(length=255), nullable=True),
        sa.Column('state', cluster_state_enum, nullable=False, default='PROVISIONING'),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('hashed_suffix', sa.String(length=12), nullable=False),
        sa.Column('registration_token', sa.String(length=58), nullable=False),
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('idx_clusters_deleted_at_created_at', 'clusters', ['deleted_at', 'created_at'])

    row = op.get_bind().execute(sa.text("select count(*) from workers;")).fetchone()
    worker_count = row[0] if row is not None else 0
    should_create_cluster = worker_count > 0
    if should_create_cluster:
        # create default cluster when embedded worker is enabled
        # state == 3 means PROVISIONED and READY
        op.execute(f"""
            INSERT INTO clusters (name, description, provider, state, hashed_suffix, registration_token, created_at, updated_at)
            VALUES ('Default Cluster', 'The default cluster for GPUStack', 'Docker', 'READY', '{secrets.token_hex(6)}', '', {utc_now_sql_literal()}, {utc_now_sql_literal()})
        """)

    op.create_table(
        'worker_pools',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('cluster_id', sa.Integer(), sa.ForeignKey('clusters.id', ondelete='CASCADE'), nullable=False),
        sa.Column('instance_type', sa.String(length=255), nullable=True),
        sa.Column('instance_spec',sa.JSON(), nullable=True),
        sa.Column('os_image', sa.String(length=255), nullable=True),
        sa.Column('image_name', sa.String(length=255), nullable=True),
        sa.Column('zone', sa.String(length=255), nullable=True),
        sa.Column('labels', sa.JSON(), nullable=True, default=dict),
        sa.Column('cloud_options', sa.JSON(), nullable=True, default=dict),
        sa.Column('batch_size', sa.Integer(), nullable=True),
        sa.Column('replicas', sa.Integer(), nullable=True, default=1),
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('idx_worker_pools_deleted_at_created_at', 'worker_pools', ['deleted_at', 'created_at'])

    user_role_enum = sa.Enum('Cluster', 'Worker', name='userrole')
    user_role_enum.create(op.get_bind(), checkfirst=True)
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('worker_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('is_system', sa.Boolean(), nullable=True, default=False))
        batch_op.add_column(sa.Column('role', user_role_enum, nullable=True))
        batch_op.create_foreign_key('fk_users_cluster_id', 'clusters', ['cluster_id'], ['id'], ondelete='CASCADE')
        batch_op.create_foreign_key('fk_users_worker_id', 'workers', ['worker_id'], ['id'], ondelete='CASCADE')

    op.execute("""
        UPDATE users SET is_system = false WHERE is_system IS NULL
    """)
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('is_system', existing_type=sa.Boolean(), nullable=False)

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('provider', cluster_provider_enum, nullable=True))
        batch_op.add_column(sa.Column('worker_pool_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('machine_id', sa.String(length=255), nullable=True))
        batch_op.create_foreign_key('fk_workers_cluster_id', 'clusters', ['cluster_id'], ['id'], ondelete='CASCADE')
        batch_op.create_foreign_key('fk_workers_worker_pool_id', 'worker_pools', ['worker_pool_id'], ['id'], ondelete='RESTRICT')
        batch_op.add_column(sa.Column('external_id', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('ssh_key_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('provider_config', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('token', sa.String(length=58), nullable=True))

    op.execute("""
        UPDATE workers
        SET cluster_id = (SELECT max(id) FROM clusters), 
            provider = 'Docker', 
            worker_pool_id = NULL, 
            machine_id = NULL
        WHERE cluster_id IS NULL
    """)

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.alter_column('cluster_id', existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column('provider', existing_type=cluster_provider_enum, nullable=False)

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_models_cluster_id', 'clusters', ['cluster_id'], ['id'], ondelete='CASCADE')

    op.execute("""
        UPDATE models
        SET cluster_id = (SELECT max(id) FROM clusters)
        WHERE cluster_id IS NULL
    """)
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column('cluster_id', existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_model_instances_cluster_id', 'clusters', ['cluster_id'], ['id'], ondelete='CASCADE')

    op.execute("""
        UPDATE model_instances
        SET cluster_id = (SELECT max(id) FROM clusters)
        WHERE cluster_id IS NULL
    """)
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.alter_column('cluster_id', existing_type=sa.Integer(), nullable=False)


    if should_create_cluster:
        # create default cluster system user legacy worker user when embedded worker is enabled
        op.execute(f"""
            INSERT INTO users (username, is_admin, require_password_change, hashed_password, is_system, role, cluster_id, created_at, updated_at)
            VALUES
                ('system/cluster-1', false, false, '', true, 'Cluster', (SELECT max(id) FROM clusters), {utc_now_sql_literal()}, {utc_now_sql_literal()})
        """)
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_constraint('uix_name_user_id', type_='unique')
        batch_op.create_unique_constraint('uix_user_id_name', ['user_id', 'name'])

    with op.batch_alter_table('system_loads', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cluster_id', sa.Integer(), nullable=True))
    op.create_index('idx_system_loads_cluster_id', 'system_loads', ['cluster_id'])

    credential_type_enum = sa.Enum(
        'SSH',
        'CA',
        'X509',
        name='credentialtype'
    )

    op.create_table('credentials',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('external_id', sa.String(length=255), nullable=True, default=None),
        sa.Column('credential_type', credential_type_enum, nullable=False, default='ssh'),
        sa.Column('public_key', sa.Text(), nullable=False),
        sa.Column('encoded_private_key', sa.Text(), nullable=False),
        sa.Column('ssh_key_options', sa.JSON(), nullable=True, default=None),
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func()),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False, server_default=now_func(), onupdate=now_func()),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )
    op.create_index('idx_credentials_external_id', 'credentials', ['external_id'])


    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        # worker_state_enum
        for value in WORKER_STATE_ADDITIONAL_VALUES:
            conn.execute(
                sa.text(f"ALTER TYPE workerstateenum ADD VALUE '{value}'"))
    elif conn.dialect.name == 'mysql':
        # Get existing workerstateenum values
        result = conn.execute(
            sa.text("""
                SELECT COLUMN_TYPE
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = 'workers'
                AND COLUMN_NAME = 'state'
                AND TABLE_SCHEMA = DATABASE()
            """)
        ).scalar()

        existing_values = []
        if result:
            enum_str = result.split("enum(")[1].split(")")[0]
            existing_values = [v.strip("'") for v in enum_str.split("','")]

        new_values = existing_values.copy()
        for value in WORKER_STATE_ADDITIONAL_VALUES:
            if value not in new_values:
                new_values.append(value)
        if len(new_values) >= len(existing_values):
            new_enum_str = "enum('" + "','".join(new_values) + "')"

            # Construct new ALTER TABLE statement
            alter_sql = f"""
                ALTER TABLE workers 
                MODIFY COLUMN state {new_enum_str};
            """

            # Execute modification
            conn.execute(sa.text(alter_sql))

def downgrade() -> None:
    drop_view_stmt = worker_after_drop_view_stmt_sqlite
    if op.get_bind().dialect.name == 'mysql':
        drop_view_stmt = worker_after_drop_view_stmt_mysql
    elif op.get_bind().dialect.name == 'postgresql':
        drop_view_stmt = worker_after_drop_view_stmt_postgres
    op.drop_index('idx_system_loads_cluster_id', table_name='system_loads')
    with op.batch_alter_table('system_loads', schema=None) as batch_op:
        batch_op.drop_column('cluster_id')
    op.execute(drop_view_stmt)
    op.drop_index('idx_credentials_external_id', table_name='credentials')
    op.drop_index('idx_clusters_deleted_at_created_at', table_name='clusters')
    op.drop_index('idx_cloud_credentials_deleted_at_created_at', table_name='cloud_credentials')
    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.drop_constraint('uix_user_id_name', type_='unique')
        batch_op.create_unique_constraint('uix_name_user_id', ['name', 'user_id'])

    op.execute("""
        DELETE FROM users WHERE is_system = true;
    """)

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_constraint('fk_model_instances_cluster_id', type_='foreignkey')
        batch_op.drop_column('cluster_id')
    
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_constraint('fk_models_cluster_id', type_='foreignkey')
        batch_op.drop_column('cluster_id')

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_constraint('fk_workers_cluster_id', type_='foreignkey')
        batch_op.drop_constraint('fk_workers_worker_pool_id', type_='foreignkey')
        batch_op.drop_column('cluster_id')
        batch_op.drop_column('provider')
        batch_op.drop_column('worker_pool_id')
        batch_op.drop_column('machine_id')
        batch_op.drop_column('provider_config')
        batch_op.drop_column('ssh_key_id')
        batch_op.drop_column('external_id')
        batch_op.drop_column('token')

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_constraint('fk_users_cluster_id', type_='foreignkey')
        batch_op.drop_constraint('fk_users_worker_id', type_='foreignkey')
        batch_op.drop_column('cluster_id')
        batch_op.drop_column('worker_id')
        batch_op.drop_column('is_system')
        batch_op.drop_column('role')
    user_role_enum = sa.Enum('cluster', 'worker', name='userrole')
    user_role_enum.drop(op.get_bind(), checkfirst=True)

    op.drop_table('credentials')
    op.drop_table('worker_pools')
    op.drop_table('clusters')
    op.drop_table('cloud_credentials')
    cluster_provider_enum = sa.Enum('Docker', 'Kubernetes', 'DigitalOcean', name='clusterprovider')
    cluster_provider_enum.drop(op.get_bind(), checkfirst=True)
    cluster_state_enum = sa.Enum(
        'PROVISIONING',
        'PROVISIONED',
        'READY',
        name='clusterstateenum',
    )
    cluster_state_enum.drop(op.get_bind(), checkfirst=True)

    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("DROP TYPE credentialtype;")
        # update state to NOT_READY if it is in provisioning related state
        op.execute(f"""
            UPDATE workers SET state = 'NOT_READY' WHERE state IN ({','.join(repr(v) for v in WORKER_STATE_ADDITIONAL_VALUES)});
        """)
        op.execute("CREATE TYPE workerstateenumtmp AS ENUM ('NOT_READY', 'READY', 'UNREACHABLE');")
        op.execute("ALTER TABLE workers ALTER COLUMN state TYPE workerstateenumtmp USING state::text::workerstateenumtmp;")
        op.execute("DROP TYPE workerstateenum;")
        op.execute("CREATE TYPE workerstateenum AS ENUM ('NOT_READY', 'READY', 'UNREACHABLE');")
        op.execute("ALTER TABLE workers ALTER COLUMN state TYPE workerstateenum USING state::text::workerstateenum;")
        op.execute("DROP TYPE workerstateenumtmp;")
