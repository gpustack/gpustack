"""add cache services

Introduces the shared KV cache service resource:

1. New ``cache_services`` table. A cache service is an Org-scoped,
   cluster-scoped resource that model deployments attach to for shared
   KV cache. ``mode`` distinguishes managed (cache server containers
   run on cluster workers) from external (connection reference to a
   cache system running outside GPUStack).

2. New ``cache_service_instances`` table: one row per cache server
   container of a managed service. The provider's declared topology
   dictates the desired set (singleton: one instance on the user-picked
   worker; per_node: one instance per active worker of the cluster,
   narrowed by the service's ``worker_selector`` labels when set).
   Runtime fields (ports, state, health, restart bookkeeping) live
   here; the service row carries the aggregate state.

3. ``model_instances.cache_config`` JSON column: the shared-cache
   connection info resolved at instance creation, so the worker can
   inject engine config without a server round-trip.


Revision ID: d5e8f0a1b2c3
Revises: a3f5c1d9e0b2
Create Date: 2026-07-25 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'd5e8f0a1b2c3'
down_revision: Union[str, None] = 'a3f5c1d9e0b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'cache_services',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            'provider_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False
        ),
        sa.Column(
            'provider_version', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column('mode', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('worker_selector', sa.JSON(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('endpoint', sa.JSON(), nullable=True),
        sa.Column('state', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('healthy', sa.Boolean(), nullable=True),
        sa.Column('last_check_at', sa.DateTime(), nullable=True),
        sa.Column('restart_on_error', sa.Boolean(), nullable=True),
        sa.Column('owner_principal_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['cluster_id'], ['clusters.id']),
        sa.ForeignKeyConstraint(
            ['owner_principal_id'], ['principals.id'], ondelete='CASCADE'
        ),
        sa.UniqueConstraint(
            'owner_principal_id', 'name', name='uix_cache_services_name_per_owner'
        ),
    )
    op.create_index(
        op.f('ix_cache_services_name'), 'cache_services', ['name'], unique=False
    )

    op.create_table(
        'cache_service_instances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('cache_service_id', sa.Integer(), nullable=False),
        sa.Column('worker_id', sa.Integer(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('port', sa.Integer(), nullable=True),
        sa.Column('metrics_port', sa.Integer(), nullable=True),
        sa.Column('state', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('healthy', sa.Boolean(), nullable=True),
        sa.Column('last_check_at', sa.DateTime(), nullable=True),
        sa.Column('restart_count', sa.Integer(), nullable=True),
        sa.Column('last_restart_time', sa.DateTime(), nullable=True),
        sa.Column('spec_digest', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(
            ['cache_service_id'], ['cache_services.id'], ondelete='CASCADE'
        ),
        sa.UniqueConstraint(
            'cache_service_id',
            'worker_id',
            name='uix_cache_service_instances_service_worker',
        ),
    )
    op.create_index(
        op.f('ix_cache_service_instances_name'),
        'cache_service_instances',
        ['name'],
        unique=False,
    )
    op.create_index(
        op.f('ix_cache_service_instances_cache_service_id'),
        'cache_service_instances',
        ['cache_service_id'],
        unique=False,
    )

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('cache_config', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('cache_config')

    op.drop_index(
        op.f('ix_cache_service_instances_cache_service_id'),
        table_name='cache_service_instances',
    )
    op.drop_index(
        op.f('ix_cache_service_instances_name'),
        table_name='cache_service_instances',
    )
    op.drop_table('cache_service_instances')

    op.drop_index(op.f('ix_cache_services_name'), table_name='cache_services')
    op.drop_table('cache_services')
