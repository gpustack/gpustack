"""add cache services

Introduces the shared KV cache service resource:

1. New ``cache_services`` table. A cache service is an Org-scoped,
   cluster-scoped resource that model deployments attach to for shared
   KV cache; the platform runs its cache servers on cluster workers.

2. New ``cache_service_instances`` table: one row per cache server
   container of a service. The provider's declared topology
   dictates the desired set (singleton: one instance on the user-picked
   worker; per_node: one instance per active worker of the cluster,
   narrowed by the service's ``worker_selector`` labels when set).
   Runtime fields (ports, state, health, restart bookkeeping) live
   here; the service row carries the aggregate state.

3. ``model_instances.cache_config`` JSON column: the shared-cache
   connection info resolved at instance creation, so the worker can
   inject engine config without a server round-trip.

4. New ``cache_provider_sources`` / ``cache_provider_entries`` tables:
   where the provider catalog a service picks from comes from — the
   packaged baseline and the document an admin configures in its place —
   and the declarations the leader materializes out of them, which is
   what every reader queries.

   The source table has the same shape as the other content sources,
   which the shared source layer reads through ``SourceMixin``.
   ``content`` is LONGTEXT on MySQL, where ``TEXT`` caps at 64 KiB and a
   catalog carrying every provider's declaration is past it. PostgreSQL
   keeps TEXT, which has no length limit there.


This revision is edited in place as the feature it creates changes, rather
than each change getting a revision of its own. It ships in no release: cache
services exist only in v2.3.0rc1, and upgrading from an rc is not a supported
path — such a cluster starts from a fresh database. A database that already
applied an earlier form of this file will not receive the later ones, which is
the same thing said differently.

Revision ID: d5e8f0a1b2c3
Revises: a3f5c1d9e0b2
Create Date: 2026-07-25 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
import sqlmodel

import gpustack


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
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('worker_selector', sa.JSON(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
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
        sa.Column(
            'component',
            sa.String(length=64),
            nullable=False,
            server_default='',
        ),
        sa.Column('component_addresses', sa.JSON(), nullable=True),
        sa.Column('ports', sa.JSON(), nullable=True),
        sa.Column('port', sa.Integer(), nullable=True),
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
        # A component places one instance per worker; the database says so
        # too, so two reconcile passes racing over the same missing row
        # cannot both create it.
        sa.UniqueConstraint(
            'cache_service_id',
            'component',
            'worker_id',
            name='uix_cache_service_instances_component_per_worker',
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

    op.create_table(
        'cache_provider_sources',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            'content', sa.Text().with_variant(mysql.LONGTEXT(), 'mysql'), nullable=True
        ),
        sa.Column('url', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('content_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('remote_hash', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('auto_update_hours', sa.Integer(), nullable=False),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('cache_provider_sources', schema=None) as batch_op:
        # The rows are addressed by name ('builtin' / 'custom'), and writers
        # check-then-write, which two leaders can interleave.
        batch_op.create_index(
            batch_op.f('ix_cache_provider_sources_name'),
            ['name'],
            unique=True,
        )

    op.create_table(
        'cache_provider_entries',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        # Card order, which is the document's own order.
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('source_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        # The upsert key, named so the constraint is droppable by name on every
        # dialect.
        sa.UniqueConstraint('name', name='uix_cache_provider_entries_name'),
    )
    with op.batch_alter_table('cache_provider_entries', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_cache_provider_entries_name'),
            ['name'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('cache_provider_entries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_cache_provider_entries_name'))
    op.drop_table('cache_provider_entries')

    with op.batch_alter_table('cache_provider_sources', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_cache_provider_sources_name'))
    op.drop_table('cache_provider_sources')

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
