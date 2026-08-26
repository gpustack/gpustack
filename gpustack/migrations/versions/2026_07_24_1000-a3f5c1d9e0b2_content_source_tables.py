"""content sources: source tables and materialized entries

Creates five tables for the shared "source" pipeline (catalog / community
backend / runner) and adds two columns to ``inference_backends``:

- ``inference_runner_sources`` / ``runner_override_entries``: the built-in-
  backend-version source rows and the materialized per-platform overrides, which
  the three consumption points serve in place of the packaged gpustack-runner
  catalog. Each override carries the same ``source_name`` / ``source_type`` stamp
  ``catalog_model_entries`` has, for origin display.
- ``catalog_sources`` / ``catalog_model_entries``: model-catalog source rows and
  the materialized catalog records (model sets + draft models, full-rewritten by
  the leader).
- ``inference_backend_sources``: community-backend source rows, smart-merged into
  the existing ``inference_backends`` table by the leader rather than into a table
  of their own — hence no materialized table of its own.
- ``inference_backends.source_name`` / ``source_type``: the card-level source
  origin for those merged rows, and the only ALTER here.

A source's ``content`` is LONGTEXT on MySQL, where ``TEXT`` caps at 64 KiB and
every published document is past it — the community-backend catalog by four
times — so a refresh would die on "Data too long for column 'content'" and no
kind would ever update. PostgreSQL and SQLite put no length limit on their text
types and keep TEXT.

Revision ID: a3f5c1d9e0b2
Revises: 367a3982fcde
Create Date: 2026-07-24 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql
import sqlmodel
import gpustack

# revision identifiers, used by Alembic.
revision: str = 'a3f5c1d9e0b2'
down_revision: Union[str, None] = '367a3982fcde'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'runner_override_entries',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('backend', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('backend_version', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            'original_backend_version',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
        ),
        sa.Column('backend_variant', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('service', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('service_version', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('platform', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('docker_image', sa.Text(), nullable=False),
        sa.Column('deprecated', sa.Boolean(), nullable=False),
        sa.Column('source_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('runner_override_entries', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_runner_override_entries_backend'),
            ['backend'],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f('ix_runner_override_entries_service'),
            ['service'],
            unique=False,
        )

    op.create_table(
        'inference_runner_sources',
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
    with op.batch_alter_table('inference_runner_sources', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_inference_runner_sources_name'),
            ['name'],
            unique=True,
        )

    op.create_table(
        'catalog_sources',
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
    with op.batch_alter_table('catalog_sources', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_catalog_sources_name'),
            ['name'],
            unique=True,
        )

    op.create_table(
        'catalog_model_entries',
        sa.Column('created_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('updated_at', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('deleted_at', gpustack.schemas.common.UTCDateTime(), nullable=True),
        sa.Column('kind', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('source_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('source_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'kind', 'name', name='uix_catalog_model_entries_kind_name'
        ),
    )
    with op.batch_alter_table('catalog_model_entries', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_catalog_model_entries_kind'),
            ['kind'],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f('ix_catalog_model_entries_name'),
            ['name'],
            unique=False,
        )

    op.create_table(
        'inference_backend_sources',
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
    with op.batch_alter_table('inference_backend_sources', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_inference_backend_sources_name'),
            ['name'],
            unique=True,
        )

    # Card-level source origin stamped onto community backends.
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'source_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column(
                'source_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('inference_backends', schema=None) as batch_op:
        batch_op.drop_column('source_type')
        batch_op.drop_column('source_name')

    with op.batch_alter_table('inference_backend_sources', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inference_backend_sources_name'))
    op.drop_table('inference_backend_sources')

    with op.batch_alter_table('catalog_model_entries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_catalog_model_entries_name'))
        batch_op.drop_index(batch_op.f('ix_catalog_model_entries_kind'))
    op.drop_table('catalog_model_entries')

    with op.batch_alter_table('catalog_sources', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_catalog_sources_name'))
    op.drop_table('catalog_sources')

    with op.batch_alter_table('inference_runner_sources', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_inference_runner_sources_name'))
    op.drop_table('inference_runner_sources')

    with op.batch_alter_table('runner_override_entries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_runner_override_entries_service'))
        batch_op.drop_index(batch_op.f('ix_runner_override_entries_backend'))
    op.drop_table('runner_override_entries')
