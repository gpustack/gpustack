"""Store Benchmark execution identities, monitoring bundles and their capacity policy.

Revision ID: f8c4a291bd65
Revises: c8d3e6f1a902
Create Date: 2026-11-19 01:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGBLOB

from gpustack.migrations.utils import column_exists, table_exists

revision: str = 'f8c4a291bd65'
down_revision: Union[str, None] = 'c8d3e6f1a902'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CAPACITY_TABLE = 'artifact_storage_capacity'
_PAYLOAD_TABLE = 'benchmark_monitoring_payloads'


def upgrade() -> None:
    if table_exists('benchmarks') and not column_exists('benchmarks', 'run_id'):
        op.add_column('benchmarks', sa.Column('run_id', sa.String(64), nullable=True))

    if not table_exists(_CAPACITY_TABLE):
        op.create_table(
            _CAPACITY_TABLE,
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('max_bytes', sa.BigInteger(), nullable=False),
            sa.Column('used_bytes', sa.BigInteger(), nullable=False),
            sa.Column('initialized', sa.Boolean(), nullable=False),
        )

    connection = op.get_bind()
    capacity = sa.table(
        _CAPACITY_TABLE,
        sa.column('id', sa.Integer()),
        sa.column('max_bytes', sa.BigInteger()),
        sa.column('used_bytes', sa.BigInteger()),
        sa.column('initialized', sa.Boolean()),
    )
    if connection.execute(sa.select(capacity.c.id).where(capacity.c.id == 1)).first() is None:
        op.bulk_insert(
            capacity,
            [{'id': 1, 'max_bytes': 0, 'used_bytes': 0, 'initialized': False}],
        )

    if not table_exists(_PAYLOAD_TABLE):
        op.create_table(
            _PAYLOAD_TABLE,
            sa.Column('benchmark_id', sa.Integer(), primary_key=True, autoincrement=False),
            sa.Column('run_id', sa.String(64), nullable=False),
            sa.Column('owner_principal_id', sa.Integer()),
            sa.Column('snapshot', sa.JSON(), nullable=False),
            sa.Column('plan', sa.JSON(), nullable=False),
            sa.Column('snapshot_seq', sa.BigInteger(), nullable=False),
            sa.Column('state', sa.String(32), nullable=False),
            sa.Column('sealed', sa.Boolean(), nullable=False),
            sa.Column('capture_complete', sa.Boolean(), nullable=False),
            sa.Column('data_coverage', sa.String(16), nullable=False),
            sa.Column('hardware_visible', sa.Boolean(), nullable=False),
            sa.Column('captured_until', sa.Double()),
            sa.Column('expires_at', sa.Double()),
            sa.Column('format_version', sa.Integer(), nullable=False),
            sa.Column('payload', sa.LargeBinary().with_variant(LONGBLOB(), 'mysql')),
            sa.Column('payload_sha256', sa.String(64)),
            sa.Column('compressed_bytes', sa.BigInteger(), nullable=False),
            sa.Column('uncompressed_bytes', sa.BigInteger(), nullable=False),
            sa.Column('lease_owner', sa.String(36)),
            sa.Column('lease_until', sa.Double(), nullable=False),
            sa.Column('fencing_token', sa.BigInteger(), nullable=False),
            sa.Column('write_seq', sa.BigInteger(), nullable=False),
            sa.Column('finalize_requested_at', sa.Double()),
            sa.Column('repair_requested', sa.Boolean(), nullable=False),
            sa.Column('retry_at', sa.Double(), nullable=False),
            sa.Column('last_error', sa.String(128)),
        )
    names = {item['name'] for item in sa.inspect(connection).get_indexes(_PAYLOAD_TABLE)}
    for index, fields in (
        ('ix_benchmark_monitoring_payloads_due', ['sealed', 'retry_at']),
        ('ix_benchmark_monitoring_payloads_expiry', ['expires_at']),
    ):
        if index not in names:
            op.create_index(index, _PAYLOAD_TABLE, fields)


def downgrade() -> None:
    if table_exists(_PAYLOAD_TABLE):
        payloads = sa.table(_PAYLOAD_TABLE, sa.column('benchmark_id'))
        if op.get_bind().execute(sa.select(payloads.c.benchmark_id).limit(1)).first():
            raise RuntimeError('Remove Benchmark monitoring archives before downgrading')
        op.drop_table(_PAYLOAD_TABLE)
    if table_exists(_CAPACITY_TABLE):
        op.drop_table(_CAPACITY_TABLE)
    if table_exists('benchmarks') and column_exists('benchmarks', 'run_id'):
        op.drop_column('benchmarks', 'run_id')
