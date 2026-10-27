"""usage details: request ids and ttft

Three per-request columns on ``model_usage_details`` and its archive,
carrying what the token-usage gateway plugin reports from 1.2.0 on:

1. ``ttft_ms`` — milliseconds from request entry to the first response
   body chunk, streaming only. No ``duration_ms`` beside it: the
   duration is ``completed_at - started_at``, both already stored.

2. ``request_id`` — Envoy's ``x-request-id``. The id an audit lookup
   keys on, because it exists for every tracked request whatever the
   endpoint or outcome and matches the Envoy access log. **Indexed, not
   unique**: it identifies a downstream request while a row is written
   per filter-chain run, so a fallback pass legitimately produces two
   rows under one value.

3. ``upstream_response_id`` — the model's own id for the response
   (``chatcmpl-…`` / ``resp_…`` / ``msg_…``), which is what a caller
   reads off an SDK response.

All three are nullable with no backfill. Rows written before this
revision have no such values, and no default could be anything but a
fabricated one.

Revision ID: f1a2b3c4d5e6
Revises: e4a1c8b7d0f3
Create Date: 2026-09-11 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.migrations.utils import column_exists, index_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e4a1c8b7d0f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Hot table and archive must share a column layout: bulk archival is an
# ``INSERT ... SELECT`` over the archive's full column list, so the two lining
# up is what makes it correct. ``UsageDetailsArchiver`` compares the two sets
# at construction and raises, so adding a column to one table and not the
# other fails the server at startup rather than at the next sweep.
_TABLES = ('model_usage_details', 'model_usage_details_archive')

_COLUMNS = (
    ('ttft_ms', sa.Integer()),
    ('request_id', sqlmodel.sql.sqltypes.AutoString()),
    ('upstream_response_id', sqlmodel.sql.sqltypes.AutoString()),
)


def _index_name(table_name: str) -> str:
    return f'ix_{table_name}_request_id'


def upgrade() -> None:
    for table_name in _TABLES:
        if not table_exists(table_name):
            continue
        for column_name, column_type in _COLUMNS:
            if column_exists(table_name, column_name):
                continue
            op.add_column(
                table_name, sa.Column(column_name, column_type, nullable=True)
            )
        index_name = _index_name(table_name)
        if not index_exists(table_name, index_name):
            op.create_index(index_name, table_name, ['request_id'])


def downgrade() -> None:
    for table_name in _TABLES:
        if not table_exists(table_name):
            continue
        index_name = _index_name(table_name)
        if index_exists(table_name, index_name):
            op.drop_index(index_name, table_name=table_name)
        for column_name, _ in _COLUMNS:
            if column_exists(table_name, column_name):
                op.drop_column(table_name, column_name)
