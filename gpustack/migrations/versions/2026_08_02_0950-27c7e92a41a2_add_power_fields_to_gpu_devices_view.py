"""add power fields to gpu_devices_view

Revision ID: 27c7e92a41a2
Revises: ddb9eb4a7eab
Create Date: 2026-08-02 09:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from gpustack.migrations.utils import is_opengauss
from gpustack.schemas.stmt import (
    worker_after_drop_view_stmt_sqlite,
    worker_after_drop_view_stmt_mysql,
    worker_after_drop_view_stmt_postgres,
    worker_after_create_view_stmt_sqlite,
    worker_after_create_view_stmt_mysql,
    worker_after_create_view_stmt_postgres,
    worker_after_create_view_stmt_opengauss,
)

# revision identifiers, used by Alembic.
revision: str = '27c7e92a41a2'
down_revision: Union[str, None] = 'ddb9eb4a7eab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    dialect_name = op.get_bind().dialect.name

    # Drop existing view
    if dialect_name == 'mysql':
        op.execute(worker_after_drop_view_stmt_mysql)
    elif dialect_name == 'postgresql':
        op.execute(worker_after_drop_view_stmt_postgres)
    else:
        op.execute(worker_after_drop_view_stmt_sqlite)

    # Recreate view with new power fields
    if dialect_name == 'mysql':
        op.execute(worker_after_create_view_stmt_mysql)
    elif dialect_name == 'postgresql':
        if is_opengauss(conn):
            op.execute(worker_after_create_view_stmt_opengauss)
        else:
            op.execute(worker_after_create_view_stmt_postgres)
    else:
        op.execute(worker_after_create_view_stmt_sqlite)


def downgrade() -> None:
    conn = op.get_bind()
    dialect_name = op.get_bind().dialect.name

    # Drop the view; it will be recreated with the old schema on next
    # server start via the after_create event listener.
    if dialect_name == 'mysql':
        op.execute(worker_after_drop_view_stmt_mysql)
    elif dialect_name == 'postgresql':
        op.execute(worker_after_drop_view_stmt_postgres)
    else:
        op.execute(worker_after_drop_view_stmt_sqlite)
