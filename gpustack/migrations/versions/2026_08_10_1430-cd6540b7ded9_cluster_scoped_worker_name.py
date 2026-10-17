"""cluster-scoped worker name uniqueness

Revision ID: cd6540b7ded9
Revises: 367a3982fcde
Create Date: 2026-08-10 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'cd6540b7ded9'
down_revision: Union[str, None] = '367a3982fcde'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change workers.name from globally unique to cluster-scoped unique.

    Before: workers.name is globally unique (ix_workers_name)
    After:  workers.(cluster_id, name) is unique per cluster

    This allows workers with the same name in different clusters,
    which is required for multi-cluster deployments where each cluster
    may have its own identically-named worker (e.g., "worker-1").
    """
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == 'postgresql':
        # Find and drop the existing unique constraint/index on workers.name
        for row in bind.execute(
            sa.text(
                """
                SELECT c.conname FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE t.relname = 'workers'
                AND c.contype = 'u'
                AND array_length(c.conkey, 1) = 1
                AND (SELECT a.attname FROM pg_attribute a
                     WHERE a.attrelid = c.conrelid
                     AND a.attnum = c.conkey[1]) = 'name'
                """
            )
        ):
            op.execute(
                f'ALTER TABLE workers DROP CONSTRAINT IF EXISTS "{row[0]}"'
            )
        # Also check for the auto-created unique index
        op.execute('DROP INDEX IF EXISTS ix_workers_name')
        # Create composite unique constraint
        op.execute(
            """
            ALTER TABLE workers
            ADD CONSTRAINT uix_workers_cluster_name
            UNIQUE (cluster_id, name)
            WHERE deleted_at IS NULL
            """
        )
    else:
        # MySQL / SQLite: drop the unique index on name
        op.execute("DROP INDEX IF EXISTS ix_workers_name ON workers")
        # MySQL requires a different syntax for partial unique indexes
        # For MySQL, we create a regular composite unique constraint
        # Note: MySQL doesn't support partial indexes, so this includes deleted rows
        # The application layer already filters by deleted_at
        op.execute(
            """
            ALTER TABLE workers
            ADD CONSTRAINT uix_workers_cluster_name
            UNIQUE (cluster_id, name)
            """
        )


def downgrade() -> None:
    """Restore global uniqueness on workers.name.

    Warning: This may fail if there are workers with duplicate names
    in different clusters.
    """
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == 'postgresql':
        # Drop the composite constraint
        op.execute('ALTER TABLE workers DROP CONSTRAINT IF EXISTS uix_workers_cluster_name')
        # Restore the original unique index
        op.create_index('ix_workers_name', 'workers', ['name'], unique=True)
    else:
        # MySQL / SQLite
        op.execute('ALTER TABLE workers DROP INDEX uix_workers_cluster_name')
        op.create_index('ix_workers_name', 'workers', ['name'], unique=True)
