"""v2.1.1 migrate legacy backend_source to CUSTOM

Revision ID: 8ad0f94c92e8
Revises: 53667f33f000
Create Date: 2026-03-09 13:59:00.000000

"""
import logging
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8ad0f94c92e8'
down_revision: Union[str, None] = '53667f33f000'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger(__name__)

def upgrade() -> None:
    """Migrate legacy backends with null backend_source to CUSTOM.

    This migration addresses legacy inference backends that were created
    before backend_source field was introduced. Backends with names ending
    in '-custom' are migrated to have backend_source='custom'.
    """
    conn = op.get_bind()

    result = conn.execute(
        sa.text("""
            UPDATE inference_backends
            SET backend_source = :source
            WHERE backend_source IS NULL
            AND backend_name LIKE :pattern
        """),
        {"source": "CUSTOM", "pattern": "%-custom"}
    )

    try:
        affected_rows = result.rowcount
        if affected_rows > 0:
            logger.info(f"Migrated {affected_rows} legacy backend(s) to backend_source='CUSTOM'")
    except Exception:
        pass


def downgrade() -> None:
    """Revert backend_source migration for legacy custom backends."""
    conn = op.get_bind()

    conn.execute(
        sa.text("""
            UPDATE inference_backends
            SET backend_source = NULL
            WHERE backend_source = :source
            AND backend_name LIKE :pattern
        """),
        {"source": "CUSTOM", "pattern": "%-custom"}
    )
