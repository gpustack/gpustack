"""Add model instance drain fields for graceful replica delete.

Adds ``drain_started_at`` and ``drain_idle`` on ``model_instances`` so scale-down
/ DELETE can enter DRAINING, wait for in-flight proxy requests (or timeout),
then hard-delete.

Revision ID: a1b2c3d4e5f6
Revises: 367a3982fcde
Create Date: 2026-08-16 11:18:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from gpustack.schemas.common import UTCDateTime
from gpustack.migrations.utils import column_exists

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "367a3982fcde"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not column_exists("model_instances", "drain_started_at"):
        op.add_column(
            "model_instances",
            sa.Column("drain_started_at", UTCDateTime(), nullable=True),
        )
    if not column_exists("model_instances", "drain_idle"):
        op.add_column(
            "model_instances",
            sa.Column(
                "drain_idle",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    if column_exists("model_instances", "drain_idle"):
        op.drop_column("model_instances", "drain_idle")
    if column_exists("model_instances", "drain_started_at"):
        op.drop_column("model_instances", "drain_started_at")
