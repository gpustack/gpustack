"""migrate model instance port to ports

Revision ID: e30134bd18dc
Revises: 433346ba0efd
Create Date: 2025-11-17 15:55:51.794278

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e30134bd18dc'
down_revision: Union[str, None] = '433346ba0efd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    model_instances = sa.table(
        "model_instances",
        sa.column("id", sa.Integer),
        sa.column("port", sa.Integer),
        sa.column("ports", sa.JSON),
    )

    conn = op.get_bind()

    rows = conn.execute(
        sa.select(
            model_instances.c.id,
            model_instances.c.port,
            model_instances.c.ports,
        )
    ).fetchall()

    for r in rows:
        if r.port is not None and (r.ports is None or r.ports == []):
            conn.execute(
                model_instances.update()
                .where(model_instances.c.id == r.id)
                .values(ports=[r.port])
            )



def downgrade() -> None:
    pass
