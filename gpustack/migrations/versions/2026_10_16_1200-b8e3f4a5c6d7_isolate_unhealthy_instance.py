"""add isolate support for unhealthy model instances

Revision ID: b8e3f4a5c6d7
Revises: a7e2c9d4f681
Create Date: 2026-10-16 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


revision: str = 'b8e3f4a5c6d7'
down_revision: Union[str, None] = 'a7e2c9d4f681'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Add new columns to model_instances
    if table_exists('model_instances'):
        if not column_exists('model_instances', 'isolated_since'):
            op.add_column(
                'model_instances',
                sa.Column('isolated_since', sa.DateTime(), nullable=True),
            )
        if not column_exists('model_instances', 'isolation_reason'):
            op.add_column(
                'model_instances',
                sa.Column('isolation_reason', sa.Text(), nullable=True),
            )

        # Extend the state enum with ISOLATED value.
        if conn.dialect.name == 'postgresql':
            result = conn.execute(
                sa.text(
                    "SELECT unnest(enum_range(NULL::modelinstancestateenum))::text"
                )
            ).fetchall()
            existing = [row[0] for row in result]
            if 'ISOLATED' not in existing:
                conn.execute(
                    sa.text(
                        "ALTER TYPE modelinstancestateenum ADD VALUE 'ISOLATED'"
                    )
                )
        elif conn.dialect.name == 'mysql':
            result = conn.execute(
                sa.text(
                    """
                    SELECT COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                    FROM information_schema.COLUMNS
                    WHERE TABLE_NAME = 'model_instances'
                    AND COLUMN_NAME = 'state'
                    AND TABLE_SCHEMA = DATABASE()
                    """
                )
            ).first()
            if result is not None:
                column_type, is_nullable, default = result
                enum_str = column_type.split("enum(")[1].split(")")[0]
                existing = [v.strip("'") for v in enum_str.split("','")]
                if 'ISOLATED' not in existing:
                    new_values = existing + ['ISOLATED']
                    new_enum_str = "enum('" + "','".join(new_values) + "')"
                    constraints = "" if is_nullable == 'YES' else " NOT NULL"
                    if default is not None:
                        constraints += f" DEFAULT '{default}'"
                    conn.execute(
                        sa.text(
                            f"ALTER TABLE model_instances MODIFY COLUMN "
                            f"state {new_enum_str}{constraints};"
                        )
                    )


def downgrade() -> None:
    conn = op.get_bind()

    # Note: Downgrading the ISOLATED enum value is intentionally skipped.
    # The ALTER TYPE ... DROP VALUE syntax is not portable across the four
    # supported databases (PostgreSQL / openGauss / MySQL / OceanBase), and
    # removing it would fail on any row already holding ISOLATED. Keeping the
    # value in the enum is harmless — it simply becomes unused again — which
    # matches how the existing migrations treat STARTING and UNREACHABLE.

    if table_exists('model_instances'):
        if column_exists('model_instances', 'isolated_since'):
            op.drop_column('model_instances', 'isolated_since')
        if column_exists('model_instances', 'isolation_reason'):
            op.drop_column('model_instances', 'isolation_reason')
