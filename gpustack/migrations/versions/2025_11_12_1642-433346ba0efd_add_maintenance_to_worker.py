"""add maintenance to worker

Revision ID: 433346ba0efd
Revises: bb9ce21dcc10
Create Date: 2025-11-12 16:42:07.659454

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '433346ba0efd'
down_revision: Union[str, None] = 'bb9ce21dcc10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

WORKER_STATE_ADDITIONAL_VALUE = "MAINTENANCE"

def upgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('maintenance', sa.JSON(), nullable=True))
        
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        conn.execute(sa.text(f"ALTER TYPE workerstateenum ADD VALUE '{WORKER_STATE_ADDITIONAL_VALUE}'"))
    elif conn.dialect.name == 'mysql':
        # Get existing workerstateenum values
        result = conn.execute(
            sa.text("""
                SELECT COLUMN_TYPE
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = 'workers'
                AND COLUMN_NAME = 'state'
                AND TABLE_SCHEMA = DATABASE()
            """)
        ).scalar()

        existing_values = []
        if result:
            enum_str = result.split("enum(")[1].split(")")[0]
            existing_values = [v.strip("'") for v in enum_str.split("','")]

        new_values = existing_values.copy()
        new_values.append(WORKER_STATE_ADDITIONAL_VALUE)
        if len(new_values) >= len(existing_values):
            new_enum_str = "enum('" + "','".join(new_values) + "')"

            # Construct new ALTER TABLE statement
            alter_sql = f"""
                ALTER TABLE workers 
                MODIFY COLUMN state {new_enum_str};
            """

            # Execute modification
            conn.execute(sa.text(alter_sql))

def downgrade() -> None:
    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('maintenance')
