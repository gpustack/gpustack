"""add is_active field to users

Revision ID: 2025_10_07_add_is_active
Revises: f3d36b2d11f8
Create Date: 2025-10-07 23:41:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2025_10_07_add_is_active'
down_revision: Union[str, None] = 'f3d36b2d11f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add is_active column to users table.
    
    This migration adds the is_active boolean field to the users table
    with a default value of True to ensure existing users remain active.
    """
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'is_active', 
                sa.Boolean(), 
                nullable=False, 
                server_default='1',
                comment='Indicates if the user account is active and can authenticate'
            )
        )


def downgrade() -> None:
    """Remove is_active column from users table."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('is_active')
