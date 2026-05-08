"""add_api_key_usage_stats

Revision ID: acb303585994
Revises: 8bf38a6bb3b5
Create Date: 2026-05-08 08:45:45.608786

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'acb303585994'
down_revision: Union[str, None] = '8bf38a6bb3b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'api_keys',
        sa.Column(
            'total_requests', sa.BigInteger(), nullable=False, server_default=sa.text('0')
        ),
    )
    op.add_column(
        'api_keys',
        sa.Column(
            'total_tokens', sa.BigInteger(), nullable=False, server_default=sa.text('0')
        ),
    )
    op.add_column(
        'api_keys',
        sa.Column(
            'total_cached_tokens',
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text('0'),
        ),
    )


def downgrade() -> None:
    op.drop_column('api_keys', 'total_cached_tokens')
    op.drop_column('api_keys', 'total_tokens')
    op.drop_column('api_keys', 'total_requests')
