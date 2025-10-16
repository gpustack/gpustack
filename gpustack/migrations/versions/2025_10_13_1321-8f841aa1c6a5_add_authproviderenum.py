"""add authproviderenum

Revision ID: 8f841aa1c6a5
Revises: eeacfbc6a2bf
Create Date: 2025-10-13 13:21:16.618530

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8f841aa1c6a5'
down_revision: Union[str, None] = 'eeacfbc6a2bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

auth_providerenum_enum = sa.Enum('Local', 'OIDC', 'SAML', name='authproviderenum')


def upgrade() -> None:
    auth_providerenum_enum.create(op.get_bind(), checkfirst=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    auth_providerenum_enum.drop(op.get_bind(), checkfirst=True)
    # ### end Alembic commands ###
