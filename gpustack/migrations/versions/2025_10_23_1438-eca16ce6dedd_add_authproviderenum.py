"""add authproviderenum

Revision ID: eca16ce6dedd
Revises: 35e8ebcae667
Create Date: 2025-10-23 14:38:50.060479

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'eca16ce6dedd'
down_revision: Union[str, None] = '35e8ebcae667'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


auth_providerenum_enum = sa.Enum('Local', 'OIDC', 'SAML', name='authproviderenum')


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == 'postgresql':
        auth_providerenum_enum.create(bind, checkfirst=True)
        op.execute("ALTER TABLE users ALTER COLUMN source DROP DEFAULT")
        op.alter_column('users', 'source',
            type_=sa.Enum('Local', 'OIDC', 'SAML', name='authproviderenum'),
            postgresql_using="source::text::authproviderenum"
        )
        op.execute("ALTER TABLE users ALTER COLUMN source SET DEFAULT 'Local'")
    elif dialect == "mysql":
        op.alter_column('users', 'source',
            type_=auth_providerenum_enum,
            existing_type=sa.String(length=255)
        )
    # ### end Alembic commands ###


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('source',
            existing_type=sa.Enum('Local', 'OIDC', 'SAML', name='authproviderenum'),
            type_=sa.VARCHAR(length=255),
            nullable=False,
            existing_server_default=sa.text("'Local'"))
        
    if dialect == 'postgresql':
        auth_providerenum_enum.drop(bind, checkfirst=True)
    # ### end Alembic commands ###
