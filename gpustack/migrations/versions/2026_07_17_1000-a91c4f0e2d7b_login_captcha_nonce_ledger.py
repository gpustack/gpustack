"""Add the shared login CAPTCHA nonce ledger.

Revision ID: a91c4f0e2d7b
Revises: c4d7e8f9a0b1
Create Date: 2026-07-17 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "a91c4f0e2d7b"
down_revision: Union[str, None] = "c4d7e8f9a0b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "login_captcha_nonces",
        sa.Column("nonce_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("nonce_hash"),
    )
    op.create_index(
        op.f("ix_login_captcha_nonces_expires_at"),
        "login_captcha_nonces",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_login_captcha_nonces_expires_at"),
        table_name="login_captcha_nonces",
    )
    op.drop_table("login_captcha_nonces")
