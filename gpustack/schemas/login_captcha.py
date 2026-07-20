"""Database ledger for globally single-use login CAPTCHA challenges."""

from datetime import datetime
from typing import ClassVar

from sqlalchemy import Column
from sqlmodel import Field, SQLModel

from gpustack.schemas.common import UTCDateTime


class LoginCaptchaNonce(SQLModel, table=True):
    __tablename__: ClassVar[str] = "login_captcha_nonces"

    nonce_hash: str = Field(primary_key=True, max_length=64)
    expires_at: datetime = Field(
        sa_column=Column(UTCDateTime, nullable=False, index=True)
    )
