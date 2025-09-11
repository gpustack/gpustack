from datetime import datetime
from typing import Optional
from sqlmodel import Field
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


class TimestampsMixin:
    """Mixin that define timestamp columns."""

    __abstract__ = True
    __config__ = None

    __created_at_name__ = "created_at"
    __updated_at_name__ = "updated_at"
    __deleted_at_name__ = "deleted_at"
    __datetime_func__ = sa.func.now()

    created_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __created_at_name__,
            "default": __datetime_func__,
            "nullable": False,
        },
        default=None,
    )

    updated_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __updated_at_name__,
            "default": __datetime_func__,
            "onupdate": __datetime_func__,
            "nullable": False,
        },
        default=None,
    )

    deleted_at: Optional[datetime] = Field(
        sa_type=UTCDateTime,
        sa_column_kwargs={
            "name": __deleted_at_name__,
            "default": None,
            "nullable": True,
        },
        default=None,
    )
