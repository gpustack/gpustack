from datetime import datetime
import sqlalchemy as sa


class TimestampsMixin:
    """Mixin that define timestamp columns."""

    __abstract__ = True
    __config__ = None

    __created_at_name__ = "created_at"
    __updated_at_name__ = "updated_at"

    @staticmethod
    def get_local_time():
        local_timezone = datetime.now().astimezone().tzinfo
        return datetime.now(local_timezone)

    created_at = sa.Column(
        __created_at_name__,
        sa.TIMESTAMP(timezone=True),
        default=get_local_time,
        nullable=False,
    )

    updated_at = sa.Column(
        __updated_at_name__,
        sa.TIMESTAMP(timezone=True),
        default=get_local_time,
        onupdate=get_local_time,
        nullable=False,
    )
