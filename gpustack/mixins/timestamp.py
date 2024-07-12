from datetime import timezone
import sqlalchemy as sa


class UTCDateTime(sa.TypeDecorator):
    impl = sa.TIMESTAMP(timezone=True)

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            # Ensure the datetime is in UTC before storing
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            # Assume stored datetime is in UTC and attach tzinfo
            value = value.replace(tzinfo=timezone.utc)
        return value


class TimestampsMixin:
    """Mixin that define timestamp columns."""

    __abstract__ = True
    __config__ = None

    __created_at_name__ = "created_at"
    __updated_at_name__ = "updated_at"
    __datetime_func__ = sa.func.now()

    created_at = sa.Column(
        __created_at_name__,
        UTCDateTime,
        default=__datetime_func__,
        nullable=False,
    )

    updated_at = sa.Column(
        __updated_at_name__,
        UTCDateTime,
        default=__datetime_func__,
        onupdate=__datetime_func__,
        nullable=False,
    )
