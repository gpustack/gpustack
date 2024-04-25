from .active_record import ActiveRecordMixin
from .timestamp import TimestampsMixin


class BaseModelMixin(ActiveRecordMixin, TimestampsMixin):
    pass
