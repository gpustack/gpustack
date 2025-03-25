from datetime import datetime, timezone
from enum import Enum
from logging import Logger
from typing import Dict, List, Optional
from gpustack.schemas.models import Model


class EventLevelEnum(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Event:
    def __init__(
        self,
        level: EventLevelEnum,
        action: str,
        message: str,
        reason: Optional[str] = None,
        **kwargs,
    ):
        self.level = level
        self.action = action
        self.message = message
        self.reason = reason
        self.timestamp = datetime.now(timezone.utc).isoformat()
        for key, value in kwargs.items():
            setattr(self, key, value)


class EventCollector:
    def __init__(self, model: Model, event_logger: Optional[Logger] = None):
        self.events: List[Event] = []
        self._event_logger = event_logger
        self._model = model
        self._event_keys: Dict[str, True] = {}

    def add(self, event_level: EventLevelEnum, action: str, message: str, **kwargs):
        key = f"{event_level}-{action}-{message}"
        if key in self._event_keys:
            return

        self.events.append(Event(event_level, action, message, **kwargs))
        self._event_keys[key] = True

        if self._event_logger:
            msg = f"Action: {action}, {message}, {kwargs}, model: {self._model.readable_source}"
            if event_level == EventLevelEnum.ERROR:
                self._event_logger.error(msg)
            elif event_level == EventLevelEnum.WARNING:
                self._event_logger.warning(msg)
            else:
                self._event_logger.info(msg)
