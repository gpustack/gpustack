from datetime import date
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel
from gpustack.mixins.active_record import ActiveRecordMixin


class OperationEnum(str, Enum):
    CHAT_COMPLETION = "chat_completion"


class ModelUsage(SQLModel, ActiveRecordMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=None, foreign_key="user.id")
    model_id: int = Field(default=None, foreign_key="model.id")
    date: date
    prompt_token_count: int
    completion_token_count: int
    request_count: int
    operation: OperationEnum

    model_config = ConfigDict(protected_namespaces=())
