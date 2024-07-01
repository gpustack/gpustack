from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column
from sqlmodel import Field, SQLModel
from gpustack.mixins.active_record import ActiveRecordMixin
from gpustack.schemas.common import pydantic_column_type


class OperationEnum(str, Enum):
    CHAT_COMPLETION = "chat_completion"


class ResourceClaim(BaseModel):
    memory: Optional[int] = Field(default=None)  # in bytes
    gpu_memory: Optional[int] = Field(default=None)  # in bytes


class ModelUsage(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usages'
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=None, foreign_key="users.id")
    model_id: int = Field(default=None, foreign_key="models.id")
    date: date
    prompt_token_count: int
    completion_token_count: int
    request_count: int
    operation: OperationEnum
    resource_claim: Optional[ResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ResourceClaim)), default=None
    )

    model_config = ConfigDict(protected_namespaces=())
