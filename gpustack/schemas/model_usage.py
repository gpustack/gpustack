from datetime import date
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlalchemy import BigInteger, Column
from sqlmodel import Field, SQLModel
from gpustack.mixins.active_record import ActiveRecordMixin


class OperationEnum(str, Enum):
    COMPLETION = "completion"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    AUDIO_SPEECH = "audio_speech"
    AUDIO_TRANSCRIPTION = "audit_transcription"


class ModelUsage(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usages'
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    model_id: Optional[int] = Field(default=None, foreign_key="models.id")
    provider_id: Optional[int] = Field(default=None, foreign_key="model_providers.id")
    model_name: str = Field(default=...)
    access_key: Optional[str] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    request_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    operation: Optional[OperationEnum] = Field(default=None)

    model_config = ConfigDict(protected_namespaces=())
