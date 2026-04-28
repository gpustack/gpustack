from datetime import date
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlalchemy import BigInteger, Column, ForeignKey, Integer
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
    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("users.id", ondelete="SET NULL")),
    )
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("models.id", ondelete="SET NULL")),
    )
    model_name: str = Field(default=...)
    provider_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer, ForeignKey("model_providers.id", ondelete="SET NULL")
        ),
    )
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("api_keys.id", ondelete="SET NULL")),
    )
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    date: date
    prompt_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    completion_token_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    prompt_cached_token_count: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, default=0)
    )
    request_count: int = Field(
        default=..., sa_column=Column(BigInteger, nullable=False)
    )
    operation: Optional[OperationEnum] = Field(default=None)

    model_config = ConfigDict(protected_namespaces=())
