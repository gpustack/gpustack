from datetime import date
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlalchemy import BigInteger, Column, Integer
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
    # FK-less on purpose — the whole table mirrors ``model_usage_details`` /
    # ``metered_usage``: a usage row is an attribution / audit record that must
    # outlive every entity it references. A ``SET NULL`` on parent delete would
    # erase which user / tenant / model / key the usage belongs to; keeping the
    # raw ids preserves attribution, and the read path resolves "does this still
    # exist?" live (marking gone entities ``(Deleted)``). The ``*_name`` columns
    # hold display snapshots alongside for when an id no longer resolves.
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    user_name: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_name: str = Field(default=...)
    model_route_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    model_route_name: Optional[str] = Field(default=None)
    provider_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    provider_name: Optional[str] = Field(default=None)
    provider_type: Optional[str] = Field(default=None)
    cluster_name: Optional[str] = Field(default=None)
    api_key_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    api_key_name: Optional[str] = Field(default=None)
    access_key: Optional[str] = Field(default=None)
    api_key_is_custom: Optional[bool] = Field(default=None)
    # Tenant scope. Denormalized from the api_key/model used for the request.
    # This is the model/deployment owner principal.
    # NULL = global (admin acting in "All" mode). FK-less — see ``user_id``.
    owner_principal_id: Optional[int] = Field(default=None, sa_column=Column(Integer))
    # Consumer tenant scope, denormalized from the API key used for the request.
    # This can differ from owner_principal_id when one Org calls another Org's
    # shared models. FK-less — see ``user_id``.
    consumer_principal_id: Optional[int] = Field(
        default=None, sa_column=Column(Integer)
    )
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
