from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional
from sqlmodel import JSON, BigInteger, Column, Field, Relationship, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.links import (
    ModelInstanceDraftModelFileLink,
    ModelInstanceModelFileLink,
)
from gpustack.schemas.models import ModelSource, ModelInstance


class ModelFileStateEnum(str, Enum):
    ERROR = "error"
    DOWNLOADING = "downloading"
    READY = "ready"


class ModelFileBase(SQLModel, ModelSource):
    local_dir: Optional[str] = None
    worker_id: Optional[int] = None
    cleanup_on_delete: Optional[bool] = None

    size: Optional[int] = Field(sa_column=Column(BigInteger), default=None)
    download_progress: Optional[float] = None
    resolved_paths: List[str] = Field(sa_column=Column(JSON), default=[])
    state: ModelFileStateEnum = ModelFileStateEnum.DOWNLOADING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class ModelFile(ModelFileBase, BaseModelMixin, table=True):
    __tablename__ = 'model_files'
    id: Optional[int] = Field(default=None, primary_key=True)

    # Unique index of the model source
    source_index: Optional[str] = Field(index=True, unique=True, default=None)

    instances: list[ModelInstance] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"},
        back_populates="model_files",
        link_model=ModelInstanceModelFileLink,
    )

    draft_instances: list[ModelInstance] = Relationship(
        back_populates="draft_model_files",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "selectin"},
    )


class ModelFileListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "source",
        "worker_id",
        "state",
        "resolved_paths",
        "created_at",
        "updated_at",
    ]


class ModelFileCreate(ModelFileBase):
    pass


class ModelFileUpdate(ModelFileBase):
    pass


class ModelFilePublic(
    ModelFileBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelFilesPublic = PaginatedList[ModelFilePublic]
