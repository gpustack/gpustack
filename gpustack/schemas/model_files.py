from datetime import datetime
from enum import Enum
from typing import List, Optional
from sqlmodel import JSON, BigInteger, Column, Field, Relationship, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.models import ModelSource, ModelInstance


RESET_DOWNLOAD_MESSAGE = "Retrying download"


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
