from datetime import datetime
from enum import Enum
from typing import List, Optional
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.models import ModelSource, ModelInstance


class ModelFileStateEnum(str, Enum):
    ERROR = "error"
    DOWNLOADING = "downloading"
    READY = "ready"


class ModelFileBase(SQLModel, ModelSource):
    local_dir: Optional[str] = None
    worker_id: Optional[int] = None

    size: Optional[int] = None
    download_progress: Optional[float] = None
    resolved_paths: List[str] = Field(sa_column=Column(JSON), default=[])
    state: ModelFileStateEnum = ModelFileStateEnum.DOWNLOADING
    state_message: Optional[str] = None


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
