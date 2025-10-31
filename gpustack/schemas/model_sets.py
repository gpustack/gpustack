from datetime import date
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.models import (
    ModelSource,
    ModelSpecBase,
)


class ModelSpec(ModelSpecBase):
    name: Optional[str] = None
    quantization: Optional[str] = None
    mode: Optional[str] = "standard"


class ModelSetBase(BaseModel):
    name: str
    id: Optional[int] = None
    description: Optional[str] = None
    order: Optional[int] = None
    home: Optional[str] = None
    icon: Optional[str] = None
    categories: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    size: Optional[float] = None
    activated_size: Optional[float] = None
    licenses: Optional[List[str]] = None
    release_date: Optional[date] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelSetPublic(ModelSetBase):
    pass


class ModelSet(ModelSetBase):
    specs: List[ModelSpec]


class DraftModel(ModelSource):
    name: str
    algorithm: str
    description: Optional[str] = None


class Catalog(BaseModel):
    model_sets: List[ModelSet]
    draft_models: List[DraftModel]
