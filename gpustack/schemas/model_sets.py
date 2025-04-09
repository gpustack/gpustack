from datetime import date
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.models import (
    ModelSpecBase,
)


class ModelTemplate(ModelSpecBase):
    name: Optional[str] = None
    quantizations: Optional[List[str]] = None
    sizes: Optional[List[float]] = None


class ModelSpec(ModelSpecBase):
    name: Optional[str] = None
    quantization: Optional[str] = None
    size: Optional[float] = None


class ModelSetBase(BaseModel):
    name: str
    id: Optional[int] = None
    description: Optional[str] = None
    order: Optional[int] = None
    home: Optional[str] = None
    icon: Optional[str] = None
    categories: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    sizes: Optional[List[float]] = None
    licenses: Optional[List[str]] = None
    release_date: Optional[date] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelSetPublic(ModelSetBase):
    pass


class ModelSet(ModelSetBase):
    quantizations: Optional[List[str]] = None

    templates: List[ModelTemplate]
