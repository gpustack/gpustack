from datetime import date
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, field_validator

from gpustack.schemas.models import (
    ModelSource,
    ModelSpecBase,
)


class GPUFilters(BaseModel):
    vendor: Optional[Union[str, List[str]]] = None
    """List of GPU vendors, e.g., ['nvidia', 'amd'] or 'nvidia'."""
    compute_capability: Optional[str] = None
    """Compute capability filter expressed using pip-style version specifiers. E.g., '>=7.0,<8.0'."""
    vendor_variant: Optional[Union[str, List[str]]] = None
    """List of GPU vendor variants. For example, ['910b', '310p'] or '910b' for Ascend NPUs."""

    @field_validator("vendor", "vendor_variant", mode="before")
    def normalize_str_or_list_fields(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class ModelSpec(ModelSpecBase):
    name: Optional[str] = None
    quantization: Optional[str] = None
    mode: Optional[str] = "standard"
    gpu_filters: Optional[GPUFilters] = None


class SizeUnit(str, Enum):
    MILLION = "M"
    BILLION = "B"
    TRILLION = "T"


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
    size_unit: Optional[SizeUnit] = None
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
