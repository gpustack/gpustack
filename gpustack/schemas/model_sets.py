from datetime import date
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gpustack.schemas.models import (
    ModelSource,
    ModelSpecBase,
)
from gpustack.schemas.source import SourceTypeEnum


class GPUFilters(BaseModel):
    # These two default to [] rather than None because a mode="before" validator
    # never runs on a missing field: a None default would serialize as null and
    # only collapse to [] on the next pass, breaking normalize_catalog_yaml's
    # idempotence (every re-save of an unchanged catalog would look changed).
    vendor: Optional[Union[str, List[str]]] = Field(default_factory=list)
    """List of GPU vendors, e.g., ['nvidia', 'amd'] or 'nvidia'."""
    compute_capability: Optional[str] = None
    """Compute capability filter expressed using pip-style version specifiers. E.g., '>=7.0,<8.0'."""
    vendor_variant: Optional[Union[str, List[str]]] = Field(default_factory=list)
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
    # The principal the model would be created under, stamped
    # server-side by the evaluation route from the caller's context
    # (never trusted from the client). Lets compatibility checks see
    # the same Org-scoped backend versions a real deploy would —
    # BackendFrameworkFilter resolves Hybrid backend rows by it, and
    # make_hashable_key adds it to the evaluation cache key explicitly
    # so results don't leak across Orgs. None = Platform-only view
    # (catalog specs, admin in "All" mode).
    #
    # ``exclude=True`` keeps it out of every serialized response: the
    # UI merges the evaluation result's default_spec back into the
    # model-create payload, and ModelCreate.owner_principal_id is a
    # non-nullable int — echoing ``null`` there fails validation with
    # a 422.
    owner_principal_id: Optional[int] = Field(default=None, exclude=True)

    # The cluster the model would be evaluated against, stamped
    # server-side by the evaluation route from the request's top-level
    # cluster_id (never trusted from the client). ``ModelSpecBase``
    # deliberately has no cluster_id, but vGPU/InstanceType checks
    # resolve pools per cluster. Excluded from responses for the same
    # merge-back reason as owner_principal_id.
    cluster_id: Optional[int] = Field(default=None, exclude=True)


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
    # Source-origin display: which source this set came from (None for the
    # packaged built-in catalog, which the UI shows without a badge).
    source_name: Optional[str] = None
    source_type: Optional[SourceTypeEnum] = None


class ModelSet(ModelSetBase):
    specs: List[ModelSpec]


class DraftModel(ModelSource):
    name: str
    algorithm: str
    description: Optional[str] = None


class Catalog(BaseModel):
    model_sets: List[ModelSet]
    draft_models: List[DraftModel]
