from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class GPUFilters(BaseModel):
    """Which accelerators a declaration applies to.

    Shared by two catalogs that both need "this entry only fits these cards":
    a model spec in ``model-catalog.yaml`` and a PD recipe in
    ``pd-modes.yaml``. Lives in its own module because ``pd_modes`` is a leaf
    schema loaded at start-up -- importing it from ``model_sets`` would drag
    ``schemas.models`` in behind it.

    The *matching* rule is not shared. A model catalog picks one spec out of
    several and a miss falls through to another spec, so "any card in the
    cluster matches" is right there. A PD recipe has no fallback -- a miss
    means no recipe at all -- and a PD group cannot straddle two vendors, so it
    has to be resolved per vendor partition instead. See
    ``routes/model_sets.py:filter_specs_by_gpu`` for the former.
    """

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
