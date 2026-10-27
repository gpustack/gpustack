from typing import Optional

from pydantic import ConfigDict, BaseModel

from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    ItemList,
)


class GPUInstanceTypeFlavorSpec(BaseModel):
    """
    Represents the specification of a GPU instance type flavor.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    accelerator_group: Optional[str] = None
    """
    The accelerator group (the acceleratable key) of an accelerated pool,
    e.g. "nvidia-a10g"; empty for a generic pool.
    """

    general_group: Optional[str] = None
    """
    The general (CPU) group of the pool: the real CPU key when
    instance-type-aware-cpu-manufacturer is on, or the "generic" sentinel for a
    collapsed (unaware) generic pool; empty for an accelerated pool when
    awareness is off.
    """

    acceleratable: bool = False
    """
    Indicates whether the pool represents accelerated hardware; a generic
    (CPU-only) pool is false. It delimits generic from accelerated flavors.
    """

    manufacturer: Optional[str] = None
    """
    The device (or CPU) manufacturer, e.g. "nvidia", "generic".
    """

    product: Optional[str] = None
    """
    The product name, e.g. "NVIDIA A10G"; empty for a generic pool.
    """

    family: Optional[str] = None
    """
    The product family, e.g. "ampere"; empty for a generic pool.
    """

    memory: Optional[str] = None
    """
    The per-card VRAM, e.g. "24576Mi"; empty for a generic pool.
    """

    cores: Optional[str] = None
    """
    The per-card accelerator core count, e.g. "9216"; empty for a generic pool.
    """


class GPUInstanceTypeFlavorBase(BaseModel):
    """
    Base model for GPU instance type flavor, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPU instance type flavor.
    """

    spec: GPUInstanceTypeFlavorSpec
    """
    Specification of the GPU instance type flavor.
    """


class GPUInstanceTypeFlavorPublic(GPUInstanceTypeFlavorBase):
    """
    Public representation of a GPU instance type flavor,
    containing fields that are exposed to users.
    """

    pass


GPUInstanceTypeFlavorsPublic = ItemList[GPUInstanceTypeFlavorPublic]


class GPUAggregatedInstanceTypeFlavorSpec(GPUInstanceTypeFlavorSpec):
    """
    Represents the specification of an aggregated GPU instance type flavor,
    which may include additional fields for aggregation purposes.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    clusters: Optional[list[str]] = None
    """
    List of clusters where this aggregated GPU instance type flavor is available.
    """


class GPUAggregatedInstanceTypeFlavorBase(GPUInstanceTypeFlavorBase):
    """
    Aggregated representation of a GPU instance type flavor,
    which may include additional fields for aggregation purposes.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    spec: GPUAggregatedInstanceTypeFlavorSpec
    """
    Specification of the aggregated GPU instance type flavor.
    """


class GPUAggregatedInstanceTypeFlavorPublic(GPUAggregatedInstanceTypeFlavorBase):
    """
    Public representation of an aggregated GPU instance type flavor,
    containing fields that are exposed to users.
    """

    pass


GPUAggregatedInstanceTypeFlavorsPublic = ItemList[GPUAggregatedInstanceTypeFlavorPublic]
