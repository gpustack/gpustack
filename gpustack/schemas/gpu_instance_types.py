from typing import Optional, List

from pydantic import ConfigDict, BaseModel

from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    ItemList,
)


class GPUAggregatedInstanceTypeSpec(BaseModel):
    """
    Represents the specification of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    group: str
    """
    Indicates the group of the GPU instance type.
    """

    acceleratable: bool = False
    """
    Indicates whether the GPU instance type is acceleratable.
    """

    manufacturer: str = "cpu"
    """
    The name of the GPU instance type manufacturer, e.g. "amd", "nvidia", "intel".
    """

    product: Optional[str] = None
    """
    The name of the GPU instance type product, e.g. "A100", "V100", "T4".
    """

    memory: Optional[str] = None
    """
    The VRAM size of the GPU instance type, e.g. "65535Mi".
    """

    family: Optional[str] = None
    """
    The family of the GPU instance type, e.g. "Ampere", "Volta", "Turing".
    """

    compute_capability: Optional[str] = None
    """
    The compute capability of the GPU instance type, e.g. "8.0", "7.0".
    """

    sliced: Optional[str] = None
    """
    Indicates whether the GPU instance type is sliced.
    When it is blank, that means the GPU instance type is not sliced.
    """


class GPUAggregatedInstanceTypeResource(BaseModel):
    """
    Represents the resource information of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: str
    """
    The maximum resource that can be requested once, e.g. "4".
    """

    remaining: str
    """
    The remaining resource that can be requested, e.g. "16".
    """

    capacity: str
    """
    The total capacity of the resource, e.g. "20".
    """


class GPUAggregatedInstanceTypeOnceMaxRequestCandidate(BaseModel):
    """
    Represents the candidate GPU instance type for once max request accelerator tier.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cluster: str
    """
    The cluster where the GPU instance type is available, e.g. "cluster-1".
    """

    name: str
    """
    The name of the GPU instance type, e.g. "nvidia-a100-40gb-sxm4".
    """

    accelerator: GPUAggregatedInstanceTypeResource
    """
    The accelerator resource information of the GPU instance type.
    """

    cpu: GPUAggregatedInstanceTypeResource
    """
    The CPU resource information of the GPU instance type.
    """

    ram: GPUAggregatedInstanceTypeResource
    """
    The RAM resource information of the GPU instance type.
    """

    local_storage: GPUAggregatedInstanceTypeResource
    """
    The local storage resource information of the GPU instance type.
    """


class GPUAggregatedInstanceTypeOnceMaxRequestTier(BaseModel):
    """
    Represents the accelerator tier for selecting GPU instance types based on once max request.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: str
    """
    Maximum resource that can be requested in this tier, e.g. "4".
    """

    candidates: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestCandidate]] = None
    """
    Candidate GPU instance types for this once max request tier.
    """


class GPUAggregatedInstanceTypeRemainingResource(BaseModel):
    """
    Represents the remaining resources of a GPU instance type that can be requested.

    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    accelerator: Optional[str] = None
    """
    The remaining accelerator resource that can be requested, e.g. "16".
    """

    cpu: str
    """
    The remaining CPU resource that can be requested, e.g. "64".
    """

    ram: str
    """
    The remaining RAM resource that can be requested, e.g. "256Gi".
    """

    local_storage: str
    """
    The remaining local storage resource that can be requested, e.g. "1Ti".
    """


class GPUAggregatedInstanceTypeStatus(BaseModel):
    """
    Represents the status of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    remaining: GPUAggregatedInstanceTypeRemainingResource
    """
    The remaining resources of the GPU instance type.
    """

    accelerator_tiers: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestTier]] = (
        None
    )
    """
    The accelerator tiers for selecting GPU instance types.
    """


class GPUAggregatedInstanceTypeBase(BaseModel):
    """
    Base model for GPU instance type, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Aggregated name of the GPU instance type.
    """

    spec: GPUAggregatedInstanceTypeSpec
    """
    Specification of the GPU instance type.
    """

    status: GPUAggregatedInstanceTypeStatus
    """
    Status of the GPU instance type.
    """


class GPUAggregatedInstanceTypePublic(GPUAggregatedInstanceTypeBase):
    """
    Public representation of a GPU instance type,
    containing fields that are exposed to users.
    """

    pass


GPUAggregatedInstanceTypesPublic = ItemList[GPUAggregatedInstanceTypePublic]
