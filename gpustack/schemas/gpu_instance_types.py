from typing import Optional, List

from pydantic import ConfigDict, BaseModel

from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    ItemList,
)


class GPUAggregatedInstanceTypeUnitResources(BaseModel):
    """
    Represents the unit resources of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cpu: Optional[str] = None
    """
    The per-device unit CPU resources of the GPU instance type, ends with "m".
    """

    ram: Optional[str] = None
    """
    The per-device RAM resources of the GPU instance type, ends with "Mi".
    """


class GPUAggregatedInstanceTypeCPUCache(BaseModel):
    """
    Represents the cache information of the CPU of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    l1i: Optional[str] = None
    """
    The L1 instruction cache size in bytes of the CPU, e.g. "64".
    """

    l1d: Optional[str] = None
    """
    The L1 data cache size in bytes of the CPU, e.g. "64".
    """

    l2: Optional[str] = None
    """
    The L2 cache size in bytes of the CPU, e.g. "256", "512".
    """

    l3: Optional[str] = None
    """
    The L3 cache size in bytes of the CPU, e.g. "8192", "16384".
    """


class GPUAggregatedInstanceTypeCPU(BaseModel):
    """
    Represents the CPU resource information of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    physical_cores: Optional[str] = None
    """
    The number of physical cores of the CPU, e.g. "4", "8".
    """

    threads_per_physical_core: Optional[str] = None
    """
    The number of threads per physical core of the CPU, e.g. "2", "4".
    """

    logical_cores: Optional[str] = None
    """
    The number of logical cores of the CPU, e.g. "8", "16".
    """

    stepping: Optional[str] = None
    """
    The stepping of the CPU, e.g. "0", "1".
    """

    clock_speed: Optional[str] = None
    """
    The speed in Hz of the CPU, e.g. "2000".
    """

    max_clock_speed: Optional[str] = None
    """
    The maximum speed in Hz of the CPU, e.g. "3000".
    """

    cache_line: Optional[str] = None
    """
    The cache line size in bytes of the CPU, e.g. "64", "128".
    """

    cache: Optional[GPUAggregatedInstanceTypeCPUCache] = None
    """
    The cache information of the CPU.
    """


class GPUAggregatedInstanceTypeAcceleratorCPU(GPUAggregatedInstanceTypeCPU):
    """
    Represents the CPU information of the accelerator of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    manufacturer: Optional[str] = None
    """
    The name of the CPU manufacturer, e.g. "amd", "intel".
    """

    product: Optional[str] = None
    """
    The name of the CPU product.
    """

    family: Optional[str] = None
    """
    The family of the CPU.
    """


class GPUAggregatedInstanceTypeAccelerator(BaseModel):
    """
    Represents the accelerator resource information of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    memory: Optional[str] = None
    """
    The VRAM size of the accelerator, e.g. "65535Mi".
    """

    cores: Optional[str] = None
    """
    The number of cores of the accelerator, e.g. "128", "256".
    """

    compute_capability: Optional[str] = None
    """
    The compute capability of the accelerator, e.g. "8.0", "7.0".
    """

    sliced: Optional[str] = None
    """
    Indicates whether the accelerator is sliced.
    When sliced is blank, that means the instance type is not sliced.
    """

    cpu: Optional[GPUAggregatedInstanceTypeAcceleratorCPU] = None
    """
    The CPU information of the accelerator.
    """


class GPUAggregatedInstanceTypeSpec(
    GPUAggregatedInstanceTypeCPU, GPUAggregatedInstanceTypeAccelerator
):
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

    family: Optional[str] = None
    """
    The family of the GPU instance type, e.g. "Ampere", "Volta", "Turing".
    """

    os: Optional[str] = None
    """
    The operating system of the GPU instance type, e.g. "linux", "windows".
    """

    arch: Optional[str] = None
    """
    The architecture of the GPU instance type, e.g. "amd64", "arm64".
    """

    unit_resources: Optional[GPUAggregatedInstanceTypeUnitResources] = None
    """
    The unit resources of the GPU instance type, which represents the resources of one GPU card.
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


class GPUAggregatedInstanceTypeOverviewResource(BaseModel):
    """
    Represents the overview resources of a GPU instance type that can be requested.

    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    accelerator: Optional[str] = None
    """
    The overview accelerator resource that can be requested, e.g. "16".
    """

    cpu: str
    """
    The overview CPU resource that can be requested, e.g. "64".
    """

    ram: str
    """
    The overview RAM resource that can be requested, e.g. "256Gi".
    """

    local_storage: str
    """
    The overview local storage resource that can be requested, e.g. "1Ti".
    """


class GPUAggregatedInstanceTypeOnceMaxRequestTier(BaseModel):
    """
    Represents the accelerator tier for selecting GPU instance types based on once max request.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: GPUAggregatedInstanceTypeOverviewResource
    """
    The once max request overview resources of this accelerator tier.
    """

    remaining: Optional[GPUAggregatedInstanceTypeOverviewResource] = None
    """
    The remaining overview resources of this accelerator tier.
    """

    candidates: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestCandidate]] = None
    """
    Candidate GPU instance types for this once max request tier.
    """


class GPUAggregatedInstanceTypeStatus(BaseModel):
    """
    Represents the status of a GPU instance type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    once_max_request: GPUAggregatedInstanceTypeOverviewResource
    """
    The once max request overview resources of the GPU instance type.
    """

    remaining: Optional[GPUAggregatedInstanceTypeOverviewResource] = None
    """
    The remaining overview resources of the GPU instance type.
    """

    tiers: Optional[List[GPUAggregatedInstanceTypeOnceMaxRequestTier]] = None
    """
    The tiers for selecting GPU instance types.
    If the spec.acceleratable is true, the dimension is accelerator, and the once max request tiers are grouped by accelerator resource.
    If the spec.acceleratable is false, the dimension is cpu, and the once max request tiers are grouped by cpu resource.
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
