"""Field-shape / passthrough-fidelity tests for the GPU instance-type schemas.

These lock the Pydantic models to the worker-gateway JSON contract: the REST
routes feed raw gateway dicts straight into these models, so an alias that does
not match the gateway key silently drops data. Each test round-trips a
gateway-shaped payload through ``model_validate`` → ``model_dump(by_alias=True,
exclude_none=True)`` and asserts it comes back unchanged.
"""

import pytest
from pydantic import ValidationError

from gpustack.schemas.gpu_instance_types import (
    GPUAggregatedInstanceTypeOverviewResource,
    GPUAggregatedInstanceTypePublic,
    GPUInstanceTypeAcceleratorProfileCount,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeDetail,
    GPUInstanceTypePartitionedResource,
    GPUInstanceTypePublic,
    GPUInstanceTypeStatus,
)

# The reference pool: one H100 80GB HBM3 card in MIG mode holding a single 1g.10gb.
#
# The zero entries carry no "count" key at all — the operator's Count is an int32 with
# json:"count,omitempty", so Go elides it at zero. "Offered but currently full" therefore
# arrives as {"name": ...} alone and has to survive as such, because that is what keeps it
# distinguishable from "not offered", which is an absent entry.
PARTITIONED_RESOURCE = {
    "onceMaxRequest": "1",
    "remaining": "6",
    "capacity": "7",
    "allocatedProfiles": [{"name": "1g.10gb", "count": 1}],
    "remainingProfiles": [
        {"name": "1g.10gb", "count": 6},
        {"name": "1g.20gb", "count": 3},
        {"name": "2g.20gb", "count": 2},
        {"name": "3g.40gb", "count": 1},
        {"name": "4g.40gb"},
        {"name": "7g.80gb"},
    ],
}

# At tier and item level the partition dimension of the overview resource is a list, not a
# scalar. In remaining it is the Σ of every Active member's obtainable profiles — one member
# here, so the pool's own ledger. In onceMaxRequest every obtainable profile is capped at one:
# a partition request is always a single instance on a single card.
PARTITION_REMAINING = PARTITIONED_RESOURCE["remainingProfiles"]
PARTITION_ONCE_MAX_REQUEST = [
    {"name": "1g.10gb", "count": 1},
    {"name": "1g.20gb", "count": 1},
    {"name": "2g.20gb", "count": 1},
    {"name": "3g.40gb", "count": 1},
    {"name": "4g.40gb"},
    {"name": "7g.80gb"},
]

# --- sliced-detail family (aligns devices.go AcceleratorSlicedDetail) ------- #


def test_sliced_detail_round_trips_unchanged():
    payload = {
        "logical": {"coresPercentageOvercommit": True, "count": 8},
        "physical": {
            "profiles": [
                {"name": "1g.5gb", "count": 7},
                {"name": "2g.10gb", "count": 3},
            ],
            "count": 7,
        },
    }
    model = GPUInstanceTypeAcceleratorSlicedDetail.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload


def test_sliced_detail_logical_only_round_trips():
    payload = {"logical": {"coresPercentageOvercommit": False, "count": 4}}
    model = GPUInstanceTypeAcceleratorSlicedDetail.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload


# --- detail descriptor (aligns instance_type.go InstanceTypeDetail) --------- #


def test_detail_emits_flat_key_set():
    aliases = {f.alias or n for n, f in GPUInstanceTypeDetail.model_fields.items()}
    assert aliases == {
        "manufacturer",
        "product",
        "family",
        "physicalCores",
        "threadsPerPhysicalCore",
        "logicalCores",
        "stepping",
        "clockSpeed",
        "maxClockSpeed",
        "cacheLine",
        "cache",
        "memory",
        "cores",
        "computeCapability",
        "slicedDetail",
        "cpu",
    }


def test_detail_round_trips_unchanged():
    payload = {
        "manufacturer": "nvidia",
        "product": "NVIDIA A100",
        "family": "ampere",
        "physicalCores": "64",
        "cache": {"l1i": "64", "l2": "512"},
        "memory": "40960Mi",
        "cores": "6912",
        "computeCapability": "8.0",
        "slicedDetail": {
            "logical": {"coresPercentageOvercommit": True, "count": 8},
            "physical": {"profiles": [{"name": "1g.5gb", "count": 7}], "count": 7},
        },
        "cpu": {"manufacturer": "amd", "product": "EPYC 7742", "physicalCores": "64"},
    }
    model = GPUInstanceTypeDetail.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload


# --- status detail passthrough (the REST models are the gateway contract) --- #


def test_public_status_detail_survives_passthrough():
    detail = {
        "manufacturer": "nvidia",
        "physicalCores": "64",
        "memory": "40960Mi",
        "slicedDetail": {
            "physical": {"profiles": [{"name": "1g.5gb", "count": 7}], "count": 7}
        },
    }
    payload = {
        "name": "nvidia-a100",
        "spec": {"acceleratorGroup": "nvidia-a100", "acceleratable": True},
        "status": {"phase": "Active", "detail": detail},
    }
    model = GPUInstanceTypePublic.model_validate(payload)
    dumped = model.model_dump(by_alias=True, exclude_none=True)
    assert dumped["status"]["detail"] == detail


def test_status_resource_views_round_trip():
    payload = {
        "phase": "Active",
        "accelerator": {"onceMaxRequest": "1", "remaining": "2", "capacity": "4"},
        "acceleratorShared": {
            "onceMaxRequest": "10",
            "remaining": "20",
            "capacity": "40",
        },
        "acceleratorSliced": {
            "onceMaxRequest": "100",
            "remaining": "200",
            "capacity": "400",
        },
        "acceleratorPartitioned": PARTITIONED_RESOURCE,
        "cpu": {"onceMaxRequest": "4", "remaining": "8", "capacity": "16"},
    }
    model = GPUInstanceTypeStatus.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload


def test_partitioned_resource_emits_ledger_beside_the_scalars():
    aliases = {
        f.alias or n for n, f in GPUInstanceTypePartitionedResource.model_fields.items()
    }
    assert aliases == {
        "onceMaxRequest",
        "remaining",
        "capacity",
        "allocatedProfiles",
        "remainingProfiles",
    }


def test_partitioned_resource_without_allocated_profiles_round_trips():
    # An empty card: the operator omits allocatedProfiles entirely (json omitempty on a
    # nil slice) rather than sending an empty list, so absence must not become [].
    payload = {
        "onceMaxRequest": "1",
        "remaining": "7",
        "capacity": "7",
        "remainingProfiles": [{"name": "1g.10gb", "count": 7}],
    }
    model = GPUInstanceTypePartitionedResource.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload
    assert model.allocated_profiles is None


def test_aggregated_overview_resource_carries_the_partition_ledger():
    # The partition dimension alone is a list where every other dimension is a scalar
    # quantity, because it has no honest scalar: the profiles of one card compete for the
    # same physical slices, so a total over them is not a capacity and a best case over them
    # is not a total.
    payload = {
        "accelerator": "4",
        "acceleratorShared": "40",
        "acceleratorSliced": "400",
        "acceleratorPartitioned": PARTITION_REMAINING,
        "cpu": "16",
    }
    model = GPUAggregatedInstanceTypeOverviewResource.model_validate(payload)
    assert model.model_dump(by_alias=True, exclude_none=True) == payload


def test_aggregated_public_sliced_detail_survives_passthrough():
    sliced = {
        "logical": {"coresPercentageOvercommit": True, "count": 8},
        "physical": {"profiles": [{"name": "1g.5gb", "count": 7}], "count": 7},
    }
    payload = {
        "name": "nvidia-a100",
        "spec": {"acceleratorGroup": "nvidia-a100", "acceleratable": True},
        "status": {
            "detail": {"manufacturer": "nvidia", "slicedDetail": sliced},
            "onceMaxRequest": {
                "accelerator": "4",
                "acceleratorPartitioned": PARTITION_ONCE_MAX_REQUEST,
            },
            "remaining": {
                "accelerator": "16",
                "acceleratorPartitioned": PARTITION_REMAINING,
            },
            "tiers": [
                {
                    "onceMaxRequest": {
                        "accelerator": "4",
                        "acceleratorPartitioned": PARTITION_ONCE_MAX_REQUEST,
                    },
                    "acceleratorSlicedDetail": sliced,
                    "candidates": [
                        {
                            "cluster": "cluster-1",
                            "name": "nvidia-a100",
                            "acceleratorPartitioned": PARTITIONED_RESOURCE,
                            "acceleratorSlicedDetail": sliced,
                        }
                    ],
                }
            ],
        },
    }
    model = GPUAggregatedInstanceTypePublic.model_validate(payload)
    status = model.model_dump(by_alias=True, exclude_none=True)["status"]
    assert status["detail"]["slicedDetail"] == sliced
    assert status["onceMaxRequest"]["acceleratorPartitioned"] == (
        PARTITION_ONCE_MAX_REQUEST
    )
    assert status["remaining"]["acceleratorPartitioned"] == PARTITION_REMAINING
    tier = status["tiers"][0]
    assert tier["acceleratorSlicedDetail"] == sliced
    assert tier["onceMaxRequest"]["acceleratorPartitioned"] == (
        PARTITION_ONCE_MAX_REQUEST
    )
    candidate = tier["candidates"][0]
    assert candidate["acceleratorSlicedDetail"] == sliced
    assert candidate["acceleratorPartitioned"] == PARTITIONED_RESOURCE


def test_profile_count_requires_a_name_but_not_a_count():
    # The two fields are deliberately asymmetric, mirroring the operator's tags:
    # Name is json:"name" with no omitempty, Count is json:"count,omitempty". So a
    # bare {"name": ...} is a real profile at zero and must validate, while an entry
    # without a name is nothing the ledger can be keyed by and must not.
    at_zero = GPUInstanceTypeAcceleratorProfileCount.model_validate({"name": "4g.40gb"})
    assert at_zero.name == "4g.40gb"
    assert at_zero.count is None

    with pytest.raises(ValidationError):
        GPUInstanceTypeAcceleratorProfileCount.model_validate({"count": 3})
