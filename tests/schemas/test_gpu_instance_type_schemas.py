"""Field-shape / passthrough-fidelity tests for the GPU instance-type schemas.

These lock the Pydantic models to the worker-gateway JSON contract: the REST
routes feed raw gateway dicts straight into these models, so an alias that does
not match the gateway key silently drops data. Each test round-trips a
gateway-shaped payload through ``model_validate`` → ``model_dump(by_alias=True,
exclude_none=True)`` and asserts it comes back unchanged.
"""

from gpustack.schemas.gpu_instance_types import (
    GPUAggregatedInstanceTypePublic,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeDetail,
    GPUInstanceTypePublic,
)

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
            "onceMaxRequest": {"accelerator": "4"},
            "remaining": {"accelerator": "16"},
            "tiers": [
                {
                    "onceMaxRequest": {"accelerator": "4"},
                    "acceleratorSlicedDetail": sliced,
                    "candidates": [
                        {
                            "cluster": "cluster-1",
                            "name": "nvidia-a100",
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
    assert status["tiers"][0]["acceleratorSlicedDetail"] == sliced
    assert status["tiers"][0]["candidates"][0]["acceleratorSlicedDetail"] == sliced
