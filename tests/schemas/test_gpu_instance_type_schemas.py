"""Field-shape / passthrough-fidelity tests for the GPU instance-type schemas.

These lock the Pydantic models to the worker-gateway JSON contract: the REST
routes feed raw gateway dicts straight into these models, so an alias that does
not match the gateway key silently drops data. Each test round-trips a
gateway-shaped payload through ``model_validate`` → ``model_dump(by_alias=True,
exclude_none=True)`` and asserts it comes back unchanged.

The public model is also validated straight off a ``GPUInstanceType`` record
row (``ActiveRecordMixin._convert_to_public_class`` on the watch path), not only
from a gateway dict, so the row → public projection is locked here as well.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.common import Pagination
from gpustack.schemas.gpu_instance_types import (
    GPUAggregatedInstanceTypeOverviewResource,
    GPUAggregatedInstanceTypePublic,
    GPUInstanceType,
    GPUInstanceTypeAcceleratorProfileCount,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeDetail,
    GPUInstanceTypeListParams,
    GPUInstanceTypePartitionedResource,
    GPUInstanceTypePublic,
    GPUInstanceTypeSpec,
    GPUInstanceTypeStatus,
    GPUInstanceTypeStatusPublic,
    GPUInstanceTypesPublic,
    GPUInstanceTypeUnitResources,
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


# --- record row -> public projection (the watch path) ----------------------- #

# The fixed definition the frozen digests below are measured over. Table models
# skip validation on init, so a row is built directly.
FIXED_SPEC = dict(
    display_name="H100 Pool",
    accelerator_group="nvidia-h100-80gb-hbm3",
    acceleratable=True,
    os="linux",
    arch="amd64",
    unit_resources=GPUInstanceTypeUnitResources(cpu="8000m", ram="65536Mi"),
    local_storage="100Gi",
)


def _row(**overrides) -> GPUInstanceType:
    row = dict(
        id=7,
        cluster_id=3,
        name="nvidia-h100-80gb-hbm3",
        spec=GPUInstanceTypeSpec(**FIXED_SPEC),
        status=GPUInstanceTypeStatusPublic(
            detail=GPUInstanceTypeDetail(manufacturer="nvidia", memory="81920Mi"),
            phase="Active",
            phase_message="ClusterQueue is admitting workloads",
        ),
        derived_from_node=True,
        snapshot="sha1:unused",
        created_at=datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 18, 9, 5, tzinfo=timezone.utc),
    )
    row.update(overrides)
    return GPUInstanceType(**row)


def test_public_validates_from_a_record_row():
    # ``GPUInstanceType.streaming`` hands the ORM row to
    # ``_convert_to_public_class``, which calls ``model_validate`` on it, so the
    # narrow persisted status has to widen into the single public status type.
    model = GPUInstanceTypePublic.model_validate(_row())

    assert (model.id, model.cluster_id) == (7, 3)
    assert model.name == "nvidia-h100-80gb-hbm3"
    assert model.created_at == datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    assert model.updated_at == datetime(2026, 8, 18, 9, 5, tzinfo=timezone.utc)
    assert model.derived_from_node is True
    assert model.spec.accelerator_group == "nvidia-h100-80gb-hbm3"

    assert isinstance(model.status, GPUInstanceTypeStatus)
    assert model.status.detail.manufacturer == "nvidia"
    assert model.status.detail.memory == "81920Mi"
    assert model.status.phase == "Active"
    assert model.status.phase_message == "ClusterQueue is admitting workloads"
    # The resource ledger is not persisted, so this endpoint reports it as
    # absent. Consumers must read that as "not provided here", never as zero.
    assert model.status.accelerator is None
    assert model.status.accelerator_partitioned is None
    assert model.status.cpu is None

    # The wire shape the list route serves (response_model_exclude_none=True,
    # aliases on). ``clusterId`` is camelCased while the timestamps and ``id``
    # are not: ``pydantic_camel_case_generator`` exempts the record-keeping
    # fields. Same keys as GPUInstancePublic, the sibling table-backed list.
    dumped = model.model_dump(by_alias=True, exclude_none=True)
    assert set(dumped) == {
        "id",
        "clusterId",
        "created_at",
        "updated_at",
        "name",
        "spec",
        "status",
        "derivedFromNode",
    }
    assert dumped["status"]["phaseMessage"] == "ClusterQueue is admitting workloads"


def test_public_validates_a_row_without_status():
    # ``status`` stays NULL until the operator first reports one, and a DELETED
    # bus event replays the row as it stands, so None must validate as None.
    model = GPUInstanceTypePublic.model_validate(_row(status=None))
    assert model.status is None


def test_public_still_builds_from_a_raw_cr_dict():
    # The write routes return the cluster's live CR, so the same model is built
    # by keyword from raw gateway JSON — carrying the full status, and without a
    # row id / cluster_id / timestamps.
    model = GPUInstanceTypePublic(
        name="nvidia-h100-80gb-hbm3",
        spec={"acceleratorGroup": "nvidia-h100-80gb-hbm3", "acceleratable": True},
        status={
            "phase": "Active",
            "detail": {"manufacturer": "nvidia"},
            "accelerator": {"onceMaxRequest": "1", "remaining": "2", "capacity": "4"},
            "acceleratorPartitioned": PARTITIONED_RESOURCE,
            "cpu": {"onceMaxRequest": "4", "remaining": "8", "capacity": "16"},
        },
        derived_from_node=True,
    )

    assert model.status.accelerator.remaining == "2"
    assert model.status.accelerator_partitioned.remaining_profiles[0].count == 6
    assert model.status.cpu.capacity == "16"
    assert (model.id, model.cluster_id) == (None, None)
    assert (model.created_at, model.updated_at) == (None, None)


def test_status_public_persists_only_detail_and_the_phase_pair():
    # The persisted subset is a hard boundary, not an accident of what was needed
    # first. Every row write publishes on the internal bus, and the list route
    # puts one SSE stream per open page on the other end, so a field's cost is
    # its change frequency. ``detail`` / ``phase`` / ``phase_message`` move on
    # real state transitions; the resource ledger (accelerator*, cpu, the
    # partitioned profile ledger) is recomputed on every workload movement in the
    # cluster. Do NOT widen this set to make a new field available — check its
    # churn first.
    aliases = {
        f.alias or n for n, f in GPUInstanceTypeStatusPublic.model_fields.items()
    }
    assert aliases == {"detail", "phase", "phaseMessage"}


def test_instance_types_public_requires_pagination():
    # The list route serves a paginated read now, so ``pagination`` is part of
    # the response contract rather than an optional extra.
    with pytest.raises(ValidationError):
        GPUInstanceTypesPublic(items=[])

    listed = GPUInstanceTypesPublic(
        items=[GPUInstanceTypePublic.model_validate(_row())],
        pagination=Pagination(page=1, perPage=100, total=1, totalPage=1),
    )
    assert listed.pagination.total == 1
    assert listed.items[0].cluster_id == 3


def test_list_params_rejects_the_status_phase_field():
    # ``status.phase`` lives inside the ``status`` JSON column: it is not a
    # sortable SQL column, and name search is the page's discovery affordance.
    with pytest.raises(InvalidException, match="not sortable"):
        GPUInstanceTypeListParams(sort_by="status.phase")


def test_list_params_rejects_an_unknown_sort_field():
    with pytest.raises(InvalidException, match="not sortable"):
        GPUInstanceTypeListParams(sort_by="bogus")


def test_list_params_accepts_the_four_sortable_fields():
    params = GPUInstanceTypeListParams(
        sort_by="name,-cluster_id,created_at,-updated_at"
    )
    assert params.order_by == [
        ("name", "asc"),
        ("cluster_id", "desc"),
        ("created_at", "asc"),
        ("updated_at", "desc"),
    ]


# --- snapshot byte-stability ------------------------------------------------ #

# Measured against the pre-change code for FIXED_SPEC. ``snapshot`` doubles as
# ``metered_usage.sku`` — the metering / pricing reference key — so these bytes
# are a published contract, not an implementation detail. A diff here means
# existing metering history no longer resolves to its type.
FROZEN_SNAPSHOT = "sha1:73c7923909795e87b00c9930536065a8c61c0170"
FROZEN_DEFINITION_SNAPSHOT = "sha1:394ee7f2c23d00e78547391c6142974ec5a4399f"


def test_snapshot_digests_are_frozen():
    row = _row()
    assert row.compute_snapshot() == FROZEN_SNAPSHOT
    assert row.compute_definition_snapshot() == FROZEN_DEFINITION_SNAPSHOT


def test_persisted_status_and_derived_marker_stay_out_of_the_digests():
    # Identity is (cluster_id, name, spec) only. The row built by ``_row`` above
    # already carries a phase-bearing status and derived_from_node=True, so if
    # either had been folded into the hash these two would diverge from the bare
    # definition's — and every metered row written before the change would stop
    # resolving.
    bare = _row(status=None, derived_from_node=False)
    assert bare.compute_snapshot() == FROZEN_SNAPSHOT
    assert bare.compute_definition_snapshot() == FROZEN_DEFINITION_SNAPSHOT
