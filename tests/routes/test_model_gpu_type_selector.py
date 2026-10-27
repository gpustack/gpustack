"""Validation tests for ``Model.gpu_type_selector`` (sliced-GPU deployments).

``validate_model_in`` is called directly against an in-memory SQLite session
seeded with ``GPUInstanceType`` projection rows — no live cluster, no
Kubernetes client: validation reads only the local DB projection.
"""

from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_model_in
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeAcceleratorSlicedDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetail,
    GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile,
    GPUInstanceTypeDetail,
    GPUInstanceTypeSpec,
    GPUInstanceTypeStatusPublic,
)
from gpustack.schemas.models import (
    GPUSelector,
    GPUTypeSelector,
    ModelCreate,
    SourceEnum,
)

CLUSTER_ID = 1
TYPE_NAME = "a10g-pool"
PROFILE_NAME = "1g.5gb"
CARD_MEMORY = "24576Mi"


@pytest_asyncio.fixture
async def engine():
    e = create_async_engine("sqlite+aiosqlite://")
    async with e.begin() as conn:
        await conn.run_sync(GPUInstanceType.__table__.create)
    yield e
    await e.dispose()


def _status_backfilled() -> GPUInstanceTypeStatusPublic:
    """A status the operator has backfilled: the card's memory is published,
    which is the minimum any mode needs to size its claim. A soft-slicing-only
    pool carries no ``sliced_detail.physical``."""
    return GPUInstanceTypeStatusPublic(detail=GPUInstanceTypeDetail(memory=CARD_MEMORY))


def _status_with_profiles(*profile_names: str) -> GPUInstanceTypeStatusPublic:
    return GPUInstanceTypeStatusPublic(
        detail=GPUInstanceTypeDetail(
            memory=CARD_MEMORY,
            sliced_detail=GPUInstanceTypeAcceleratorSlicedDetail(
                physical=GPUInstanceTypeAcceleratorSlicedPhysicalDetail(
                    profiles=[
                        GPUInstanceTypeAcceleratorSlicedPhysicalDetailProfile(
                            name=name, count=4, memory_mib=5120
                        )
                        for name in profile_names
                    ]
                )
            ),
        )
    )


async def _seed_type(
    engine,
    *,
    cluster_id=CLUSTER_ID,
    name=TYPE_NAME,
    snapshot=None,
    status=None,
    deleted=False,
    acceleratable=True,
):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add(
            GPUInstanceType(
                cluster_id=cluster_id,
                name=name,
                spec=GPUInstanceTypeSpec(acceleratable=acceleratable),
                status=status,
                snapshot=snapshot or f"sha1:{cluster_id}-{name}",
                deleted_at=datetime(2020, 1, 1) if deleted else None,
            )
        )
        await s.commit()


def _model(**kwargs) -> ModelCreate:
    kwargs.setdefault("name", "m1")
    kwargs.setdefault("source", SourceEnum.HUGGING_FACE)
    kwargs.setdefault("huggingface_repo_id", "org/model")
    kwargs.setdefault("cluster_id", CLUSTER_ID)
    return ModelCreate(**kwargs)


async def _validate(engine, model_in: ModelCreate):
    async with AsyncSession(engine, expire_on_commit=False) as s:
        await validate_model_in(s, model_in)


# --- happy paths ----------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sliced_percentage_accepted(engine):
    await _seed_type(engine, status=_status_backfilled())
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=50,
        )
    )
    await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_zero_percentage_is_whole_card_exclusive(engine):
    # Both percentages 0 (or unset) means whole-card exclusive: valid, and the
    # unset cores percentage is normalized to 0.
    await _seed_type(engine, status=_status_backfilled())
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=0
        )
    )
    await _validate(engine, model_in)
    assert model_in.gpu_type_selector.accelerator_sliced_memory_percentage == 0
    assert model_in.gpu_type_selector.accelerator_sliced_cores_percentage == 0


@pytest.mark.asyncio
async def test_partitioned_profile_accepted(engine):
    await _seed_type(engine, status=_status_with_profiles(PROFILE_NAME, "2g.10gb"))
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_partitioned_profile=PROFILE_NAME
        )
    )
    await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_partitioned_profile_with_zero_percentage_accepted(engine):
    # A 0 percentage is "unset" for the profile exclusivity rule.
    await _seed_type(engine, status=_status_with_profiles(PROFILE_NAME))
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=0,
            accelerator_partitioned_profile=PROFILE_NAME,
        )
    )
    await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_gpus_per_replica_one_accepted(engine):
    # gpu_type_selector implies exactly 1 card per worker per replica; an
    # explicit gpus_per_replica=1 is consistent with that.
    await _seed_type(engine, status=_status_backfilled())
    model_in = _model(
        gpu_selector=GPUSelector(gpu_ids=[], gpus_per_replica=1),
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=25
        ),
    )
    await _validate(engine, model_in)


# --- percentage normalization (mirrors the operator webhook) ---------------- #


@pytest.mark.asyncio
async def test_memory_only_defaults_cores_to_100(engine):
    # The operator webhook defaults an unset cores percentage to 100 — it does
    # NOT copy the memory percentage.
    await _seed_type(engine, status=_status_backfilled())
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=40
        )
    )
    await _validate(engine, model_in)
    assert model_in.gpu_type_selector.accelerator_sliced_memory_percentage == 40
    assert model_in.gpu_type_selector.accelerator_sliced_cores_percentage == 100


def test_cores_only_rejected():
    # memory-percentage is required for a sliced request.
    with pytest.raises(ValueError):
        GPUTypeSelector(type=TYPE_NAME, accelerator_sliced_cores_percentage=30)


def test_both_unset_normalized_to_whole_card():
    selector = GPUTypeSelector(type=TYPE_NAME)
    assert selector.accelerator_sliced_memory_percentage == 0
    assert selector.accelerator_sliced_cores_percentage == 0


def test_both_zero_is_whole_card():
    selector = GPUTypeSelector(
        type=TYPE_NAME,
        accelerator_sliced_memory_percentage=0,
        accelerator_sliced_cores_percentage=0,
    )
    assert selector.accelerator_sliced_memory_percentage == 0
    assert selector.accelerator_sliced_cores_percentage == 0


def test_zero_memory_with_nonzero_cores_rejected():
    with pytest.raises(ValueError):
        GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=0,
            accelerator_sliced_cores_percentage=50,
        )


def test_nonzero_memory_with_zero_cores_rejected():
    with pytest.raises(ValueError):
        GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=50,
            accelerator_sliced_cores_percentage=0,
        )


def test_both_percentages_kept_as_given():
    # Compute and memory are independent dimensions.
    selector = GPUTypeSelector(
        type=TYPE_NAME,
        accelerator_sliced_memory_percentage=40,
        accelerator_sliced_cores_percentage=60,
    )
    assert selector.accelerator_sliced_memory_percentage == 40
    assert selector.accelerator_sliced_cores_percentage == 60


# --- rejections ------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_combined_with_gpu_selector_rejected(engine):
    await _seed_type(engine)
    model_in = _model(
        gpu_selector=GPUSelector(gpu_ids=["worker1:cuda:0"]),
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        ),
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_percentage_and_profile_combined_rejected(engine):
    await _seed_type(engine, status=_status_with_profiles(PROFILE_NAME))
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME,
            accelerator_sliced_memory_percentage=50,
            accelerator_partitioned_profile=PROFILE_NAME,
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_cluster_without_instance_types_rejected(engine):
    # No synced InstanceTypes for the cluster (non-k8s cluster / operator
    # absent) → fail closed.
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_soft_deleted_instance_types_do_not_count(engine):
    await _seed_type(engine, deleted=True)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_unknown_instance_type_rejected(engine):
    await _seed_type(engine)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type="no-such-pool", accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_non_accelerator_instance_type_rejected(engine):
    # A cluster publishes a CPU-only pool alongside the GPU ones, and its name
    # says nothing about that. The deploy form hides it, so this guards the API.
    await _seed_type(engine, acceleratable=False)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException) as exc:
        await _validate(engine, model_in)
    assert "not an accelerator type" in str(exc.value.message)


@pytest.mark.asyncio
async def test_type_from_another_cluster_not_visible(engine):
    await _seed_type(engine, cluster_id=999)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_unknown_partitioned_profile_rejected(engine):
    await _seed_type(engine, status=_status_with_profiles(PROFILE_NAME))
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_partitioned_profile="7g.40gb"
        )
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_profile_without_sliced_detail_rejected(engine):
    # The pool is backfilled but offers no partitioning at all → the profile
    # cannot be verified, fail closed.
    await _seed_type(engine, status=_status_backfilled())
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_partitioned_profile=PROFILE_NAME
        )
    )
    with pytest.raises(BadRequestException) as exc:
        await _validate(engine, model_in)
    assert "is not offered by" in str(exc.value.message)


@pytest.mark.asyncio
async def test_type_without_status_detail_rejected(engine):
    # spec is projected as soon as the type appears; status.detail is backfilled
    # afterwards. Accepting the model here would defer the failure to scheduling,
    # where the fit reports the type as unavailable and nothing says why.
    await _seed_type(engine, status=None)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException) as exc:
        await _validate(engine, model_in)
    assert "does not report its accelerator memory yet" in str(exc.value.message)


@pytest.mark.asyncio
async def test_type_without_accelerator_memory_rejected(engine):
    # A detail that exists but carries no memory is equally unschedulable: every
    # mode sizes its claim out of the card's memory.
    await _seed_type(
        engine, status=GPUInstanceTypeStatusPublic(detail=GPUInstanceTypeDetail())
    )
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        )
    )
    with pytest.raises(BadRequestException) as exc:
        await _validate(engine, model_in)
    assert "does not report its accelerator memory yet" in str(exc.value.message)


@pytest.mark.asyncio
async def test_whole_card_without_accelerator_memory_rejected(engine):
    # Whole-card exclusive takes the same path: the card size is what the fit
    # claims, so an unbackfilled type cannot serve it either.
    await _seed_type(engine, status=None)
    model_in = _model(
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=0
        )
    )
    with pytest.raises(BadRequestException) as exc:
        await _validate(engine, model_in)
    assert "does not report its accelerator memory yet" in str(exc.value.message)


@pytest.mark.asyncio
async def test_gpus_per_replica_greater_than_one_rejected(engine):
    await _seed_type(engine)
    model_in = _model(
        gpu_selector=GPUSelector(gpu_ids=[], gpus_per_replica=2),
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        ),
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


@pytest.mark.asyncio
async def test_cluster_id_required(engine):
    # Without a cluster the InstanceType projection cannot be scoped → reject.
    await _seed_type(engine)
    model_in = _model(
        cluster_id=None,
        gpu_type_selector=GPUTypeSelector(
            type=TYPE_NAME, accelerator_sliced_memory_percentage=50
        ),
    )
    with pytest.raises(BadRequestException):
        await _validate(engine, model_in)


# --- schema-level range checks ---------------------------------------------- #


@pytest.mark.parametrize(
    "field",
    [
        "accelerator_sliced_memory_percentage",
        "accelerator_sliced_cores_percentage",
    ],
)
@pytest.mark.parametrize("value", [-1, 101])
def test_percentage_out_of_range_rejected(field, value):
    with pytest.raises(ValueError):
        GPUTypeSelector(type=TYPE_NAME, **{field: value})
