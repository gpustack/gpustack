"""Owner stamping on ModelSpec: response-hidden, cache-partitioning."""

import pytest

from gpustack.scheduler.evaluator import make_hashable_key
from gpustack.schemas.model_sets import ModelSpec


def make_spec(owner_principal_id=None) -> ModelSpec:
    return ModelSpec(
        source="huggingface",
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        backend="vLLM",
        replicas=1,
        owner_principal_id=owner_principal_id,
    )


def test_owner_principal_id_excluded_from_serialization():
    """The UI merges the evaluation result's default_spec back into the
    model-create payload; ModelCreate.owner_principal_id is a
    non-nullable int, so the stamped field must never be serialized
    (echoing null produces a 422 on deploy)."""
    spec = make_spec(owner_principal_id=42)
    assert "owner_principal_id" not in spec.model_dump(mode="json")
    # Internal consumers still read it, and copies keep it.
    assert spec.owner_principal_id == 42
    assert spec.model_copy().owner_principal_id == 42


def test_owner_principal_id_partitions_evaluation_cache():
    """Excluded from model_dump, so the cache key must add it
    explicitly — otherwise one Org's cached evaluation result would be
    served to another."""
    key_platform = make_hashable_key(make_spec(), [])
    key_org_a = make_hashable_key(make_spec(owner_principal_id=42), [])
    key_org_b = make_hashable_key(make_spec(owner_principal_id=7), [])
    assert len({key_platform, key_org_a, key_org_b}) == 3
    # Same owner, same spec → stable key.
    assert key_org_a == make_hashable_key(make_spec(owner_principal_id=42), [])


@pytest.mark.asyncio
@pytest.mark.parametrize('projection', ['model', 'spec'])
async def test_input_revalidation_uses_the_model_owner_without_an_explicit_scope(
    monkeypatch, projection
):
    from unittest.mock import AsyncMock
    from gpustack.scheduler.evaluator import evaluate_model_input
    from gpustack.schemas.inference_backend import InferenceBackend
    from gpustack.schemas.models import Model, DisaggregationSpec, RoleSpec, SourceEnum

    model = Model(
        name='pd',
        source=SourceEnum.LOCAL_PATH,
        local_path='/model',
        backend='Custom',
        owner_principal_id=42,
        image_name='parent:engine',
        run_command='engine',
        roles=[
            RoleSpec(name='prefill'),
            RoleSpec(name='decode'),
            RoleSpec(name='router', backend='router-custom'),
        ],
        disaggregation=DisaggregationSpec(mode='custom'),
    )
    if projection == 'spec':
        model = ModelSpec(**model.model_dump())
    platform = InferenceBackend(
        backend_name='router-custom', default_run_command='router {{PLATFORM_ONLY}}'
    )
    owner = InferenceBackend(
        backend_name='router-custom',
        owner_principal_id=42,
        default_run_command='router --port {{port}}',
    )
    lookup = AsyncMock(
        side_effect=lambda session, fields: (
            owner if fields['owner_principal_id'] == 42 else platform
        )
    )
    monkeypatch.setattr(InferenceBackend, 'one_by_fields', lookup)
    assert await evaluate_model_input(None, model) == (True, [])
    assert [call.args[1]['owner_principal_id'] for call in lookup.call_args_list] == [
        None,
        42,
    ]
