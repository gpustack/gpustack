"""Owner stamping on ModelSpec: response-hidden, cache-partitioning."""

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
