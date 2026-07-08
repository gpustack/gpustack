from sqlalchemy import ForeignKeyConstraint

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.users import User
from gpustack.utils.usage_snapshots import build_model_usage_snapshot
from tests.utils.model import new_model


def test_build_model_usage_snapshot_includes_cluster_fields():
    cluster = Cluster(id=9, name="cluster-a", provider=ClusterProvider.Docker)
    model = new_model(
        1,
        "qwen3.5-9b",
        huggingface_repo_id="Qwen/Qwen3.5-9B",
        cluster_id=cluster.id,
    )
    model.cluster = cluster

    snapshot = build_model_usage_snapshot(model)

    assert snapshot == {
        "model_id": 1,
        "model_name": "qwen3.5-9b",
        "cluster_name": "cluster-a",
        "owner_principal_id": 1,
    }


def test_build_model_usage_snapshot_includes_user_and_api_key_fields():
    model = new_model(
        1,
        "qwen3.5-9b",
        huggingface_repo_id="Qwen/Qwen3.5-9B",
    )
    user = User(id=2, name="alice")
    api_key = ApiKey(
        id=5,
        name="test",
        user_id=2,
        access_key="abcd1234",
        hashed_secret_key="secret",
    )

    snapshot = build_model_usage_snapshot(model, user=user, api_key=api_key)

    assert snapshot["user_id"] == 2
    assert snapshot["user_name"] == "alice"
    assert snapshot["api_key_id"] == 5
    assert snapshot["api_key_name"] == "test"
    assert snapshot["access_key"] == "abcd1234"
    assert snapshot["api_key_is_custom"] is False


def test_model_usage_is_fully_fk_less():
    # model_usages carries NO foreign keys — it is an attribution / audit table
    # (like model_usage_details / metered_usage) whose rows must outlive every
    # entity they reference. Ids are kept (dangling) on parent delete instead of
    # nulled/cascaded; the read path resolves existence live to tag ``(Deleted)``.
    foreign_keys = [
        constraint
        for constraint in ModelUsage.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]
    assert foreign_keys == []
