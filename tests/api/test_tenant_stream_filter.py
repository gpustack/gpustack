"""Owner-scoped streams share the same tenant and system-account boundary."""

from types import SimpleNamespace

import pytest
from sqlalchemy.orm.evaluator import _EvaluatorCompiler

from gpustack.api.tenant import (
    TenantContext,
    tenant_list_conditions,
    tenant_stream_filter,
)
from gpustack.schemas.benchmark import Benchmark
from gpustack.schemas.cache_services import CacheService, CacheServiceInstance
from gpustack.schemas.clusters import CloudCredential, WorkerPool
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workers import Worker


@pytest.mark.parametrize(
    "model", [Worker, GPUDevice, ModelFile, Benchmark, GPUInstance]
)
@pytest.mark.parametrize(
    "kind,admin,principal_id,cluster_id,row,expected",
    [
        ("user", False, 10, None, {"owner_principal_id": 10}, True),
        ("user", False, 10, None, {"owner_principal_id": 20}, False),
        ("user", False, 10, None, {"owner_principal_id": None}, False),
        ("user", False, 10, None, {}, False),
        ("user", False, None, None, {}, False),
        ("user", True, None, None, {"owner_principal_id": 20}, True),
        ("user", True, 10, None, {"owner_principal_id": 20}, False),
        ("system", False, None, None, {"cluster_id": 30}, True),
        ("system", False, None, 30, {"cluster_id": 30}, True),
        ("system", False, None, 30, {"cluster_id": 40}, False),
        ("system", False, None, 30, {"cluster_id": None}, True),
        ("system", False, None, 30, {}, False),
    ],
)
def test_owner_stream_visibility(
    model, kind, admin, principal_id, cluster_id, row, expected
):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType(kind)),
        is_platform_admin=admin,
        current_principal_id=principal_id,
        org_role=None,
        scoped_cluster_id=cluster_id,
    )
    visible = tenant_stream_filter(ctx, model)
    assert visible(SimpleNamespace(**row)) is expected


@pytest.mark.parametrize(
    "model",
    [
        CloudCredential,
        WorkerPool,
        ModelProvider,
        Model,
        Worker,
        GPUDevice,
        ModelFile,
        Benchmark,
        GPUInstance,
        CacheService,
        CacheServiceInstance,
    ],
)
@pytest.mark.parametrize(
    "kind,admin,principal_id,cluster_id",
    [
        (PrincipalType.USER, False, 10, None),
        (PrincipalType.USER, True, 10, None),
        (PrincipalType.USER, True, None, None),
        (PrincipalType.SYSTEM, False, None, None),
        (PrincipalType.SYSTEM, False, 10, None),
        (PrincipalType.SYSTEM, False, None, 30),
        (PrincipalType.SYSTEM, False, 10, 30),
    ],
)
def test_stream_predicate_matches_list_conditions(
    model, kind, admin, principal_id, cluster_id
):
    ctx = TenantContext(
        user=SimpleNamespace(kind=kind),
        is_platform_admin=admin,
        current_principal_id=principal_id,
        org_role=None,
        scoped_cluster_id=cluster_id,
    )
    conditions = tenant_list_conditions(ctx, model)
    # Evaluate the actual SQLAlchemy expressions with SQL NULL semantics,
    # without duplicating their predicates or opening a database connection.
    sql_matches = (
        _EvaluatorCompiler(model).process(*conditions)
        if conditions
        else lambda row: True
    )
    stream_matches = tenant_stream_filter(ctx, model)
    for owner in (None, 10, 20):
        for cluster in (None, 30, 40):
            row = model(id=1, owner_principal_id=owner, cluster_id=cluster)
            assert stream_matches(row) == (sql_matches(row) is True), (
                model.__name__,
                owner,
                cluster,
            )
