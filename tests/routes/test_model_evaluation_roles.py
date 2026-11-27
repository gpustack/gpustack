"""The shape the deployment form sends to /model-evaluations.

A role held in FORM shape — override switches, GPU ids still in the cascader's
[worker, gpu] pairs — is rejected outright, and a 422 surfaces as an error
toast over the whole form. The form therefore runs the same transform submit
runs, and what these assert is that the transform's OUTPUT parses: a role that
inherits says so with nulls, and a role that overrides sends the flat wire
shape.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.dialects import mysql, postgresql

from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext
from gpustack.routes.model_evaluations import router
from gpustack.scheduler import evaluator
from gpustack.schemas.model_evaluations import (
    ModelEvaluationRequest,
    ModelEvaluationResult,
)
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.principals import PrincipalType
from gpustack.server.deps import get_session, get_tenant_context


def _request(roles):
    return ModelEvaluationRequest.model_validate(
        {
            "cluster_id": 1,
            "model_specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "x/y",
                    "backend": "vLLM",
                    "replicas": 1,
                    "roles": roles,
                    "disaggregation": {"mode": "vllm-nixl"},
                }
            ],
        }
    )


def test_an_inheriting_role_parses_with_its_nulls():
    """`null` is the wire's word for inherit, and the transform writes one into
    every field of a group left on "same as model". They have to survive
    parsing as None rather than be refused as the wrong type."""
    request = _request(
        [
            {
                "name": "prefill",
                "replicas": 2,
                "backend": None,
                "backend_version": None,
                "image_name": None,
                "run_command": None,
                "backend_parameters": None,
                "env": None,
                "gpu_selector": None,
                "worker_selector": None,
                "gpu_type_selector": None,
                "extended_kv_cache": None,
                "speculative_config": None,
            },
            {"name": "decode", "replicas": 4},
            {"name": "router", "replicas": 1, "resources": None},
        ]
    )

    roles = request.model_specs[0].roles
    assert [(r.name, r.replicas) for r in roles] == [
        ("prefill", 2),
        ("decode", 4),
        ("router", 1),
    ]
    assert roles[0].backend_parameters is None


def test_an_overriding_role_parses_its_flat_gpu_ids_and_parameters():
    """What the form's cascader holds as [worker, gpu] pairs, `generateGPUIds`
    flattens before submit. The API's `List[str]` is what rejected the form
    shape, so the flat one is the half worth pinning."""
    request = _request(
        [
            {
                "name": "decode",
                "replicas": 2,
                "backend_parameters": ["--tensor-parallel-size=4"],
                "env": {"HCCL_BUFFSIZE": "1024"},
                "gpu_selector": {
                    "gpu_ids": ["worker-a:npu:0", "worker-a:npu:1"],
                    "gpus_per_replica": 2,
                },
            },
        ]
    )

    decode = request.model_specs[0].roles[0]
    assert decode.gpu_selector.gpu_ids == ["worker-a:npu:0", "worker-a:npu:1"]
    assert decode.gpu_selector.gpus_per_replica == 2
    assert decode.backend_parameters == ["--tensor-parallel-size=4"]
    assert decode.env == {"HCCL_BUFFSIZE": "1024"}


def test_a_router_carries_its_declared_resources():
    request = _request(
        [{"name": "router", "replicas": 1, "resources": {"memory": 8589934592}}]
    )

    assert request.model_specs[0].roles[0].resources.memory == 8589934592


def test_a_plain_deployment_still_sends_no_roles():
    """The form nulls both fields when PD is off, and that has to stay the
    role-less shape rather than becoming a group with no members."""
    request = ModelEvaluationRequest.model_validate(
        {
            "cluster_id": 1,
            "model_specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "x/y",
                    "roles": None,
                    "disaggregation": None,
                }
            ],
        }
    )

    assert request.model_specs[0].roles is None
    assert request.model_specs[0].disaggregation is None


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit', [None, 3])
async def test_evaluation_reads_only_visible_clusters_but_counts_shared_claims(
    monkeypatch, explicit
):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=17,
        org_role=None,
        accessible_cluster_ids={2},
    )
    worker_rows = [
        SimpleNamespace(id=1, cluster_id=1),
        SimpleNamespace(id=2, cluster_id=2),
    ]
    shared_claim = SimpleNamespace(id=8, cluster_id=2, owner_principal_id=77)
    model_rows = [shared_claim]

    async def query(statement):
        for dialect in (postgresql.dialect(), mysql.dialect()):
            sql = str(
                statement.compile(
                    dialect=dialect, compile_kwargs={'literal_binds': True}
                )
            )
            assert 'cluster_id IN (SELECT clusters.id' in sql
            assert 'clusters.deleted_at IS NULL' in sql
            assert '(clusters.owner_principal_id = 17 OR clusters.id IN (2))' in sql
        rows = (
            model_rows
            if statement.column_descriptions[0]['entity'] is ModelInstance
            else worker_rows
        )
        return SimpleNamespace(all=lambda: rows)

    session = SimpleNamespace(exec=AsyncMock(side_effect=query))
    monkeypatch.setattr(
        evaluator.Cluster,
        'one_by_id',
        AsyncMock(return_value=SimpleNamespace(id=3, owner_principal_id=77)),
    )
    evaluate = AsyncMock(return_value=ModelEvaluationResult())
    monkeypatch.setattr(evaluator, 'evaluate_model_with_cache', evaluate)
    monkeypatch.setattr(
        evaluator.scheduler,
        'prioritize_workers_with_model_files',
        AsyncMock(return_value=worker_rows),
    )
    app = FastAPI()
    register_handlers(app)
    app.state.server_config = None
    app.include_router(router, prefix='/model-evaluations')
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_context] = lambda: ctx
    payload = _request([{'name': 'router', 'replicas': 1}]).model_dump(mode='json')
    payload['cluster_id'] = explicit
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        response = await client.post('/model-evaluations', json=payload)

    if explicit is not None:
        assert response.status_code == 404, response.text
        session.exec.assert_not_awaited()
        evaluate.assert_not_awaited()
    else:
        assert response.status_code == 200, response.text
        assert session.exec.await_count == 2
        assert evaluate.await_args.args[3:5] == (worker_rows, [shared_claim])
