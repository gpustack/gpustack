"""Deploy-time routes: policy load-balancing defaults.

A model deployed with replicas > 1 spawns a route whose traffic is
scheduled between the deployment's instances by the gateway — the
auto-created target carries no split weight and the least-load
capability is enabled by default. Single-replica deployments keep the
plain weighted shape.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gpustack.routes.models as models_routes
from gpustack.schemas.models import ModelCreate, SourceEnum


class _Recorder:
    def __init__(self):
        self.hook_calls = []

    async def on_route_write(self, action, route, section, session, removed=False):
        self.hook_calls.append((action, section))


def _model_in(replicas: int) -> ModelCreate:
    return ModelCreate(
        name="qwen",
        replicas=replicas,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="repo/qwen",
        enable_model_route=True,
    )


async def _run_create(monkeypatch, replicas: int):
    recorder = _Recorder()
    created = {}

    async def fake_model_create(cls, session, source, auto_commit=True):
        from gpustack.schemas.models import Model

        model = Model(**source)
        return model

    async def fake_route_create(cls, session, source, auto_commit=True):
        source.id = 3
        created["route"] = source
        return source

    async def fake_target_create(cls, session, source, auto_commit=True):
        created["target"] = source
        return source

    def fake_get_plugin(name):
        assert name == "least-load"
        return recorder

    session = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    ctx = SimpleNamespace(current_principal_id=1)
    monkeypatch.setattr(models_routes, "assert_cluster_belongs_to_org", AsyncMock())
    monkeypatch.setattr(models_routes, "validate_model_in", AsyncMock())
    monkeypatch.setattr(models_routes, "validate_shared_kv_cache", AsyncMock())
    monkeypatch.setattr(
        models_routes, "apply_scaling_schedule_baseline", lambda m: None
    )
    monkeypatch.setattr(models_routes, "revoke_model_access_cache", AsyncMock())
    monkeypatch.setattr(models_routes, "create_lora_model_routes", AsyncMock())
    monkeypatch.setattr(
        models_routes.Model, "one_by_fields", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        models_routes.ModelRoute, "one_by_fields", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(models_routes.Model, "create", classmethod(fake_model_create))
    monkeypatch.setattr(
        models_routes.ModelRoute, "create", classmethod(fake_route_create)
    )
    monkeypatch.setattr(
        models_routes.ModelRouteTarget, "create", classmethod(fake_target_create)
    )
    monkeypatch.setattr("gpustack.routes.plugins.get_route_plugin", fake_get_plugin)

    await models_routes.create_model(
        session=session, ctx=ctx, model_in=_model_in(replicas)
    )
    return created, recorder


@pytest.mark.asyncio
async def test_multi_replica_route_enables_least_load(monkeypatch):
    created, recorder = await _run_create(monkeypatch, replicas=2)

    # no split weight: capability scoring decides between instances
    assert created["target"].weight == 0
    # the least-load capability rides the plugin's own write path
    assert recorder.hook_calls == [("create", {"enabled": True})]


@pytest.mark.asyncio
async def test_single_replica_route_keeps_weighted_shape(monkeypatch):
    created, recorder = await _run_create(monkeypatch, replicas=1)

    assert created["target"].weight == 100
    # nothing to schedule between instances — no capability default
    assert recorder.hook_calls == []
