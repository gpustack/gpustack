from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import yaml

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.routes import metrics
from gpustack.routes.routes import management_router
from gpustack.schemas.principals import PrincipalType


def _principal(kind):
    return SimpleNamespace(
        id=None if kind == "server" else 10,
        name="system/worker/test" if kind == "server" else kind,
        kind=PrincipalType.USER if kind in ("admin", "user") else PrincipalType.SYSTEM,
        is_admin=kind == "admin",
        worker=SimpleNamespace(id=20, cluster_id=30) if kind == "worker" else None,
        cluster=SimpleNamespace(id=30) if kind == "cluster" else None,
    )


@pytest.fixture
def metrics_api(monkeypatch, config, tmp_path):
    monkeypatch.setattr(config, "data_dir", str(tmp_path))
    monkeypatch.setattr(metrics, "_config_cache", {})
    monkeypatch.setattr(metrics, "_cache_locks", {})
    app = FastAPI()
    register_handlers(app)
    app.include_router(management_router)
    return SimpleNamespace(
        app=app,
        path=tmp_path / "custom_metrics_config.yaml",
    )


@pytest.mark.parametrize("principal_kind", ["user", "worker", "cluster", "server"])
@pytest.mark.parametrize("existing", [False, True])
def test_non_admin_cannot_write_metrics_config(metrics_api, principal_kind, existing):
    metrics_api.app.dependency_overrides[get_current_user] = lambda: _principal(
        principal_kind
    )
    original = b"runtime_mapping: {vllm: {}}\n"
    cached = {"runtime_mapping": {"vllm": {}}}
    if existing:
        metrics_api.path.write_bytes(original)
        metrics._config_cache[str(metrics_api.path)] = cached

    with TestClient(metrics_api.app) as client:
        response = client.post("/v2/metrics/config", json={"runtime_mapping": {}})

    assert response.status_code == 403
    if existing:
        assert metrics_api.path.read_bytes() == original
        assert metrics._config_cache[str(metrics_api.path)] == cached
    else:
        assert not metrics_api.path.exists()
        assert not metrics._config_cache


@pytest.mark.parametrize("existing", [False, True])
def test_platform_admin_can_update_metrics_config(metrics_api, existing):
    metrics_api.app.dependency_overrides[get_current_user] = lambda: _principal("admin")
    if existing:
        metrics_api.path.write_text("runtime_mapping: {}\n")
    updated = {"runtime_mapping": {"vllm": {"*": {"source": "target"}}}}

    with TestClient(metrics_api.app) as client:
        before = client.get("/v2/metrics/config")
        assert before.status_code == 200
        assert before.json() != updated
        response = client.post("/v2/metrics/config", json=updated)
        after = client.get("/v2/metrics/config")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert yaml.safe_load(metrics_api.path.read_text()) == updated
    assert after.status_code == 200
    assert after.json() == updated


@pytest.mark.parametrize("principal_kind", ["user", "worker", "cluster", "server"])
@pytest.mark.parametrize("endpoint", ["config", "default-config"])
def test_authenticated_principals_can_read_metrics_config(
    metrics_api, principal_kind, endpoint
):
    metrics_api.app.dependency_overrides[get_current_user] = lambda: _principal(
        principal_kind
    )
    custom = {"runtime_mapping": {"vllm": {}}}
    metrics_api.path.write_text(yaml.safe_dump(custom))

    with TestClient(metrics_api.app) as client:
        response = client.get(f"/v2/metrics/{endpoint}")

    assert response.status_code == 200
    if endpoint == "config":
        assert response.json() == custom
    else:
        with open(metrics.get_builtin_metrics_config_file_path()) as f:
            assert response.json() == yaml.safe_load(f)
