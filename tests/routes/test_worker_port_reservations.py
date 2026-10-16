from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from gpustack.api.exceptions import register_handlers
from gpustack.routes.worker import port_reservations as routes
from gpustack.worker.port_reservations import WorkerPortReservations
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model_instance


@pytest.fixture
def app(tmp_path, monkeypatch):
    clientset = MagicMock()
    instance = new_model_instance(7, "model", 1, worker_id=1)
    clientset.model_instances.get.return_value = instance
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    config = SimpleNamespace(
        token="registration-token",
        log_dir=str(tmp_path),
        service_port_range="40000-40063",
    )
    manager = ServeManager(lambda: 1, lambda: clientset, config)
    app = FastAPI()
    app.state.config = config
    app.state.token = "worker-token"
    app.state.serve_manager = manager
    app.state.worker_ip_getter = lambda: "127.0.0.1"
    app.include_router(routes.router)
    register_handlers(app)
    monkeypatch.setattr("gpustack.utils.network.is_port_available", lambda *args: True)
    return app


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://worker",
        headers={"Authorization": "Bearer worker-token"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
async def test_coordination_requires_worker_management_token(app, method):
    async with _client(app) as client:
        client.headers["Authorization"] = "Bearer inference-only-key"
        response = await client.request(method, "/port-reservations/7")
    assert response.status_code == 401
    app.state.serve_manager._clientset.model_instances.get.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_contains_local_claims_cache_and_socket_occupancy(
    app, monkeypatch
):
    manager = app.state.serve_manager
    registry = manager._port_reservations
    registry.cache_ports[1] = (40021, 40029)
    registry.model_claims[8] = ("another-attempt", {40022})
    monkeypatch.setattr(
        "gpustack.utils.network.is_port_available", lambda port, host: port != 40023
    )
    async with _client(app) as client:
        response = await client.get("/port-reservations/7")
    assert response.status_code == 200
    assert response.json()["unavailable_ports"] == [40021, 40022, 40023, 40029]
    assert response.json()["pending_cleanup"]
    assert 7 in manager._model_instance_by_instance_id
    manager._clientset.model_instances.get.assert_called_once_with(
        id=7, use_cache=False
    )


@pytest.mark.asyncio
async def test_claim_is_atomic_idempotent_and_fenced_until_cancelled(app):
    token = str(uuid4())
    registry = app.state.serve_manager._port_reservations
    generations = {1: str(registry.claim_generation(7))}
    payload = {
        "token": token,
        "ports": list(range(40019, 40029)),
        "generations": generations,
    }
    async with _client(app) as client:
        for _ in range(2):
            response = await client.post("/port-reservations/7", json=payload)
            assert response.json()["accepted"]
        registry.cache_ports[9] = (40030,)
        response = await client.post(
            "/port-reservations/7",
            json={"token": str(uuid4()), "ports": [40030], "generations": generations},
        )
        assert not response.json()["accepted"]
        await client.delete(
            "/port-reservations/7", headers={"X-Claim-Token": str(uuid4())}
        )
        assert registry.model_claims[7][1] == set(payload["ports"])
        response = await client.delete(
            "/port-reservations/7", headers={"X-Claim-Token": token}
        )
        assert response.status_code == 204
        assert 7 not in registry.model_claims
        response = await client.post("/port-reservations/7", json=payload)
        assert not response.json()["accepted"]


@pytest.mark.asyncio
async def test_published_claim_cannot_be_recovered(app):
    instance = app.state.serve_manager._clientset.model_instances.get.return_value
    async with _client(app) as client:
        snapshot = (await client.get("/port-reservations/7")).json()
        claim = {
            "token": str(uuid4()),
            "ports": [40020],
            "generations": {1: snapshot["generation"]},
        }
        assert (await client.post("/port-reservations/7", json=claim)).json()[
            "accepted"
        ]
        snapshot = (await client.get("/port-reservations/7")).json()
        instance.port = 40020
        instance.ports = [40020]
        claim.update(token=str(uuid4()), replaces={1: snapshot["claim_version"]})
        response = await client.post("/port-reservations/7", json=claim)
    assert response.status_code == 200
    assert not response.json()["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unassigned", "published", "out_of_range"])
async def test_claim_rejects_invalid_placement_or_ports(app, kind):
    manager = app.state.serve_manager
    instance = manager._clientset.model_instances.get.return_value
    if kind == "unassigned":
        instance.worker_id = 2
    if kind == "published":
        instance.port = 40001
        instance.ports = [40001]
    ports = [39999] if kind == "out_of_range" else [40020]
    async with _client(app) as client:
        response = await client.post(
            "/port-reservations/7",
            json={
                "token": str(uuid4()),
                "ports": ports,
                "generations": {1: str(manager._port_reservations.claim_generation(7))},
            },
        )
    assert response.status_code == {"unassigned": 404, "published": 409}.get(kind, 200)
    if kind == "out_of_range":
        assert not response.json()["accepted"]
    assert not manager._port_reservations.model_claims


@pytest.mark.asyncio
async def test_worker_restart_rejects_claim_from_an_earlier_generation(app):
    manager = app.state.serve_manager
    async with _client(app) as client:
        snapshot = (await client.get("/port-reservations/7")).json()
        assert not snapshot["pending_cleanup"]
        payload = {
            "token": str(uuid4()),
            "ports": [40020],
            "generations": {1: snapshot["generation"]},
        }
        manager._port_reservations = WorkerPortReservations()
        response = await client.post("/port-reservations/7", json=payload)
        assert response.status_code == 200
        assert not response.json()["accepted"]
        fresh = (await client.get("/port-reservations/7")).json()
        payload["generations"] = {1: fresh["generation"]}
        response = await client.post("/port-reservations/7", json=payload)
        assert response.json()["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
@pytest.mark.parametrize("missing", ["not_found", "none"])
async def test_deleted_instance_returns_not_found(app, method, missing):
    from gpustack.api.exceptions import NotFoundException

    manager = app.state.serve_manager
    if missing == "not_found":
        manager._clientset.model_instances.get.side_effect = NotFoundException(
            "Model instance not found"
        )
    else:
        manager._clientset.model_instances.get.return_value = None
    async with _client(app) as client:
        response = await client.request(
            method,
            "/port-reservations/7",
            headers={"X-Claim-Token": str(uuid4())},
            json={"token": str(uuid4()), "ports": [40020]},
        )
    assert response.status_code == 404
    assert not manager._model_instance_by_instance_id
    assert not manager._port_reservations.model_claims


@pytest.mark.asyncio
async def test_socket_scan_does_not_hold_the_allocation_lock(app, monkeypatch):
    registry = app.state.serve_manager._port_reservations

    def probe(port, host):
        acquired = registry.lock.acquire(blocking=False)
        assert acquired, "Socket probes must leave the allocation lock available"
        registry.lock.release()
        return True

    monkeypatch.setattr("gpustack.utils.network.is_port_available", probe)
    async with _client(app) as client:
        response = await client.get("/port-reservations/7")
    assert response.status_code == 200
