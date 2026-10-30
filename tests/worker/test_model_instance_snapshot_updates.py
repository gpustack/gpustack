from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from gpustack.client.generated_model_instance_client import ModelInstanceClient
from gpustack.schemas.models import (
    DistributedServers,
    ModelConfigSnapshot,
    ModelInstancePublic,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    SourceEnum,
)
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.runtime_metrics_aggregator import RuntimeMetricsAggregator
from gpustack.worker.serve_manager import ServeManager


def _client(snapshot):
    now = datetime.now(timezone.utc)
    public = ModelInstancePublic(
        id=11,
        name="mi-1",
        model_id=1,
        model_name="m1",
        worker_id=1,
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        created_at=now,
        updated_at=now,
        model_config_snapshot=snapshot,
        distributed_servers=DistributedServers(
            subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)]
        ),
    )
    http_client = MagicMock()
    http_client.put.return_value = httpx.Response(
        200,
        json=public.model_dump(mode="json"),
        request=httpx.Request("PUT", "http://server/model-instances/11"),
    )
    transport = MagicMock()
    transport.get_httpx_client.return_value = http_client
    api = ModelInstanceClient(transport)
    api.get = MagicMock(return_value=public)
    owner = SimpleNamespace(
        _clientset=SimpleNamespace(model_instances=api),
        _model_instance=public,
        _worker=SimpleNamespace(id=2),
    )
    return owner, http_client


@pytest.mark.parametrize(
    "cached_snapshot", [None, ModelConfigSnapshot(backend_parameters=["--old"])]
)
@pytest.mark.parametrize(
    "writer", ["backend", "manager", "metrics", "subordinate_error"]
)
def test_unrelated_updates_do_not_supply_configuration_snapshot(
    cached_snapshot, writer
):
    owner, http_client = _client(cached_snapshot)
    if writer == "subordinate_error":
        InferenceServer._update_subordinate_worker_error(owner, "worker failed")
    else:
        cls = {
            "backend": InferenceServer,
            "manager": ServeManager,
            "metrics": RuntimeMetricsAggregator,
        }[writer]
        cls._update_model_instance(owner, 11, state=ModelInstanceStateEnum.RUNNING)

    http_client.put.assert_called_once()
    payload = json.loads(http_client.put.call_args.kwargs["content"])
    assert payload.get("model_config_snapshot") is None
    if writer == "subordinate_error":
        assert (
            payload["distributed_servers"]["subordinate_workers"][0]["state"] == "error"
        )
    else:
        assert payload["state"] == "running"


def test_explicit_startup_update_includes_configuration_snapshot():
    owner, http_client = _client(ModelConfigSnapshot(backend_parameters=["--old"]))
    snapshot = ModelConfigSnapshot(backend_parameters=[], env={})
    InferenceServer._update_model_instance(owner, 11, model_config_snapshot=snapshot)
    payload = json.loads(http_client.put.call_args.kwargs["content"])
    assert (
        ModelConfigSnapshot.model_validate(payload["model_config_snapshot"]) == snapshot
    )
