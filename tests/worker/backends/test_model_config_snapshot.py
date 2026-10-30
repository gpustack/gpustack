from datetime import datetime, timezone
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.inference_backend import InferenceBackend
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelConfigSnapshot,
    ModelInstancePublic,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.worker.backends import base
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.backends.vllm import VLLMServer


def _model(**kwargs):
    return Model(
        id=1,
        name="m1",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        backend=BackendEnum.VLLM,
        **kwargs,
    )


@pytest.mark.parametrize("role", [None, "prefill"])
def test_capture_is_detached_from_model_and_backend_mutations(role):
    parameters = ["--cache-dir={data_dir}/cache"]
    model = _model(
        backend_parameters=parameters,
        env={"MODE": "A"},
        roles=(
            [
                RoleSpec(
                    name="prefill", backend_parameters=parameters, env={"MODE": "ROLE"}
                )
            ]
            if role
            else None
        ),
    )
    client = MagicMock()
    client.models.get.return_value = model
    server = SimpleNamespace(
        _clientset=client,
        _model_instance=SimpleNamespace(model_id=1, group_id=None, role=role),
        _config=SimpleNamespace(data_dir="/worker"),
    )
    InferenceServer.get_model(server)

    captured = getattr(server, "_model_config_snapshot", None)
    assert captured is not None
    assert captured.backend_parameters == ["--cache-dir={data_dir}/cache"]
    assert captured.env == {"MODE": "ROLE" if role else "A"}
    server._model.backend_parameters.append("--later")
    server._model.env["MODE"] = "changed"
    assert captured.backend_parameters == ["--cache-dir={data_dir}/cache"]
    assert captured.env == {"MODE": "ROLE" if role else "A"}
    assert server._model.backend_parameters[0] == "--cache-dir=/worker/cache"
    client.model_instances.update.assert_not_called()


def _startup(monkeypatch, config, worker_id=1, failure=False):
    now = datetime.now(timezone.utc)
    instance = ModelInstancePublic(
        id=11,
        name="mi-1",
        model_id=1,
        model_name="m1",
        worker_id=1,
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        created_at=now,
        updated_at=now,
        state=ModelInstanceStateEnum.STARTING,
        model_config_snapshot=ModelConfigSnapshot(backend_parameters=["--old"]),
    )
    client = MagicMock()
    client.workers.get.return_value = SimpleNamespace(id=worker_id)
    client.models.get.return_value = _model(backend_parameters=["--A"])
    client.model_instances.get.return_value = instance
    events = []

    def ready(self):
        events.append("starting")
        client.models.get.return_value = _model(backend_parameters=["--B"])

    def persist(id, model_update):
        events.append("write")
        if failure and model_update.model_config_snapshot is not None:
            raise RuntimeError("snapshot write failed")
        return instance

    client.model_instances.update.side_effect = persist
    monkeypatch.setattr(VLLMServer, "_until_model_instance_starting", ready)
    monkeypatch.setattr(base, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(base, "setup_runtime_logging", lambda: None)
    return instance, client, events


@pytest.mark.parametrize("worker_id", [1, 2])
def test_common_startup_persists_frozen_config_only_from_main_worker(
    monkeypatch, config, worker_id
):
    instance, client, events = _startup(monkeypatch, config, worker_id)
    VLLMServer(
        client, instance, config, worker_id, InferenceBackend(backend_name="vllm")
    )

    if worker_id == 1:
        assert events == ["starting", "write"]
        update = client.model_instances.update.call_args.kwargs["model_update"]
        assert update.model_config_snapshot.backend_parameters == ["--A"]
    else:
        assert events == ["starting"]
        client.model_instances.update.assert_not_called()
    client.models.get.assert_called_once()


def test_a_new_launch_of_the_same_instance_replaces_its_snapshot(monkeypatch, config):
    instance, client, _ = _startup(monkeypatch, config)
    backend = InferenceBackend(backend_name="vllm")
    VLLMServer(client, instance, config, 1, backend)
    first = client.model_instances.update.call_args.kwargs["model_update"]
    instance.model_config_snapshot = first.model_config_snapshot
    VLLMServer(client, instance, config, 1, backend)
    second = client.model_instances.update.call_args.kwargs["model_update"]
    assert first.model_config_snapshot.backend_parameters == ["--A"]
    assert second.model_config_snapshot.backend_parameters == ["--B"]


@pytest.mark.parametrize("failure_stage", ["read", "write"])
def test_snapshot_persistence_failure_allows_inference_startup(
    monkeypatch, config, caplog, failure_stage
):
    instance, client, _ = _startup(
        monkeypatch, config, failure=failure_stage == "write"
    )
    if failure_stage == "read":
        client.model_instances.get.side_effect = RuntimeError("private-config-value")

    with caplog.at_level(logging.WARNING, logger=base.__name__):
        server = VLLMServer(
            client, instance, config, 1, InferenceBackend(backend_name="vllm")
        )
    launch = MagicMock()
    monkeypatch.setattr(server, "_start", launch)
    server.start()

    launch.assert_called_once()
    assert instance.state == ModelInstanceStateEnum.STARTING
    assert client.model_instances.update.call_count == int(failure_stage == "write")
    assert "continuing inference startup" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "private-config-value" not in caplog.text


def test_model_read_failure_still_aborts_initialization(monkeypatch, config):
    instance, client, _ = _startup(monkeypatch, config)
    client.models.get.side_effect = RuntimeError("Model unavailable")
    with pytest.raises(SystemExit) as error:
        VLLMServer(client, instance, config, 1, InferenceBackend(backend_name="vllm"))
    assert error.value.code == 1
    update = client.model_instances.update.call_args.kwargs["model_update"]
    assert update.state == ModelInstanceStateEnum.ERROR
    assert "Model unavailable" in update.state_message
