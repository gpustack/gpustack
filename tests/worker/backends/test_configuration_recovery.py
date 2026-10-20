"""A backend must not consume pending configuration after recovery was queued."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.models import RoleSpec
from gpustack.worker.backends.base import InferenceServer
from tests.utils.model import new_model


@pytest.mark.parametrize("role", ["prefill", "decode", "router"])
@pytest.mark.parametrize(
    "group_id, stale, blocked",
    [
        ("1-old", True, True),
        ("1-current", False, False),
        ("1-legacy", None, False),
        (None, True, False),
    ],
)
def test_backend_configuration_read_respects_group_recreation(
    group_id, stale, blocked, role
):
    model = new_model(1, "pd", huggingface_repo_id="org/model")
    model.roles = [RoleSpec(name=role, backend_parameters=["--max-model-len=8192"])]
    model.stale = stale
    client = MagicMock()
    client.models.get.return_value = model
    server = SimpleNamespace(
        _clientset=client,
        _model_instance=SimpleNamespace(model_id=1, role=role, group_id=group_id),
        _config=SimpleNamespace(data_dir="/tmp"),
    )

    if blocked:
        with pytest.raises(RuntimeError):
            InferenceServer.get_model(server)
    else:
        InferenceServer.get_model(server)
        assert server._model.backend_parameters == ["--max-model-len=8192"]

    client.models.get.assert_called_once_with(id=1, use_cache=group_id is None)
