from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes import model_instances as route
from gpustack.schemas.models import (
    ModelConfigSnapshot,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceUpdate,
    SourceEnum,
)
from gpustack.schemas.principals import PrincipalType


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot", ["omitted", None, {}, {"backend_parameters": ["--B"]}]
)
async def test_instance_update_preserves_snapshot_unless_a_replacement_is_supplied(
    monkeypatch, snapshot
):
    stored_config = ModelConfigSnapshot(backend_parameters=["--A"])
    instance = ModelInstance(
        id=11,
        name="mi-1",
        model_id=1,
        model_name="m1",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        model_config_snapshot=stored_config,
        state=ModelInstanceStateEnum.STARTING,
    )
    data = {
        "name": instance.name,
        "model_name": instance.model_name,
        "model_id": instance.model_id,
        "source": SourceEnum.LOCAL_PATH,
        "local_path": instance.local_path,
        "state": ModelInstanceStateEnum.RUNNING,
    }
    if snapshot != "omitted":
        data["model_config_snapshot"] = snapshot
    update = ModelInstanceUpdate.model_validate(data)
    monkeypatch.setattr(ModelInstance, "one_by_id", AsyncMock(return_value=instance))
    monkeypatch.setattr(ModelInstance, "save", AsyncMock())
    monkeypatch.setattr(ModelInstance, "_publish_event_after_commit", MagicMock())
    monkeypatch.setattr(route, "assert_resource_visible", lambda *args, **kwargs: None)
    session = MagicMock()

    async def persist(stored, incoming):
        await stored.update(session, incoming)

    monkeypatch.setattr(
        route, "ModelInstanceService", lambda _: SimpleNamespace(update=persist)
    )
    result = await route.update_model_instance(
        session,
        SimpleNamespace(user=SimpleNamespace(kind=PrincipalType.SYSTEM)),
        instance.id,
        update,
    )

    assert result.state == ModelInstanceStateEnum.RUNNING
    if snapshot is None or snapshot == "omitted":
        assert result.model_config_snapshot == stored_config
    else:
        assert result.model_config_snapshot == ModelConfigSnapshot.model_validate(
            snapshot
        )
