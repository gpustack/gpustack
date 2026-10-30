from datetime import datetime, timezone

import pytest
from sqlalchemy.dialects import mysql, postgresql

from gpustack.schemas import models


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {
            "backend_parameters": [],
            "env": {},
            "image_name": None,
            "run_command": None,
        },
        {
            "backend_parameters": ["--max-model-len=4096"],
            "env": {"HCCL_CONNECT_TIMEOUT": "120"},
            "image_name": "example/engine:1",
            "run_command": "engine serve",
            "extended_kv_cache": {"enabled": True, "mode": "local"},
            "speculative_config": {"enabled": False},
        },
    ],
)
def test_instance_config_snapshot_survives_public_and_update_round_trip(snapshot):
    assert "model_config_snapshot" in models.ModelInstance.model_fields
    now = datetime.now(timezone.utc)
    instance = models.ModelInstancePublic(
        id=11,
        name="mi-1",
        model_name="m1",
        model_id=1,
        source=models.SourceEnum.LOCAL_PATH,
        local_path="/models/demo",
        created_at=now,
        updated_at=now,
        model_config_snapshot=snapshot,
    )
    restored = models.ModelInstancePublic.model_validate_json(
        instance.model_dump_json()
    )
    update = models.ModelInstanceUpdate.model_validate(restored.model_dump())

    assert update.model_config_snapshot == restored.model_config_snapshot
    if snapshot is None:
        assert restored.model_config_snapshot is None
    else:
        assert isinstance(restored.model_config_snapshot, models.ModelConfigSnapshot)
        for name, value in snapshot.items():
            actual = restored.model_config_snapshot.model_dump(mode="json")[name]
            if isinstance(value, dict):
                assert all(actual[key] == item for key, item in value.items())
            else:
                assert actual == value


@pytest.mark.parametrize("dialect", [postgresql.dialect(), mysql.dialect()])
@pytest.mark.parametrize("snapshot", [None, {}, {"backend_parameters": [], "env": {}}])
def test_database_json_round_trip_preserves_snapshot_presence(dialect, snapshot):
    config = (
        None
        if snapshot is None
        else models.ModelConfigSnapshot.model_validate(snapshot)
    )
    column_type = models.ModelInstance.__table__.c.model_config_snapshot.type
    serialized = column_type.bind_processor(dialect)(config)
    restored = column_type.result_processor(dialect, None)(serialized)
    assert restored == config
    assert (restored is None) == (snapshot is None)
