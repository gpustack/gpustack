from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gpustack.routes.benchmarks as benchmarks_route
from gpustack.schemas.benchmark import (
    Benchmark,
    BenchmarkSnapshot,
    ModelInstanceSnapshot,
)
from gpustack.schemas.models import (
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    Model,
    ModelConfigSnapshot,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
    SpeculativeConfig,
    role_effective_model,
)


def _patch_worker_side(monkeypatch):
    """The worker/GPU half of the snapshot is not under test here."""
    monkeypatch.setattr(
        benchmarks_route,
        "WorkerService",
        lambda session: MagicMock(get_by_id=AsyncMock(return_value=None)),
    )
    monkeypatch.setattr(
        benchmarks_route, "create_worker_snapshot", lambda *args: (None, None)
    )


def _instance(extended_kv_cache=None) -> ModelInstance:
    return ModelInstance(
        id=11,
        name="mi-1",
        worker_id=3,
        worker_name="w1",
        state=ModelInstanceStateEnum.RUNNING,
        model_config_snapshot=ModelConfigSnapshot(extended_kv_cache=extended_kv_cache),
    )


def _model(extended_kv_cache=None) -> Model:
    return Model(name="m1", extended_kv_cache=extended_kv_cache)


@pytest.mark.asyncio
async def test_snapshot_preserves_the_instances_model_name(monkeypatch):
    _patch_worker_side(monkeypatch)
    instance = _instance()
    instance.model_name = "original-model"
    model = _model()
    model.name = "renamed-model"

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(), mi=instance, model=model
    )

    restored = BenchmarkSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.instances[instance.name].model_dump().get("model_name") == (
        "original-model"
    )


@pytest.mark.asyncio
async def test_snapshot_names_the_attached_shared_cache_service(monkeypatch):
    _patch_worker_side(monkeypatch)
    one_by_id = AsyncMock(return_value=SimpleNamespace(name="svc-cache"))
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", one_by_id)

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
        model=_model(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
    )

    assert snapshot.instances["mi-1"].cache_service_name == "svc-cache"
    assert one_by_id.await_args.args[1] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extended_kv_cache",
    [
        None,
        ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL),
    ],
)
async def test_snapshot_skips_the_lookup_without_a_shared_cache(
    monkeypatch, extended_kv_cache
):
    _patch_worker_side(monkeypatch)
    one_by_id = AsyncMock()
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", one_by_id)

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(extended_kv_cache),
        model=_model(extended_kv_cache),
    )

    assert snapshot.instances["mi-1"].cache_service_name is None
    one_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_tolerates_a_deleted_cache_service(monkeypatch):
    _patch_worker_side(monkeypatch)
    monkeypatch.setattr(
        benchmarks_route.CacheService, "one_by_id", AsyncMock(return_value=None)
    )

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        session=MagicMock(),
        mi=_instance(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
        model=_model(
            ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
            )
        ),
    )

    assert snapshot.instances["mi-1"].cache_service_name is None


@pytest.mark.asyncio
async def test_old_and_new_replicas_keep_their_own_configuration(monkeypatch):
    _patch_worker_side(monkeypatch)
    model = _model()
    old = _instance()
    new = _instance()
    new.id, new.name = 12, "mi-2"
    old.model_config_snapshot = ModelConfigSnapshot(
        backend_parameters=["--max-model-len=4096"],
        env={"ENGINE_MODE": "A"},
        image_name="example/engine:a",
        run_command="engine-a",
        speculative_config=SpeculativeConfig(enabled=False),
    )
    new.model_config_snapshot = ModelConfigSnapshot(
        backend_parameters=["--max-model-len=8192"],
        env={"ENGINE_MODE": "B"},
        image_name="example/engine:b",
        run_command="engine-b",
    )
    for name in ModelConfigSnapshot.model_fields:
        setattr(model, name, getattr(new.model_config_snapshot, name))

    old_report = await benchmarks_route.get_benchmark_snapshot(MagicMock(), old, model)
    new_report = await benchmarks_route.get_benchmark_snapshot(MagicMock(), new, model)

    for instance, report in [(old, old_report), (new, new_report)]:
        member = report.instances[instance.name]
        assert member.config_source == "instance_snapshot"
        for name in ModelConfigSnapshot.model_fields:
            assert getattr(member, name) == getattr(
                instance.model_config_snapshot, name
            )

    old.model_config_snapshot.backend_parameters.append("--changed")
    old.model_config_snapshot.env["ENGINE_MODE"] = "C"
    old.model_config_snapshot.speculative_config.enabled = True
    restored = BenchmarkSnapshot.model_validate_json(old_report.model_dump_json())
    assert restored.instances[old.name].backend_parameters == ["--max-model-len=4096"]
    assert restored.instances[old.name].env == {"ENGINE_MODE": "A"}
    assert restored.instances[old.name].speculative_config.enabled is False


@pytest.mark.asyncio
async def test_empty_snapshot_values_do_not_inherit_current_model(monkeypatch):
    _patch_worker_side(monkeypatch)
    instance = _instance()
    instance.model_config_snapshot = ModelConfigSnapshot(backend_parameters=[], env={})
    model = Model(
        name="m1",
        backend_parameters=["--new"],
        env={"NEW": "value"},
        image_name="new-image",
        run_command="new-command",
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True),
        speculative_config=SpeculativeConfig(enabled=True),
    )
    snapshot = await benchmarks_route.get_benchmark_snapshot(
        MagicMock(), instance, model
    )
    member = snapshot.instances[instance.name]
    assert member.config_source == "instance_snapshot"
    assert member.backend_parameters == []
    assert member.env == {}
    for name in (
        "image_name",
        "run_command",
        "extended_kv_cache",
        "speculative_config",
    ):
        assert getattr(member, name) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [None, "prefill", "decode", "router"])
async def test_missing_snapshot_uses_current_model_without_backfilling_instance(
    monkeypatch, role
):
    _patch_worker_side(monkeypatch)
    instance = _instance()
    instance.model_config_snapshot = None
    instance.role = role
    model = Model(
        name="m1",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        backend_parameters=["--base"],
        env={"MODE": "base"},
        image_name="engine:current",
        run_command="engine serve",
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
        ),
        speculative_config=SpeculativeConfig(enabled=False),
        roles=(
            [RoleSpec(name=role, backend_parameters=[f"--{role}"], env={"MODE": role})]
            if role
            else None
        ),
    )
    lookup = AsyncMock(return_value=SimpleNamespace(name="current-cache"))
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", lookup)
    session = MagicMock()
    snapshot = await benchmarks_route.get_benchmark_snapshot(session, instance, model)
    member = snapshot.instances[instance.name]
    assert member.config_source == "current_model"
    effective = role_effective_model(model, role)
    for name in ModelConfigSnapshot.model_fields:
        assert getattr(member, name) == getattr(effective, name)
    assert member.cache_service_name == "current-cache"
    lookup.assert_awaited_once_with(session, 7)
    assert instance.model_config_snapshot is None

    model.env["MODE"] = "later"
    if role:
        model.roles[0].env["MODE"] = "later"
    restored = BenchmarkSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.instances[instance.name].config_source == "current_model"
    assert restored.instances[instance.name].env == {"MODE": role or "base"}


@pytest.mark.asyncio
async def test_group_uses_recorded_role_configs_and_cache_services(monkeypatch):
    _patch_worker_side(monkeypatch)
    members = [_instance(), _instance()]
    model = Model(
        id=1,
        name="m1",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/m1",
        roles=[
            RoleSpec(name="prefill", env={"ROLE": "edited"}),
            RoleSpec(name="decode", env={"ROLE": "current-decode"}),
        ],
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=99
        ),
    )
    for index, member in enumerate(members, start=1):
        member.id = index
        member.name = f"mi-{index}"
        member.model_id = model.id
        member.group_id = "group"
        member.role = "prefill" if index == 1 else "decode"
        member.model_config_snapshot = ModelConfigSnapshot(
            env={"ROLE": member.role},
            extended_kv_cache=ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=index
            ),
        )
    monkeypatch.setattr(ModelInstance, "all_by_field", AsyncMock(return_value=members))
    one_by_id = AsyncMock(
        side_effect=lambda session, id: SimpleNamespace(name=f"cache-{id}")
    )
    monkeypatch.setattr(benchmarks_route.CacheService, "one_by_id", one_by_id)

    snapshot = await benchmarks_route.get_benchmark_snapshot(
        MagicMock(), members[0], model
    )
    assert snapshot.instances["mi-1"].env == {"ROLE": "prefill"}
    assert snapshot.instances["mi-2"].env == {"ROLE": "decode"}
    assert snapshot.instances["mi-1"].cache_service_name == "cache-1"
    assert snapshot.instances["mi-2"].cache_service_name == "cache-2"

    members[1].model_config_snapshot = None
    mixed = await benchmarks_route.get_benchmark_snapshot(
        MagicMock(), members[0], model
    )
    assert mixed.instances["mi-1"].config_source == "instance_snapshot"
    assert mixed.instances["mi-1"].env == {"ROLE": "prefill"}
    assert mixed.instances["mi-1"].cache_service_name == "cache-1"
    assert mixed.instances["mi-2"].config_source == "current_model"
    assert mixed.instances["mi-2"].env == {"ROLE": "current-decode"}
    assert mixed.instances["mi-2"].cache_service_name == "cache-99"
    assert members[1].model_config_snapshot is None


@pytest.mark.asyncio
async def test_export_benchmark_serializes_shared_cache_mode(monkeypatch):
    benchmark = Benchmark(
        name="bm",
        model_instance_name="mi-1",
        dataset_name="Random",
        snapshot=BenchmarkSnapshot(
            instances={
                "mi-1": ModelInstanceSnapshot(
                    id=11,
                    name="mi-1",
                    computed_resource_claim=None,
                    ports=None,
                    extended_kv_cache=ExtendedKVCacheConfig(
                        enabled=True,
                        mode=KVCacheModeEnum.SHARED,
                        cache_service_id=7,
                    ),
                    config_source="current_model",
                )
            }
        ),
    )
    monkeypatch.setattr(
        benchmarks_route.Benchmark,
        "all_by_fields",
        AsyncMock(return_value=[benchmark]),
    )
    monkeypatch.setattr(benchmarks_route, "tenant_list_conditions", lambda *args: [])

    response = await benchmarks_route.export_benchmarks(
        session=MagicMock(), ctx=MagicMock(), ids=[1]
    )

    exported = yaml.safe_load(response.body)["benchmarks"][0]
    cache = exported["snapshot"]["instances"]["mi-1"]["extended_kv_cache"]
    assert cache["mode"] == "shared"
    assert cache["cache_service_id"] == 7
    assert exported["snapshot"]["instances"]["mi-1"]["config_source"] == "current_model"
