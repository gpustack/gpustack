import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderHealthCheck,
    CacheProviderL2Backend,
    CacheProviderL2Field,
    CacheProviderVersionConfig,
)
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceConfig,
    CacheServiceInstance,
    CacheServiceL2Storage,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.cache_service_manager import (
    MAX_CONSECUTIVE_RESTARTS,
    CacheServiceManager,
)
from gpustack_runtime.deployer import WorkloadStatusStateEnum


def _build_manager(worker_id: int = 1):
    clientset = MagicMock()
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    clientset.cache_services.get.return_value = _new_cache_service()
    cfg = SimpleNamespace(
        service_port_range="40000-41000",
        system_default_container_registry=None,
    )
    manager = CacheServiceManager(lambda: worker_id, lambda: clientset, cfg)
    return manager, clientset


def _new_cache_service(**overrides) -> CacheService:
    """The parent cache service carrying provider and rendering config."""
    fields = dict(
        id=5,
        name="shared-kv",
        provider_name="mooncake",
        provider_version=None,
        mode=CacheServiceModeEnum.MANAGED,
        cluster_id=1,
        worker_id=1,
        state=CacheServiceStateEnum.PENDING,
        config=CacheServiceConfig(ram_size=8, chunk_size=256, env={"FOO": "bar"}),
    )
    fields.update(overrides)
    return CacheService(**fields)


def _new_instance(**overrides) -> CacheServiceInstance:
    fields = dict(
        id=11,
        name="mooncake-svc-a1b2c",
        cache_service_id=5,
        worker_id=1,
        cluster_id=1,
        state=CacheServiceStateEnum.PENDING,
    )
    fields.update(overrides)
    return CacheServiceInstance(**fields)


INSTANCE_WORKLOAD_NAME = "cache-svc-5-i11"


def _new_provider(**overrides) -> CacheProvider:
    fields = dict(
        name="mooncake",
        supported_modes=["managed"],
        default_version="v1",
        versions={
            "v1": CacheProviderVersionConfig(
                image="registry.example.com/mooncake/server:v1",
                run_command=(
                    "cache-server --host {{host}} --port {{port}} "
                    "--ram {{ram_size}} --metrics-port {{metrics_port}}"
                ),
                env={"CHUNK_SIZE": "{{chunk_size}}", "RAM_SIZE": "{{ram_size}}"},
            )
        },
        health_check=CacheProviderHealthCheck(scheme="tcp"),
    )
    fields.update(overrides)
    return CacheProvider(**fields)


def _l2_provider(**overrides) -> CacheProvider:
    fields = dict(
        l2_adapter_flag="--l2-adapter",
        l2_backends={
            "fs": CacheProviderL2Backend(
                fields=[
                    CacheProviderL2Field(name="base_path", required=True),
                    CacheProviderL2Field(name="use_odirect", type="boolean"),
                ]
            ),
            "resp": CacheProviderL2Backend(
                fields=[
                    CacheProviderL2Field(name="host", required=True),
                    CacheProviderL2Field(name="port", type="number", required=True),
                    CacheProviderL2Field(
                        name="username", env_name="LMCACHE_RESP_USERNAME"
                    ),
                    CacheProviderL2Field(
                        name="password",
                        type="password",
                        env_name="LMCACHE_RESP_PASSWORD",
                    ),
                    CacheProviderL2Field(name="max_capacity_gb", type="number"),
                ]
            ),
        },
    )
    fields.update(overrides)
    return _new_provider(**fields)


def _run_start(
    manager, clientset, cache_service, provider, instance=None, update_applied=True
):
    """Drive _start_cache_service_instance with the standard workload/port
    patches; returns the create_workload and _update_cache_service_instance
    mocks. ``update_applied`` is what the patched write-back reports."""
    instance = instance or _new_instance()
    clientset.cache_services.get.return_value = cache_service
    ports = iter([40001, 40002])
    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=provider,
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            side_effect=lambda **kwargs: next(ports),
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(
            manager, "_update_cache_service_instance", return_value=update_applied
        ) as update,
    ):
        manager._start_cache_service_instance(instance)
    return create, update


# ---------------------------------------------------------------------------
# Event routing
# ---------------------------------------------------------------------------


def test_event_for_other_worker_is_ignored():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(worker_id=2)

    with (
        patch.object(manager, "_start_cache_service_instance") as start,
        patch.object(manager, "_stop_cache_service_instance") as stop,
    ):
        manager._handle_cache_service_instance_event(
            Event(type=EventType.CREATED, data=instance)
        )
        manager._handle_cache_service_instance_event(
            Event(type=EventType.DELETED, data=instance)
        )

    start.assert_not_called()
    stop.assert_not_called()


def test_pending_event_triggers_start():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.PENDING)

    with patch.object(manager, "_start_cache_service_instance") as start:
        manager._handle_cache_service_instance_event(
            Event(type=EventType.CREATED, data=instance)
        )

    start.assert_called_once()
    assert start.call_args[0][0].id == instance.id


def test_non_pending_event_does_not_start():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING)

    with patch.object(manager, "_start_cache_service_instance") as start:
        manager._handle_cache_service_instance_event(
            Event(type=EventType.UPDATED, data=instance)
        )

    start.assert_not_called()


def test_deleted_event_triggers_stop():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING)

    with patch.object(manager, "_stop_cache_service_instance") as stop:
        manager._handle_cache_service_instance_event(
            Event(type=EventType.DELETED, data=instance)
        )

    stop.assert_called_once()
    assert stop.call_args[0][0].id == instance.id


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def test_gpu_resources_attaches_all_devices_on_gpu_backend():
    """The cache server needs a CUDA context on the node's GPUs to open the
    engines' IPC handles, so every device is exposed."""
    manager, _ = _build_manager(worker_id=1)
    with patch(
        "gpustack.worker.cache_service_manager.detect_backend",
        return_value="cuda",
    ):
        resources = manager._gpu_resources()

    assert resources == {"nvidia.com/devices": "all"}


def test_gpu_resources_empty_without_accelerator():
    """A CPU-only worker exposes no devices; the server runs CPU-only and
    the transfer falls back to the host-copy path."""
    manager, _ = _build_manager(worker_id=1)
    with patch(
        "gpustack.worker.cache_service_manager.detect_backend",
        return_value=[],
    ):
        resources = manager._gpu_resources()

    assert resources == {}


def test_start_instance_attaches_gpu_resources_to_container():
    manager, clientset = _build_manager(worker_id=1)
    with patch(
        "gpustack.worker.cache_service_manager.detect_backend",
        return_value="cuda",
    ):
        create, _ = _run_start(
            manager, clientset, _new_cache_service(), _new_provider()
        )

    container = create.call_args[0][0].containers[0]
    assert container.resources == {"nvidia.com/devices": "all"}


def test_start_instance_creates_workload_and_patches_starting():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    clientset.cache_services.get.return_value = _new_cache_service(
        config=CacheServiceConfig(ram_size=8, chunk_size=None, env={"FOO": "bar"})
    )

    # get_free_port mutates the shared unavailable-ports set, so snapshot
    # the set contents at each call.
    port_calls = []

    def fake_get_free_port(port_range, unavailable_ports):
        port_calls.append((port_range, set(unavailable_ports)))
        return 40001 if len(port_calls) == 1 else 40002

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            side_effect=fake_get_free_port,
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    # The parent service supplies provider, version, and rendering config.
    clientset.cache_services.get.assert_called_once_with(id=5)

    # Two ports allocated: the service port first, then the metrics port
    # with the service port excluded.
    assert port_calls == [
        ("40000-41000", set()),
        ("40000-41000", {40001}),
    ]

    plan = create.call_args[0][0]
    assert plan.name == INSTANCE_WORKLOAD_NAME
    assert plan.host_network is True
    assert plan.labels == {
        "type": "cache-service",
        "cache-service-id": "5",
        "cache-service-instance-id": "11",
    }

    container = plan.containers[0]
    assert container.name == "default"
    # Explicit registry in the provider image is preserved as-is.
    assert container.image == "registry.example.com/mooncake/server:v1"
    assert container.execution.command == [
        "cache-server",
        "--host",
        "0.0.0.0",
        "--port",
        "40001",
        "--ram",
        "8",
        "--metrics-port",
        "40002",
    ]
    envs = {e.name: e.value for e in container.envs}
    # Provider template rendered; entries rendering empty (chunk_size unset)
    # dropped; service env merged on top.
    assert envs == {"RAM_SIZE": "8", "FOO": "bar"}

    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.STARTING,
        port=40001,
        metrics_port=40002,
        state_message="",
    )
    assert manager._assigned_ports[instance.id] == (40001, 40002)


def test_start_instance_removes_stale_workload_first():
    """Start deletes any workload left from a previous run of the same
    instance (crash, manual restart) before creating the new one, so restart
    and first start share one code path."""
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    manager._assigned_ports[instance.id] = (40009, 40010)

    call_order = []
    ports = iter([40001, 40002])

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            side_effect=lambda **kwargs: next(ports),
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch(
            "gpustack.worker.cache_service_manager.delete_workload",
            side_effect=lambda name: call_order.append(("delete", name)),
        ),
        patch(
            "gpustack.worker.cache_service_manager.create_workload",
            side_effect=lambda plan: call_order.append(("create", plan.name)),
        ),
        patch.object(manager, "_update_cache_service_instance"),
    ):
        manager._start_cache_service_instance(instance)

    assert call_order == [
        ("delete", INSTANCE_WORKLOAD_NAME),
        ("create", INSTANCE_WORKLOAD_NAME),
    ]
    # The stale tracked port pair was released; a fresh pair was assigned.
    assert manager._assigned_ports[instance.id] == (40001, 40002)


def test_start_instance_tolerates_missing_stale_workload():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            return_value=40001,
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch(
            "gpustack.worker.cache_service_manager.delete_workload",
            side_effect=RuntimeError("not found"),
        ),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    create.assert_called_once()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.STARTING


def test_start_instance_drops_flags_with_empty_rendered_values():
    """A template flag whose placeholder resolves to None must not reach the
    command line as a dangling flag."""
    manager, clientset = _build_manager(worker_id=1)
    provider = _new_provider(
        versions={
            "v1": CacheProviderVersionConfig(
                image="registry.example.com/mooncake/server:v1",
                run_command=(
                    "cache-server --host {{host}} --port {{port}} "
                    "--ram {{ram_size}} --chunk-size {{chunk_size}}"
                ),
            )
        },
    )
    instance = _new_instance()
    clientset.cache_services.get.return_value = _new_cache_service(
        config=CacheServiceConfig(ram_size=8, chunk_size=None)
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=provider,
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            return_value=40001,
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance"),
    ):
        manager._start_cache_service_instance(instance)

    command = create.call_args[0][0].containers[0].execution.command
    assert command == [
        "cache-server",
        "--host",
        "0.0.0.0",
        "--port",
        "40001",
        "--ram",
        "8",
    ]


def test_start_instance_merges_user_parameters_over_template():
    """User parameters replace conflicting template flags and are appended
    verbatim otherwise."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            chunk_size=None,
            parameters=["--ram=16", "--eviction-policy=LRU"],
        )
    )

    create, _ = _run_start(manager, clientset, cache_service, _new_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command == [
        "cache-server",
        "--host",
        "0.0.0.0",
        "--port",
        "40001",
        "--metrics-port",
        "40002",
        "--ram=16",
        "--eviction-policy=LRU",
    ]


def test_start_instance_resolves_runtime_image():
    """A version's runtime_images picks the node's accelerator build —
    a CUDA 12 node gets the cu129 image; the plain image serves nodes
    whose runtime has no entry (and accelerator-less workers)."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    provider = _new_provider()
    provider.versions["v1"].runtime_images = {
        "cuda": {
            "13": "registry.example.com/mooncake/server:v1",
            "12": "registry.example.com/mooncake/server:v1-cu129",
        }
    }

    with (
        patch(
            "gpustack.worker.cache_service_manager.detect_backend",
            return_value="cuda",
        ),
        patch(
            "gpustack.worker.cache_service_manager.detect_devices",
            return_value=[SimpleNamespace(runtime_version="12.8")],
        ),
    ):
        create, _ = _run_start(manager, clientset, cache_service, provider)

    container = create.call_args[0][0].containers[0]
    assert container.image == "registry.example.com/mooncake/server:v1-cu129"


def test_start_instance_fails_fast_on_unsupported_accelerator():
    """A node whose accelerator has no runtime_images entry must error
    with the cause before any container exists — the plain image targets
    another accelerator family and would only crash-loop."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    provider = _new_provider()
    provider.versions["v1"].runtime_images = {
        "cuda": {"13": "registry.example.com/mooncake/server:v1"}
    }

    with (
        patch(
            "gpustack.worker.cache_service_manager.detect_backend",
            return_value="cann",
        ),
        patch(
            "gpustack.worker.cache_service_manager.detect_devices",
            return_value=[SimpleNamespace(runtime_version="8.0")],
        ),
    ):
        create, update = _run_start(manager, clientset, cache_service, provider)

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "no image for cann workers" in update.call_args[1]["state_message"]


def _field_provider(**overrides):
    from gpustack.schemas.cache_providers import CacheProviderField

    return _new_provider(
        managed_fields=[
            CacheProviderField(
                name="eviction_policy",
                default="LRU",
                options=["LRU", "IsolatedLRU", "noop"],
            ),
            CacheProviderField(name="eviction_ratio", type="number", default=0.2),
            CacheProviderField(name="optional_knob"),
        ],
        versions={
            "v1": CacheProviderVersionConfig(
                image="registry.example.com/mooncake/server:v1",
                run_command=(
                    "cache-server --host {{host}} --port {{port}} "
                    "--ram {{ram_size}} "
                    "--eviction-policy {{eviction_policy}} "
                    "--eviction-ratio {{eviction_ratio}} "
                    "--optional-knob {{optional_knob}}"
                ),
            )
        },
        **overrides,
    )


def test_start_instance_renders_field_defaults():
    """Declared fields fill their template placeholders: defaults render
    without any user configuration (the eviction policy is required
    upstream), and a default-less unset field drops its flag entirely."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))

    create, _ = _run_start(manager, clientset, cache_service, _field_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command[command.index("--eviction-policy") + 1] == "LRU"
    assert command[command.index("--eviction-ratio") + 1] == "0.2"
    assert "--optional-knob" not in command


def test_start_instance_field_values_and_parameter_override():
    """config.fields values replace the defaults, and a hand-written
    parameter still overrides the flag a field renders into."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            fields={"eviction_ratio": 0.5},
            parameters=["--eviction-policy=noop"],
        )
    )

    create, _ = _run_start(manager, clientset, cache_service, _field_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command[command.index("--eviction-ratio") + 1] == "0.5"
    joined = " ".join(command)
    assert "noop" in joined
    assert "LRU" not in joined


def test_start_instance_custom_version_uses_service_image():
    """The reserved "custom" version runs the user-supplied image while the
    default version's run command and env templates still render."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        provider_version="custom",
        config=CacheServiceConfig(
            ram_size=8, chunk_size=256, image="myteam/cache-server:dev"
        ),
    )

    create, update = _run_start(
        manager, clientset, cache_service, _new_provider(custom_version=True)
    )

    container = create.call_args[0][0].containers[0]
    assert container.image == "myteam/cache-server:dev"
    assert container.execution.command == [
        "cache-server",
        "--host",
        "0.0.0.0",
        "--port",
        "40001",
        "--ram",
        "8",
        "--metrics-port",
        "40002",
    ]
    envs = {e.name: e.value for e in container.envs}
    assert envs == {"CHUNK_SIZE": "256", "RAM_SIZE": "8"}
    assert update.call_args[1]["state"] == CacheServiceStateEnum.STARTING


def test_start_instance_custom_version_applies_registry_override():
    manager, clientset = _build_manager(worker_id=1)
    manager._config.system_default_container_registry = "registry.corp.local"
    cache_service = _new_cache_service(
        provider_version="custom",
        config=CacheServiceConfig(ram_size=8, image="myteam/cache-server:dev"),
    )

    create, _ = _run_start(
        manager, clientset, cache_service, _new_provider(custom_version=True)
    )

    container = create.call_args[0][0].containers[0]
    assert container.image == "registry.corp.local/myteam/cache-server:dev"


def test_start_instance_custom_version_missing_image_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        provider_version="custom",
        config=CacheServiceConfig(ram_size=8),
    )

    create, update = _run_start(
        manager, clientset, cache_service, _new_provider(custom_version=True)
    )

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "config.image is required" in update.call_args[1]["state_message"]


def test_start_instance_custom_version_without_provider_support_sets_error():
    """Route validation rejects the custom version on a provider that does
    not opt in; the worker still refuses defensively."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        provider_version="custom",
        config=CacheServiceConfig(ram_size=8, image="myteam/cache-server:dev"),
    )

    create, update = _run_start(manager, clientset, cache_service, _new_provider())

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "does not allow the custom version" in update.call_args[1]["state_message"]


def test_start_instance_custom_version_without_default_version_sets_error():
    """The custom version borrows the default version's templates, so a
    provider without a resolvable default version cannot serve it."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        provider_version="custom",
        config=CacheServiceConfig(ram_size=8, image="myteam/cache-server:dev"),
    )
    provider = _new_provider(custom_version=True, default_version=None, versions={})

    create, update = _run_start(manager, clientset, cache_service, provider)

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "default version" in update.call_args[1]["state_message"]


def test_start_instance_renders_fs_l2_adapter():
    """A single L2 storage entry renders as one flag carrying the adapter
    JSON, with the backend key as "type" and booleans as JSON booleans."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[
                CacheServiceL2Storage(
                    backend="fs",
                    params={"base_path": "/data/l2", "use_odirect": True},
                )
            ],
        )
    )

    create, update = _run_start(manager, clientset, cache_service, _l2_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command[-2:] == [
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/l2","use_odirect":true}',
    ]
    assert update.call_args[1]["state"] == CacheServiceStateEnum.STARTING


def test_start_instance_resp_l2_credentials_go_to_env():
    """resp credentials are delivered via env (the server falls back to
    LMCACHE_RESP_* when they are absent from the JSON) so secrets never
    reach the command line; number fields render as JSON integers when
    integral."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            env={"FOO": "bar"},
            l2_storages=[
                CacheServiceL2Storage(
                    backend="resp",
                    params={
                        "host": "10.0.0.8",
                        # JSON round-trips may deliver integral numbers as floats.
                        "port": 6379.0,
                        "username": "cache",
                        "password": "s3cret",
                        "max_capacity_gb": 100,
                    },
                )
            ],
        )
    )

    create, _ = _run_start(manager, clientset, cache_service, _l2_provider())

    container = create.call_args[0][0].containers[0]
    command = container.execution.command
    assert command[-2:] == [
        "--l2-adapter",
        '{"type":"resp","host":"10.0.0.8","port":6379,"max_capacity_gb":100}',
    ]
    assert "s3cret" not in " ".join(command)
    envs = {e.name: e.value for e in container.envs}
    assert envs["LMCACHE_RESP_USERNAME"] == "cache"
    assert envs["LMCACHE_RESP_PASSWORD"] == "s3cret"
    assert envs["FOO"] == "bar"


def test_start_instance_without_l2_storage_omits_flag():
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))

    create, _ = _run_start(manager, clientset, cache_service, _l2_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert "--l2-adapter" not in command


def test_start_instance_renders_l2_cascade_in_declared_order():
    """Each L2 entry renders as its own flag occurrence, kept in list order:
    the cache server reads from the earliest tier that hits and writes to
    all of them. Env-delivered fields of all entries merge into the
    container env."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/ssd"}),
                CacheServiceL2Storage(
                    backend="resp",
                    params={"host": "10.0.0.8", "port": 6379, "password": "s3cret"},
                ),
            ],
        )
    )

    create, _ = _run_start(manager, clientset, cache_service, _l2_provider())

    container = create.call_args[0][0].containers[0]
    command = container.execution.command
    assert command[-4:] == [
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/ssd"}',
        "--l2-adapter",
        '{"type":"resp","host":"10.0.0.8","port":6379}',
    ]
    envs = {e.name: e.value for e in container.envs}
    assert envs["LMCACHE_RESP_PASSWORD"] == "s3cret"


def test_start_instance_allows_repeated_l2_backend_without_env_fields():
    """A backend whose fields all ride in the adapter JSON may appear in
    several cascade tiers (e.g. SSD and HDD paths of the fs backend)."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/ssd"}),
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/hdd"}),
            ],
        )
    )

    create, update = _run_start(manager, clientset, cache_service, _l2_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command[-4:] == [
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/ssd"}',
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/hdd"}',
    ]
    assert update.call_args[1]["state"] == CacheServiceStateEnum.STARTING


def test_start_instance_l2_env_collision_sets_error():
    """Env vars are process-global, so two entries delivering a value
    through the same env var cannot start; the instance lands in ERROR."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[
                CacheServiceL2Storage(
                    backend="resp",
                    params={"host": "10.0.0.8", "port": 6379, "password": "one"},
                ),
                CacheServiceL2Storage(
                    backend="resp",
                    params={"host": "10.0.0.9", "port": 6380, "password": "two"},
                ),
            ],
        )
    )

    create, update = _run_start(manager, clientset, cache_service, _l2_provider())

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "LMCACHE_RESP_PASSWORD" in update.call_args[1]["state_message"]


def test_start_instance_l2_hand_written_adapters_append_after(caplog):
    """Hand-written --l2-adapter occurrences in config.parameters are the
    escape hatch for adapter types the declaration doesn't cover: they stay
    on the command line, appended after the structured entries so the
    UI-visible order keeps the higher read priority."""
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            parameters=[
                "--l2-adapter",
                '{"type":"s3","bucket":"kv-spill"}',
            ],
            l2_storages=[
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/l2"}),
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/hdd"}),
            ],
        )
    )

    with caplog.at_level(logging.INFO):
        create, _ = _run_start(manager, clientset, cache_service, _l2_provider())

    command = create.call_args[0][0].containers[0].execution.command
    assert command.count("--l2-adapter") == 3
    assert command[-6:] == [
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/l2"}',
        "--l2-adapter",
        '{"type":"fs","base_path":"/data/hdd"}',
        "--l2-adapter",
        '{"type":"s3","bucket":"kv-spill"}',
    ]
    assert "appending those adapters after the structured entries" in caplog.text


def test_start_instance_unknown_l2_backend_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[CacheServiceL2Storage(backend="s3", params={})],
        )
    )

    create, update = _run_start(manager, clientset, cache_service, _l2_provider())

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "'s3'" in update.call_args[1]["state_message"]


def test_start_instance_l2_without_provider_support_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    cache_service = _new_cache_service(
        config=CacheServiceConfig(
            ram_size=8,
            l2_storages=[
                CacheServiceL2Storage(backend="fs", params={"base_path": "/data/l2"})
            ],
        )
    )

    create, update = _run_start(manager, clientset, cache_service, _new_provider())

    create.assert_not_called()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "does not support L2 storage" in update.call_args[1]["state_message"]


def test_start_instance_parent_service_missing_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    clientset.cache_services.get.side_effect = NotFoundException(
        message="Cache service not found"
    )

    with (
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    create.assert_not_called()
    update.assert_called_once()
    assert update.call_args[0][0] == instance.id
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "not found" in update.call_args[1]["state_message"]


def test_start_instance_unknown_provider_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    clientset.cache_services.get.return_value = _new_cache_service(
        provider_name="nonexistent"
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=None,
        ),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    create.assert_not_called()
    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.ERROR,
        state_message="Unknown cache provider: nonexistent",
    )


def test_start_instance_unknown_version_sets_error():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    clientset.cache_services.get.return_value = _new_cache_service(
        provider_version="v9"
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch("gpustack.worker.cache_service_manager.create_workload") as create,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    create.assert_not_called()
    update.assert_called_once()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "v9" in update.call_args[1]["state_message"]


def test_start_instance_failure_sets_error_and_releases_port():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port",
            return_value=40001,
        ),
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch(
            "gpustack.worker.cache_service_manager.create_workload",
            side_effect=RuntimeError("boom"),
        ),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager._start_cache_service_instance(instance)

    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.ERROR,
        state_message="boom",
    )
    assert instance.id not in manager._assigned_ports


def test_start_instance_releases_the_in_flight_claim():
    """Both outcomes clear the claim, so the next start (event or sync) is
    not blocked by a completed one."""
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance()
    manager._starting.add(instance.id)

    _run_start(manager, clientset, _new_cache_service(), _new_provider(), instance)

    assert instance.id not in manager._starting


def test_start_instance_reports_a_dropped_state_writeback(caplog):
    """A running container whose STARTING write-back was lost is surfaced:
    the instance is still PENDING server-side and gets started again."""
    manager, clientset = _build_manager(worker_id=1)

    with caplog.at_level(logging.ERROR):
        create, _ = _run_start(
            manager,
            clientset,
            _new_cache_service(),
            _new_provider(),
            update_applied=False,
        )

    create.assert_called_once()
    assert "failed to mark instance 11 as starting" in caplog.text


def test_update_instance_reports_failed_writeback_without_raising():
    manager, clientset = _build_manager(worker_id=1)
    clientset.cache_service_instances.get.side_effect = RuntimeError("boom")

    assert (
        manager._update_cache_service_instance(11, state=CacheServiceStateEnum.RUNNING)
        is False
    )


def test_allocate_ports_excludes_ports_of_sibling_instances():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(id=12, cache_service_id=6)
    sibling = _new_instance(
        id=13,
        cache_service_id=7,
        state=CacheServiceStateEnum.RUNNING,
        port=40001,
        metrics_port=40011,
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[sibling]
    )

    # get_free_port mutates the shared unavailable-ports set, so snapshot
    # the set contents at each call.
    port_calls = []

    def fake_get_free_port(port_range, unavailable_ports):
        port_calls.append((port_range, set(unavailable_ports)))
        return 40002 if len(port_calls) == 1 else 40003

    with patch(
        "gpustack.worker.cache_service_manager.network.get_free_port",
        side_effect=fake_get_free_port,
    ):
        port, metrics_port = manager._allocate_ports(instance)

    assert (port, metrics_port) == (40002, 40003)
    # Both of the sibling's ports are excluded; the metrics-port pick also
    # excludes the service port picked just before it. Siblings are listed
    # for this worker only.
    assert clientset.cache_service_instances.list.call_args[1]["params"] == {
        "worker_id": 1,
        "page": -1,
    }
    assert port_calls == [
        ("40000-41000", {40001, 40011}),
        ("40000-41000", {40001, 40011, 40002}),
    ]
    assert manager._assigned_ports[instance.id] == (40002, 40003)


# ---------------------------------------------------------------------------
# State sync
# ---------------------------------------------------------------------------


def test_sync_workload_failed_restarts_with_incremented_count():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.STARTING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload") as delete,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    delete.assert_called_once_with(INSTANCE_WORKLOAD_NAME)
    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.PENDING,
        restart_count=1,
        last_restart_time=ANY,
        state_message=(
            f"Cache server exited; restarting (attempt 1/{MAX_CONSECUTIVE_RESTARTS})."
        ),
        healthy=False,
    )


def test_sync_workload_failed_with_restart_on_error_disabled_parks_in_error():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )
    # restart_on_error lives on the parent service and applies to all of
    # its instances.
    clientset.cache_services.get.return_value = _new_cache_service(
        restart_on_error=False
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload") as delete,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    # No restart attempt: the dead workload is left for inspection and the
    # instance is parked in ERROR for manual handling.
    delete.assert_not_called()
    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.ERROR,
        state_message=(
            "Cache server exited. Automatic restart is disabled "
            "for this service; restart it manually."
        ),
        healthy=False,
    )


def test_sync_workload_missing_restarts_and_tolerates_absent_workload():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.delete_workload",
            side_effect=RuntimeError("not found"),
        ),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_called_once()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.PENDING
    assert update.call_args[1]["restart_count"] == 1


def test_sync_crash_within_backoff_window_is_deferred():
    manager, clientset = _build_manager(worker_id=1)
    # restart_count=1 -> backoff delay of 60 seconds; the last restart was
    # only 5 seconds ago, so this round must not touch the instance.
    instance = _new_instance(
        state=CacheServiceStateEnum.STARTING,
        port=40001,
        restart_count=1,
        last_restart_time=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=None,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload") as delete,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    delete.assert_not_called()
    update.assert_not_called()


def test_sync_crash_after_backoff_window_restarts():
    manager, clientset = _build_manager(worker_id=1)
    # restart_count=1 -> backoff delay of 60 seconds; 120 seconds have
    # passed, so the restart proceeds with the incremented attempt.
    instance = _new_instance(
        state=CacheServiceStateEnum.STARTING,
        port=40001,
        restart_count=1,
        last_restart_time=datetime.now(timezone.utc) - timedelta(seconds=120),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=None,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_called_once()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.PENDING
    assert update.call_args[1]["restart_count"] == 2


def test_sync_crash_after_max_restarts_parks_in_error():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(
        state=CacheServiceStateEnum.STARTING,
        port=40001,
        restart_count=MAX_CONSECUTIVE_RESTARTS,
        last_restart_time=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=None,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload") as delete,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    delete.assert_not_called()
    update.assert_called_once()
    assert update.call_args[1]["state"] == CacheServiceStateEnum.ERROR
    assert "logs" in update.call_args[1]["state_message"]


def test_sync_ready_probe_resets_restart_count_after_stable_window():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(
        state=CacheServiceStateEnum.RUNNING,
        port=40001,
        healthy=True,
        restart_count=3,
        last_restart_time=datetime.now(timezone.utc) - timedelta(minutes=11),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch("gpustack.worker.cache_service_manager.socket.create_connection"),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_called_once_with(instance.id, restart_count=0)


def test_sync_ready_probe_keeps_restart_count_within_stable_window():
    """The consecutive-restart budget must survive a crash-after-ready loop:
    an instance that just came back must not have its count cleared yet."""
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(
        state=CacheServiceStateEnum.RUNNING,
        port=40001,
        healthy=True,
        restart_count=3,
        last_restart_time=datetime.now(timezone.utc) - timedelta(minutes=2),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch("gpustack.worker.cache_service_manager.socket.create_connection"),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_not_called()


def test_sync_ready_tcp_probe_marks_running_healthy():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.STARTING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.socket.create_connection"
        ) as connect,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    connect.assert_called_once_with(("127.0.0.1", 40001), timeout=ANY)
    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.RUNNING,
        healthy=True,
        last_check_at=ANY,
        state_message="",
    )


def test_sync_probe_failure_after_running_marks_unreachable():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(
        state=CacheServiceStateEnum.RUNNING, port=40001, healthy=True
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.socket.create_connection",
            side_effect=OSError("connection refused"),
        ),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_called_once_with(
        instance.id,
        state=CacheServiceStateEnum.UNREACHABLE,
        healthy=False,
        last_check_at=ANY,
    )


def test_sync_probe_failure_while_starting_is_left_alone():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.STARTING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.socket.create_connection",
            side_effect=OSError("connection refused"),
        ),
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    update.assert_not_called()


def test_sync_skips_instances_of_other_workers_and_states():
    manager, clientset = _build_manager(worker_id=1)
    other_worker = _new_instance(
        id=18, state=CacheServiceStateEnum.RUNNING, worker_id=2, port=40001
    )
    parked = _new_instance(id=19, state=CacheServiceStateEnum.ERROR)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[other_worker, parked]
    )

    with (
        patch("gpustack.worker.cache_service_manager.get_workload") as get_wl,
        patch.object(manager, "_update_cache_service_instance") as update,
        patch.object(manager, "_start_cache_service_instance") as start,
    ):
        manager.sync_cache_service_instances_state()

    get_wl.assert_not_called()
    update.assert_not_called()
    start.assert_not_called()


def test_sync_starts_instance_stuck_in_pending():
    """A start that never took effect (lost event, dropped state write-back)
    is re-driven by the sync pass, so a PENDING instance cannot sit forever
    next to a container nothing points at."""
    manager, clientset = _build_manager(worker_id=1)
    stale = _new_instance(
        state=CacheServiceStateEnum.PENDING,
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=90),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[stale])

    with patch.object(manager, "_start_cache_service_instance") as start:
        manager.sync_cache_service_instances_state()

    start.assert_called_once()
    assert start.call_args[0][0].id == stale.id


def test_sync_leaves_freshly_pending_instance_to_the_event_path():
    manager, clientset = _build_manager(worker_id=1)
    fresh = _new_instance(
        state=CacheServiceStateEnum.PENDING,
        updated_at=datetime.now(timezone.utc),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[fresh])

    with patch.object(manager, "_start_cache_service_instance") as start:
        manager.sync_cache_service_instances_state()

    start.assert_not_called()


def test_sync_retries_a_stuck_pending_instance_on_the_grace_cadence():
    """A start whose write-back never lands leaves the row PENDING with an
    unchanged updated_at; the retry must not recreate the container on every
    sync pass."""
    manager, clientset = _build_manager(worker_id=1)
    stale = _new_instance(
        state=CacheServiceStateEnum.PENDING,
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=90),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[stale])

    # The stand-in clears the in-flight claim like the real start's finally
    # does, so the second pass is held back by the retry cadence alone.
    with patch.object(
        manager,
        "_start_cache_service_instance",
        side_effect=lambda instance: manager._release_start(instance.id),
    ) as start:
        manager.sync_cache_service_instances_state()
        manager.sync_cache_service_instances_state()

    start.assert_called_once()


def test_sync_does_not_duplicate_a_start_in_flight():
    """A start still pulling the provider image keeps the instance PENDING
    for minutes; the sync pass must not launch a second one."""
    manager, clientset = _build_manager(worker_id=1)
    stale = _new_instance(
        state=CacheServiceStateEnum.PENDING,
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=90),
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[stale])
    manager._starting.add(stale.id)

    with patch.object(manager, "_start_cache_service_instance") as start:
        manager.sync_cache_service_instances_state()

    start.assert_not_called()


def test_sync_fetches_shared_parent_service_once_per_pass():
    """Sibling instances of a per-node service share one parent lookup per
    sync pass instead of one API call each."""
    manager, clientset = _build_manager(worker_id=1)
    instances = [
        _new_instance(id=11, state=CacheServiceStateEnum.RUNNING, port=40001),
        _new_instance(id=12, state=CacheServiceStateEnum.RUNNING, port=40003),
    ]
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=instances
    )

    with (
        patch(
            "gpustack.worker.cache_service_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        ),
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch("gpustack.worker.cache_service_manager.socket.create_connection"),
        patch.object(manager, "_update_cache_service_instance"),
    ):
        manager.sync_cache_service_instances_state()

    clientset.cache_services.get.assert_called_once_with(id=5)


def test_sync_skips_instance_when_parent_service_missing():
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING, port=40001)
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )
    clientset.cache_services.get.side_effect = NotFoundException(
        message="Cache service not found"
    )

    with (
        patch("gpustack.worker.cache_service_manager.get_workload") as get_wl,
        patch.object(manager, "_update_cache_service_instance") as update,
    ):
        manager.sync_cache_service_instances_state()

    get_wl.assert_not_called()
    update.assert_not_called()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_stop_instance_deletes_workload_and_frees_port():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING)
    manager._assigned_ports[instance.id] = (40001, 40002)

    with patch("gpustack.worker.cache_service_manager.delete_workload") as delete:
        manager._stop_cache_service_instance(instance)

    delete.assert_called_once_with(INSTANCE_WORKLOAD_NAME)
    assert instance.id not in manager._assigned_ports


def test_stop_instance_tolerates_missing_workload():
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(state=CacheServiceStateEnum.RUNNING)

    with patch(
        "gpustack.worker.cache_service_manager.delete_workload",
        side_effect=RuntimeError("not found"),
    ):
        manager._stop_cache_service_instance(instance)


def test_start_instance_without_run_command_runs_image_entrypoint():
    """A version without a run command runs the image's own entrypoint:
    no parameters means no command override, and user parameters become
    the entrypoint's argument vector instead of being dropped."""
    manager, clientset = _build_manager(worker_id=1)
    provider = _new_provider(
        versions={
            "v1": CacheProviderVersionConfig(
                image="registry.example.com/mooncake/server:v1"
            )
        }
    )

    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    create, _ = _run_start(manager, clientset, cache_service, provider)
    assert create.call_args[0][0].containers[0].execution.command is None

    cache_service = _new_cache_service(
        config=CacheServiceConfig(ram_size=8, parameters=["--host", "0.0.0.0"])
    )
    create, _ = _run_start(manager, clientset, cache_service, provider)
    command = create.call_args[0][0].containers[0].execution.command
    assert command == ["--host", "0.0.0.0"]


def test_start_instance_host_ipc_override(monkeypatch):
    """Host IPC defaults on for cache servers (the CUDA-IPC path needs
    it) but honors the service env and the worker-global GPUSTACK_HOST_IPC
    escape hatch — e.g. PodSecurity-baseline clusters reject hostIPC pods
    and the CPU host-copy path works without it."""
    manager, clientset = _build_manager(worker_id=1)

    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    create, _ = _run_start(manager, clientset, cache_service, _new_provider())
    assert create.call_args[0][0].host_ipc is True

    cache_service = _new_cache_service(
        config=CacheServiceConfig(ram_size=8, env={"GPUSTACK_HOST_IPC": "0"})
    )
    create, _ = _run_start(manager, clientset, cache_service, _new_provider())
    assert create.call_args[0][0].host_ipc is False

    from gpustack import envs as gpustack_envs

    monkeypatch.setattr(gpustack_envs, "HOST_IPC", "false")
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    create, _ = _run_start(manager, clientset, cache_service, _new_provider())
    assert create.call_args[0][0].host_ipc is False


def test_start_instance_reuses_recorded_ports():
    """A restart keeps the instance's recorded ports when they are still
    free: engines attached to this cache server carry them in snapshots
    nothing refreshes, so changed ports would strand every running
    deployment on a dead endpoint."""
    manager, clientset = _build_manager(worker_id=1)
    instance = _new_instance(port=40005, metrics_port=40015)
    cache_service = _new_cache_service(config=CacheServiceConfig(ram_size=8))
    clientset.cache_services.get.return_value = cache_service
    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=_new_provider(),
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.is_port_available",
            return_value=True,
        ),
        patch(
            "gpustack.worker.cache_service_manager.network.get_free_port"
        ) as get_free_port,
        patch(
            "gpustack.worker.cache_service_manager.registration.determine_default_registry",
            return_value=None,
        ),
        patch(
            "gpustack.worker.cache_service_manager.transform_workload_plan",
            side_effect=lambda cfg, plan, fallback: plan,
        ),
        patch("gpustack.worker.cache_service_manager.delete_workload"),
        patch("gpustack.worker.cache_service_manager.create_workload"),
        patch.object(manager, "_update_cache_service_instance", return_value=True),
    ):
        manager._start_cache_service_instance(instance)

    get_free_port.assert_not_called()
    assert manager._assigned_ports[instance.id] == (40005, 40015)


def test_probe_targets_metrics_port_for_http_health_check(monkeypatch):
    """A health check declaring target "metrics" probes the metrics port
    (LMCache's /healthcheck lives on the HTTP frontend, not the ZMQ
    control port)."""
    manager, _ = _build_manager(worker_id=1)
    instance = _new_instance(port=40001, metrics_port=40011)
    provider = _new_provider(
        health_check=CacheProviderHealthCheck(
            scheme="http", path="/healthcheck", target="metrics"
        )
    )
    with (
        patch(
            "gpustack.worker.cache_service_manager.get_cache_provider",
            return_value=provider,
        ),
        patch("gpustack.worker.cache_service_manager.httpx.get") as http_get,
    ):
        http_get.return_value = SimpleNamespace(status_code=200)
        assert manager._probe_ready(instance, "mooncake") is True

    assert http_get.call_args[0][0] == "http://127.0.0.1:40011/healthcheck"
