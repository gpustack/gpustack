import argparse
import json
import logging
import types
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gpustack.schemas.inference_backend import (
    InferenceBackend,
    ParameterFormatEnum,
    VersionConfig,
    VersionConfigDict,
)
from gpustack.schemas.cache_services import CacheConfigSnapshot
from gpustack.schemas.models import (
    BackendEnum,
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    SpeculativeConfig,
)
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.envs import ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY_ENV as _KNOB
from gpustack.worker.backends.base import (
    InferenceServer,
    is_ascend,
    is_ascend_310p,
    read_lora_max_rank,
    _parse_image_cuda_version,
)
from gpustack.worker.backends.custom import CustomServer
from gpustack.worker.backends.sglang import (
    SGLangServer,
    extend_sglang_mounted_lora_arguments,
    get_access_log_arguments as get_sglang_access_log_arguments,
    get_auto_parallelism_arguments as get_sglang_auto_parallelism_arguments,
    get_cache_report_arguments as get_sglang_cache_report_arguments,
)
from gpustack.worker.backends.vllm import (
    VLLMServer,
    extend_vllm_mounted_lora_arguments,
    _round_up_vllm_lora_rank,
    get_access_log_arguments as get_vllm_access_log_arguments,
    get_cache_report_arguments as get_vllm_cache_report_arguments,
)
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.utils.vllm_topology import vllm_lb_mode, external_lb_parallelism
from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    WorkloadPlan,
)
from gpustack.schemas.models import (
    DistributedServers,
    PortBand,
    ModelInstanceSubordinateWorker,
)
from tests.utils.model import new_model, new_model_instance


@pytest.mark.parametrize(
    "parameters,mode",
    [
        ([], "internal"),
        (["--data-parallel-size 4"], "internal"),
        (["--data-parallel-external-lb"], "external"),
        (["--data-parallel-multi-port-external-lb"], "multi_port_external"),
        (["--data-parallel-hybrid-lb"], "hybrid"),
    ],
)
def test_vllm_load_balancing_modes(parameters, mode):
    assert vllm_lb_mode(parameters) == mode


@pytest.mark.parametrize('separator', ['-', '_'])
@pytest.mark.parametrize('tokenized', [False, True])
@pytest.mark.parametrize(
    'flags',
    [
        ['--data-parallel-external-lb', '--no-data-parallel-external-lb'],
        ['--no-data-parallel-external-lb', '-dpe'],
        ['--data-parallel-hybrid-lb', '--no-data-parallel-hybrid-lb'],
        ['--no-data-parallel-hybrid-lb', '-dph'],
        ['-dpe', '-dph', '--no-data-parallel-external-lb'],
        ['-dph', '-dpe', '--no-data-parallel-hybrid-lb'],
        ['-dpe', '--no-data-parallel-external-lb', '-dph'],
        ['-dph', '--no-data-parallel-hybrid-lb', '-dpe'],
        ['-dpm', '-dpe', '--no-data-parallel-external-lb'],
        ['--no-data-parallel-external-lb', '--no-data-parallel-hybrid-lb'],
    ],
)
def test_vllm_lb_modes_match_native_negative_flag_order(flags, tokenized, separator):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-parallel-external-lb', '-dpe', action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--data-parallel-hybrid-lb', '-dph', action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--data-parallel-multi-port-external-lb', '-dpm', action='store_true'
    )
    native = parser.parse_args(flags)
    expected = next(
        (
            mode
            for mode, enabled in (
                ('external', native.data_parallel_external_lb),
                ('hybrid', native.data_parallel_hybrid_lb),
                ('multi_port_external', native.data_parallel_multi_port_external_lb),
            )
            if enabled
        ),
        'internal',
    )
    argv = [flag[:2] + flag[2:].replace('-', separator) for flag in flags]
    value = "You're helpful. --no-data-parallel-external-lb --data-parallel-hybrid-lb"
    argv += ['--chat-template', value]
    parameters = argv if tokenized else [shlex.join(argv)]
    assert vllm_lb_mode(parameters, tokenized=tokenized) == expected


def test_vllm_negative_hybrid_flag_keeps_native_start_rank_inference():
    assert (
        vllm_lb_mode(['--no-data-parallel-hybrid-lb', '--data-parallel-start-rank=0'])
        == 'hybrid'
    )


@pytest.mark.parametrize(
    "other",
    [
        "--data-parallel-multi-port-external-lb",
        "--data-parallel-hybrid-lb",
        "--data-parallel-start-rank=0",
    ],
)
def test_external_load_balancing_modes_cannot_be_combined(other):
    with pytest.raises(ValueError, match="load balancing"):
        vllm_lb_mode(["--data-parallel-external-lb", other])


def _external_backend(dp=2, vendor="NVIDIA"):
    backend = VLLMServer.__new__(VLLMServer)
    backend._model = types.SimpleNamespace(
        backend_parameters=[
            "--data-parallel-external-lb",
            f"--data-parallel-size={dp}",
            "--tensor-parallel-size=2",
        ],
        backend_version="0.32.0",
        image_name=None,
    )
    backend._model_instance = new_model_instance(1, "moe", model_id=1, worker_id=7)
    backend._model_instance.restart_count = 3
    backend._model_instance.distributed_servers = DistributedServers(workloads=[])
    backend._model_instance.ports = [*range(40000, 40000 + dp), 40010, 40020]
    backend._model_instance.named_ports = {
        "_vllm_external_rpc": PortBand(base=40010, count=1),
        "_vllm_mp_connecting": PortBand(base=40020, count=10),
    }
    backend._worker = types.SimpleNamespace(id=7, ip="10.0.0.7")
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(
            index=2 * i + 1,
            vendor=vendor,
            type="cuda" if vendor == "NVIDIA" else "cann",
        )
        for i in range(dp * 2)
    ]
    backend._get_pretrained_config = lambda: types.SimpleNamespace(num_experts=8)
    plan = WorkloadPlan(
        name="moe",
        containers=[
            Container(
                image="test/vllm:0.32",
                name="default",
                execution=ContainerExecution(),
                files=[],
                envs=[],
            )
        ],
    )
    return backend, plan


def _run_device_task_environment(task, env):
    result = subprocess.run(
        [
            sys.executable,
            *task['argv'][1:6],
            sys.executable,
            '-c',
            'import json, os; print(json.dumps({key: os.environ.get(key) for key in '
            '["CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", '
            '"NPU_VISIBLE_DEVICES", "VLLM_CACHE_ROOT"]}))',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi_port_external'])
def test_managed_launch_removes_overridden_negative_lb_flags(mode):
    backend, plan = _external_backend()
    parameters = [
        '--data-parallel-external-lb',
        '--data-parallel-hybrid-lb',
        '--no-data-parallel-external-lb',
        '--no-data-parallel-hybrid-lb',
        f'--data-parallel-{mode.replace("_", "-")}-lb',
        '-dp=2',
        '-tp=2',
    ]
    backend._model.backend_parameters = parameters
    backend._model_instance.gpu_indexes = [1, 3, 5, 7]
    backend._model_instance.named_ports['_vllm_external_api'] = PortBand(
        base=40000, count=1 if mode == 'hybrid' else 2
    )
    backend._model_instance.named_ports['_vllm_multi_port_external_supervisor'] = (
        PortBand(base=40030)
    )
    backend._override_entrypoint = lambda command, args, script: (command, args)
    configure = {
        'external': backend._configure_external_workload,
        'hybrid': backend._configure_hybrid_workload,
        'multi_port_external': backend._configure_multi_port_external_workload,
    }[mode]
    configure(plan, ['vllm', 'serve'], ['/models/moe', *parameters])
    if mode == 'external':
        tasks = json.loads(
            next(
                f.content
                for f in plan.containers[0].files
                if f.path.endswith('tasks.json')
            )
        )['tasks']
        commands = [task['argv'] for task in tasks]
    else:
        execution = plan.containers[0].execution
        commands = [(execution.command or []) + (execution.args or [])]
    for argv in commands:
        assert '--no-data-parallel-external-lb' not in argv
        assert '--no-data-parallel-hybrid-lb' not in argv
        assert vllm_lb_mode(argv, tokenized=True) == mode


@pytest.mark.parametrize("dp", [1, 2, 4])
@pytest.mark.parametrize('separator', ['-', '_'])
def test_external_plan_has_all_ranks_and_injected_supervisor(dp, separator):
    backend, plan = _external_backend(dp)
    backend._model.backend_parameters = [
        parameter.replace('-', separator).replace(separator * 2, '--', 1)
        for parameter in backend._model.backend_parameters
    ]
    manifest = backend._configure_external_workload(
        plan,
        ["vllm", "serve"],
        [
            "/models/moe",
            *backend._model.backend_parameters,
            "--dtype",
            "float16",
            "--port=1234",
        ],
    )
    assert manifest.worker_id == 7
    assert manifest.name == "moe"
    assert [p.dp_rank for p in manifest.dp_rank_placements] == list(range(dp))
    assert [p.gpu_indexes for p in manifest.dp_rank_placements] == [
        [4 * i + 1, 4 * i + 3] for i in range(dp)
    ]
    assert [p.api_port for p in manifest.dp_rank_placements] == list(
        range(40000, 40000 + dp)
    )
    assert all(p.ready is False for p in manifest.dp_rank_placements)
    files = {f.path: f for f in plan.containers[0].files}
    assert (
        files["/gpustack/supervisor.py"].content
        == Path("gpustack/worker/backends/supervisor.py").read_text()
    )
    tasks = json.loads(files["/gpustack/tasks.json"].content)["tasks"]
    assert len(tasks) == dp
    for rank, task in enumerate(tasks):
        argv = task["argv"][6:]
        assert argv[:3] == ["vllm", "serve", "/models/moe"]
        assert argv[argv.index("--data-parallel-rank") + 1] == str(rank)
        assert argv[argv.index("--port") + 1] == str(40000 + rank)
        assert "float16" in argv
        assert "--port=1234" not in argv
    assert (
        plan.termination_grace_period_seconds
        > json.loads(files["/gpustack/tasks.json"].content)["shutdown_timeout"]
    )


@pytest.mark.parametrize("config", [None, types.SimpleNamespace(num_experts=0)])
def test_external_plan_refuses_unverified_or_dense_models(config):
    backend, plan = _external_backend()
    backend._get_pretrained_config = lambda: config
    with pytest.raises(ValueError, match="MoE"):
        backend._configure_external_workload(plan, ["vllm", "serve"], ["/models/moe"])


def test_external_device_count_cannot_discard_selected_devices():
    with pytest.raises(ValueError):
        external_lb_parallelism(
            ["--data-parallel-size=2", "--tensor-parallel-size=2"], 5
        )


@pytest.mark.parametrize('primary', [False, True])
@pytest.mark.parametrize('deleted', [False, True])
def test_managed_failure_reporting_preserves_original_error_on_generation_race(
    primary, deleted
):
    from unittest.mock import MagicMock
    from gpustack.api.exceptions import NotFoundException

    backend, _ = _external_backend()
    backend._model.backend = 'vLLM'
    backend._model_instance.worker_id = backend._worker.id if primary else 8
    backend._clientset = MagicMock()
    fresh = backend._model_instance.model_copy(deep=True)
    fresh.restart_count += 1
    backend._clientset.model_instances.get.return_value = fresh
    if deleted:
        backend._clientset.model_instances.get.side_effect = NotFoundException(
            'deleted'
        )
    original = RuntimeError('engine startup failed')
    with pytest.raises(RuntimeError) as error:
        InferenceServer._handle_error(backend, original)
    assert error.value is original


@pytest.mark.parametrize('device_types', [[], ['rocm'], ['cuda', 'cann']])
def test_external_plan_rejects_missing_or_incompatible_devices(device_types):
    backend, plan = _external_backend()
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(type=device_type) for device_type in device_types
    ]
    with pytest.raises(ValueError, match="homogeneous CUDA or Ascend devices"):
        backend._configure_external_workload(plan, ["vllm", "serve"], ["/models/moe"])


@pytest.mark.parametrize(
    "vendor,key,pool",
    [
        ("NVIDIA", "CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d"),
        ("Huawei", "ASCEND_RT_VISIBLE_DEVICES", "1,3,5,7"),
    ],
)
@pytest.mark.parametrize('local_devices', [1, 2])
@pytest.mark.parametrize('cache_root', [None, '/cache/vllm', '/custom cache/'])
def test_external_task_masks_use_positions_in_the_container_device_pool(
    vendor, key, pool, local_devices, cache_root
):
    backend, plan = _external_backend(vendor=vendor)
    backend._model.backend_parameters[-1] = f'--tensor-parallel-size={local_devices}'
    devices = backend._get_selected_gpu_devices()
    backend._get_selected_gpu_devices = lambda: devices[: 2 * local_devices]
    pool = ','.join(pool.split(',')[: 2 * local_devices])
    backend._configure_external_workload(plan, ["vllm", "serve"], ["/models/moe"])
    config = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith("tasks.json")
        )
    )
    env = {**os.environ, key: pool}
    env.pop('VLLM_CACHE_ROOT', None)
    if cache_root is not None:
        env['VLLM_CACHE_ROOT'] = cache_root
    for rank, task in enumerate(config["tasks"]):
        actual = _run_device_task_environment(task, env)
        assert actual[key] == ",".join(
            pool.split(",")[rank * local_devices : (rank + 1) * local_devices]
        )
        if vendor == 'Huawei':
            assert actual['NPU_VISIBLE_DEVICES'] == actual[key]
        assert actual['VLLM_CACHE_ROOT'] == (
            os.path.join(cache_root, 'isolated_devices', str(local_devices))
            if cache_root
            else None
        )


@pytest.mark.parametrize(
    "command",
    [
        ["vllm", "serve", "/models/moe", "--dtype=float16"],
        ["/venv/bin/vllm", "serve", "/models/moe", "--dtype", "float16"],
    ],
)
def test_external_custom_command_preserves_options_and_one_entrypoint(command):
    backend, plan = _external_backend()
    backend._configure_external_workload(plan, ["vllm", "serve"], command)
    tasks = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith("tasks.json")
        )
    )["tasks"]
    assert tasks[0]["argv"][6:9] == command[:3]
    assert all("--dtype=float16" in t["argv"] or "float16" in t["argv"] for t in tasks)


@pytest.mark.parametrize(
    "key,pool", [("ASCEND_RT_VISIBLE_DEVICES", None), ("CUDA_VISIBLE_DEVICES", "GPU-a")]
)
def test_external_mask_mismatch_cannot_run_the_engine(key, pool):
    from gpustack.worker.backends.vllm import _DEVICE_TASK_EXEC

    env = dict(os.environ)
    env.pop(key, None)
    if pool is not None:
        env[key] = pool
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _DEVICE_TASK_EXEC,
            key,
            "4",
            "[0,1]",
            sys.executable,
            "-c",
            "print('engine-started')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "engine-started" not in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-c", "vllm serve model"],
        ["vllm", "serve", "model", "&&", "echo", "ready"],
    ],
)
def test_external_ambiguous_command_is_rejected(argv):
    backend, plan = _external_backend()
    with pytest.raises(ValueError, match="directly executable"):
        backend._configure_external_workload(plan, argv, [])


@pytest.mark.parametrize(
    'flag',
    [
        '--headless',
        '--data-parallel-external-lb',
        '-dpe',
        '--data-parallel-multi-port-external-lb',
        '-dpm',
        '--data-parallel-hybrid-lb',
        '-dph',
    ],
)
@pytest.mark.parametrize('value', ['false', 'true', '0', 'YES'])
def test_managed_command_consumes_explicit_boolean_values(flag, value):
    assert VLLMServer._external_command(
        ['vllm', 'serve'], ['/models/moe', flag, value, '--dtype', 'float16']
    ) == ['vllm', 'serve', '/models/moe', '--dtype', 'float16']


@pytest.mark.parametrize('flag', ['--headless', '--data-parallel-external-lb', '-dph'])
def test_managed_command_bare_boolean_preserves_following_model_path(flag):
    assert VLLMServer._external_command(
        ['vllm', 'serve'], [flag, '/models/moe', '--dtype', 'float16']
    ) == ['vllm', 'serve', '/models/moe', '--dtype', 'float16']


def test_external_ports_cannot_treat_control_ports_as_extra_rank_apis():
    backend, plan = _external_backend()
    backend._model.backend_parameters = [
        "--data-parallel-external-lb",
        "--data-parallel-size=4",
        "--tensor-parallel-size=1",
    ]
    with pytest.raises(ValueError, match="ports"):
        backend._configure_external_workload(plan, ["vllm", "serve"], ["/models/moe"])


@pytest.mark.parametrize(
    "parameter",
    [
        "--distributed-executor-backend=ray",
        "--data-parallel-backend=ray",
        "--device-ids=0,1",
    ],
)
def test_external_unsupported_device_or_executor_overrides_are_rejected(parameter):
    backend, plan = _external_backend()
    with pytest.raises(ValueError, match="multiprocessing|Device selection"):
        backend._configure_external_workload(
            plan, ["vllm", "serve"], ["/models/moe", parameter]
        )


@pytest.mark.parametrize(
    "mode",
    [
        "external",
        "internal",
        "multi_port_external",
        "multi_port_external_managed",
        "hybrid",
        "disabled_external",
        "disabled_hybrid",
        "manual_external",
        "manual_hybrid",
        "manual_mixed",
    ],
)
@pytest.mark.parametrize('manual_dp', ['0', '1'])
def test_vllm_workload_creation_uses_the_selected_lifecycle(
    monkeypatch, mode, manual_dp
):
    backend, plan = _external_backend()
    backend._model.env = {'GPUSTACK_MANUAL_DP': manual_dp}
    if mode.startswith('manual_'):
        backend._model_instance.distributed_servers = None
        backend._model.backend_parameters = [
            '--data-parallel-size=2',
            '--data-parallel-size-local=1',
            '--tensor-parallel-size=2',
            '--data-parallel-address=10.0.0.9',
            '--data-parallel-rpc-port=13345',
            '--data-parallel-rank=1',
            '--headless',
            (
                '--data-parallel-hybrid-lb'
                if mode == 'manual_hybrid'
                else '--data-parallel-external-lb'
            ),
        ]
        if mode == 'manual_mixed':
            backend._model.backend_parameters.append('--data-parallel-start-rank=1')
    elif mode == "hybrid":
        backend._model.backend_parameters[0] = '--data-parallel-hybrid-lb'
        backend._model_instance.named_ports['_vllm_external_api'] = PortBand(base=40000)
    elif mode.startswith('disabled_'):
        backend._model_instance.distributed_servers = None
        flag = mode.removeprefix('disabled_')
        backend._model.backend_parameters = [
            f'--data-parallel-{flag}-lb',
            f'--no-data-parallel-{flag}-lb',
            '-tp=2',
            '-dp=2',
        ]
    elif mode == "multi_port_external_managed":
        backend._model.backend_parameters[0] = '-dpm'
        backend._model_instance.named_ports['_vllm_external_api'] = PortBand(
            base=40000, count=2
        )
        backend._model_instance.named_ports['_vllm_multi_port_external_supervisor'] = (
            PortBand(base=40030)
        )
    elif mode != "external":
        backend._model_instance.distributed_servers = None
        backend._model.backend_parameters = (
            ["--data-parallel-multi-port-external-lb"]
            if mode == "multi_port_external"
            else []
        ) + ["--data-parallel-size=2"]
    from unittest.mock import MagicMock

    backend._get_configured_mounts = lambda: []
    backend._get_configured_ports = lambda: []
    backend._get_configured_files = lambda: []
    backend._get_container_env_config = lambda env: types.SimpleNamespace(
        user=0, group=0, shm_size_gib=1
    )
    backend._host_ipc_enabled = lambda: False
    backend._transform_workload_plan = lambda plan: plan
    backend._workload_plan_callback = MagicMock()
    create = MagicMock()
    monkeypatch.setattr("gpustack.worker.backends.vllm.create_workload", create)
    backend._create_workload(
        backend._model_instance.get_deployment_metadata(7),
        ["vllm", "serve"],
        '#!/bin/sh\nexec "$@"',
        (
            ["/models/moe", *backend._model.backend_parameters]
            if mode in ('multi_port_external_managed', 'hybrid')
            or mode.startswith(('manual_', 'disabled_'))
            else ["/models/moe"]
        ),
        {},
        "image",
    )
    if mode not in ("external", "multi_port_external_managed", "hybrid"):
        create.assert_called_once()
        backend._workload_plan_callback.assert_not_called()
        assert create.call_args.args[0].containers[0].files == []
        if mode.startswith(('manual_', 'disabled_')):
            container = create.call_args.args[0].containers[0]
            argv = (container.execution.command or []) + (
                container.execution.args or []
            )
            for parameter in backend._model.backend_parameters:
                assert parameter in argv
            backend._workload_plan_callback.assert_not_called()
        return
    create.assert_not_called()
    manifests, plans = backend._workload_plan_callback.call_args.args
    assert manifests[0].name == plans[0].name
    assert len(plans[0].containers) == 1
    assert "1,3,5,7" in plans[0].containers[0].resources.values()
    cloned = pickle.loads(pickle.dumps((manifests, plans)))
    assert cloned[0][0] == manifests[0]
    if mode == 'hybrid':
        assert manifests[0].component.kind == 'vllm_hybrid'
        assert '--data-parallel-hybrid-lb' in cloned[1][0].containers[0].execution.args
        assert cloned[1][0].containers[0].files == []
        return
    if mode == 'multi_port_external_managed':
        argv = cloned[1][0].containers[0].execution.args
        assert '--data-parallel-multi-port-external-lb' in argv
        assert cloned[1][0].containers[0].files == []
        return
    assert cloned[1][0].containers[0].execution.args == [
        "python3",
        "/gpustack/supervisor.py",
        "/gpustack/tasks.json",
    ]


@pytest.mark.parametrize("grace", [1, 10, 30, 120])
def test_external_supervisor_shutdown_finishes_before_container_deadline(grace):
    backend, plan = _external_backend()
    plan.termination_grace_period_seconds = grace
    backend._configure_external_workload(plan, ["vllm", "serve"], ["/models/moe"])
    config = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith("tasks.json")
        )
    )
    assert config["shutdown_timeout"] == 10
    assert config["cleanup_timeout"] == 5
    assert plan.termination_grace_period_seconds == max(grace, 20)


@pytest.mark.parametrize(
    'parameters,worker_timeout,expected',
    [
        ([], None, 10),
        (['--shutdown-timeout', '0'], '0', 5),
        (['--shutdown-timeout', '60'], None, 65),
        (['--shutdown-timeout=60'], '90', 95),
        (['--shutdown-timeout=90', '--shutdown-timeout', '60'], None, 65),
        (['--shutdown_timeout', '60'], None, 65),
        (['--shutdown_timeout=60'], None, 65),
        (['--shutdown-timeout=60', '--shutdown_timeout', '120'], None, 125),
        (['--shutdown_timeout=120', '--shutdown-timeout', '60'], None, 65),
        (['--shutdown_timeout=60'], '90', 95),
    ],
)
def test_supervisor_budget_uses_final_child_arguments_and_environment(
    parameters, worker_timeout, expected
):
    from gpustack_runtime.deployer import ContainerEnv

    backend, plan = _external_backend()
    if worker_timeout is not None:
        plan.containers[0].envs = [
            ContainerEnv(
                name='VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS', value=worker_timeout
            )
        ]
    backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *parameters]
    )
    config = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith('tasks.json')
        )
    )
    assert config['shutdown_timeout'] == expected
    assert config['cleanup_timeout'] == 5
    assert plan.termination_grace_period_seconds == max(30, expected + 10)


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf', 'abc', '1.5', ''])
@pytest.mark.parametrize('source', ['argument', 'underscore_argument', 'environment'])
def test_supervisor_rejects_invalid_shutdown_budget(value, source):
    from gpustack_runtime.deployer import ContainerEnv

    backend, plan = _external_backend()
    parameters = ['/models/moe']
    if source in ('argument', 'underscore_argument'):
        flag = (
            '--shutdown_timeout'
            if source == 'underscore_argument'
            else '--shutdown-timeout'
        )
        parameters.append(f'{flag}={value}')
    else:
        plan.containers[0].envs = [
            ContainerEnv(name='VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS', value=value)
        ]
    with pytest.raises(ValueError, match='non-negative integer'):
        backend._configure_external_workload(plan, ['vllm', 'serve'], parameters)


def test_supervisor_budget_covers_all_children_and_task_environment_overrides():
    from gpustack_runtime.deployer import ContainerEnv

    backend, plan = _external_backend()
    plan.containers[0].envs = [
        ContainerEnv(name='VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS', value='180')
    ]
    tasks = [
        {
            'name': 'a',
            'argv': ['vllm', 'serve', '/model', '--shutdown-timeout=60'],
            'env': {'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS': '90'},
        },
        {
            'name': 'b',
            'argv': ['vllm', 'serve', '/model', '--shutdown-timeout=120'],
            'env': {'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS': '5'},
        },
    ]
    backend._configure_rank_supervisor(plan, tasks)
    config = json.loads(plan.containers[0].files[-1].content)
    assert config['shutdown_timeout'] == 125
    assert plan.termination_grace_period_seconds == 135


@pytest.mark.parametrize(
    'parameters',
    [
        ['--shutdown-timeout'],
        ['--shutdown-timeout=60', '--shutdown-timeout'],
        ['--shutdown_timeout'],
        ['--shutdown-timeout=60', '--shutdown_timeout'],
    ],
)
def test_supervisor_rejects_missing_shutdown_timeout_value(parameters):
    backend, plan = _external_backend()
    with pytest.raises(
        ValueError, match='--shutdown-timeout must be a non-negative integer'
    ):
        backend._configure_external_workload(
            plan, ['vllm', 'serve'], ['/models/moe', *parameters]
        )


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_managed_backend_reads_current_configuration_before_preparation(managed, stale):
    from unittest.mock import MagicMock

    backend, _ = _external_backend()
    if not managed:
        backend._model_instance.distributed_servers = None
    model = new_model(1, "moe", huggingface_repo_id="org/moe")
    model.stale = stale
    backend._clientset = MagicMock()
    backend._clientset.models.get.return_value = model
    backend._config = types.SimpleNamespace(data_dir="/data")
    backend.get_model()
    backend._clientset.models.get.assert_called_once_with(id=1, use_cache=not managed)


@pytest.mark.parametrize(
    "image_name, container_registry, expect_image_name, fallback_registry",
    [
        (
            "ghcr.io/ggml-org/llama.cpp:server",
            "test-registry.io",
            "ghcr.io/ggml-org/llama.cpp:server",
            None,
        ),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            "test-registry.io",
            "test-registry.io/gpustack/runner:cuda12.8-vllm0.10.2",
            None,
        ),
        (
            "foo/bar",
            "test-registry.io",
            "test-registry.io/foo/bar",
            None,
        ),
        ("ubuntu:24.04", "test-registry.io", "test-registry.io/ubuntu:24.04", None),
        (
            "gpustack/runner:cuda12.8-vllm0.10.2",
            None,
            "quay.io/gpustack/runner:cuda12.8-vllm0.10.2",
            "quay.io",
        ),
        (
            "lmsysorg/sglang:v0.5.5",
            "",
            "lmsysorg/sglang:v0.5.5",
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_apply_registry_override(
    image_name,
    container_registry,
    expect_image_name,
    fallback_registry,
    monkeypatch,
):
    backend = CustomServer.__new__(CustomServer)
    # CustomServer inherits _apply_registry_override from InferenceServer,
    # and _apply_registry_override accesses self._config.system_default_container_registry.
    # Since we constructed the instance via __new__ (without __init__),
    # the _config attribute does not exist. We attach a minimal stub config here.
    backend._config = types.SimpleNamespace(
        system_default_container_registry=container_registry,
    )
    backend._fallback_registry = fallback_registry

    assert (
        apply_registry_override_to_image(
            backend._config, image_name, backend._fallback_registry
        )
        == expect_image_name
    )

    if container_registry:
        backend._config = types.SimpleNamespace(system_default_container_registry=None)
        assert (
            apply_registry_override_to_image(
                backend._config, image_name, backend._fallback_registry
            )
            == image_name
        )


@pytest.mark.parametrize(
    "backend_parameters, expected",
    [
        (
            ["--ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            ["--served-model-name foo"],
            ["--served-model-name", "foo"],
        ),
        (
            ['--served-model-name "foo bar"'],
            ["--served-model-name", "foo bar"],
        ),
        (
            ['--arg1', '--arg2 "val with spaces"'],
            ['--arg1', '--arg2', 'val with spaces'],
        ),
        (
            ['--arg1 "val with spaces"', '--arg2="val with spaces"'],
            ['--arg1', 'val with spaces', '--arg2=val with spaces'],
        ),
        (
            [
                """--hf-overrides '{"architectures": ["NewModel"]}'""",
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
            [
                '--hf-overrides',
                '{"architectures": ["NewModel"]}',
                """--hf-overrides={"architectures": ["NewModel"]}""",
            ],
        ),
        # Test cases for whitespace handling
        (
            [" --ctx-size=1024"],
            ["--ctx-size=1024"],
        ),
        (
            ["  --ctx-size 1024"],
            ["--ctx-size", "1024"],
        ),
        (
            [" --max-model-len=8192"],
            ["--max-model-len=8192"],
        ),
        (
            ["--foo bar=baz"],
            ["--foo", "bar=baz"],
        ),
        # Test negative number values
        (
            ["--temperature -0.5"],
            ["--temperature", "-0.5"],
        ),
        # Issue #5209: shell line-continuation backslashes pasted from docs
        (
            ["--tp 2 \\"],
            ["--tp", "2"],
        ),
        (
            ["--tp 2 \\", "--max-model-len 8192 \\"],
            ["--tp", "2", "--max-model-len", "8192"],
        ),
        (
            ["--tp 2 \\\n"],
            ["--tp", "2"],
        ),
        (
            None,
            [],
        ),
    ],
)
def test_flatten_backend_param(backend_parameters, expected):
    backend = CustomServer.__new__(CustomServer)
    backend._model = types.SimpleNamespace(backend_parameters=backend_parameters)
    backend.inference_backend = None
    assert backend._flatten_backend_param() == expected


@pytest.mark.parametrize(
    "backend_parameters, parameter_format, expected",
    [
        # Test space format conversion
        (["--ctx-size=1024"], ParameterFormatEnum.SPACE, ["--ctx-size", "1024"]),
        # llama.cpp short options like -n, -ngl, -m must keep single-dash form;
        # we never auto-coerce dash count even when normalizing format.
        (["-n-gpu-layers=0"], ParameterFormatEnum.SPACE, ["-n-gpu-layers", "0"]),
        (["-ngl 99"], ParameterFormatEnum.EQUAL, ["-ngl=99"]),
        # Test equal format conversion
        (["--ctx-size 1024"], ParameterFormatEnum.EQUAL, ["--ctx-size=1024"]),
        (["-ctx-size 1024"], ParameterFormatEnum.EQUAL, ["-ctx-size=1024"]),
        # Test no conversion (None)
        (["--ctx-size 1024"], None, ["--ctx-size", "1024"]),
        (["--ctx-size=1024"], None, ["--ctx-size=1024"]),
        # Test flag parameters (no value)
        (["--verbose"], ParameterFormatEnum.SPACE, ["--verbose"]),
        (["--verbose"], ParameterFormatEnum.EQUAL, ["--verbose"]),
        (["--verbose"], None, ["--verbose"]),
        # Test parameters with spaces in value
        (['--name "my model"'], ParameterFormatEnum.SPACE, ["--name", "my model"]),
        (['--name "my model"'], ParameterFormatEnum.EQUAL, ["--name=my model"]),
        # Test multiple parameters
        (
            ["--ctx-size=1024", "--n-gpu-layers=0"],
            ParameterFormatEnum.SPACE,
            ["--ctx-size", "1024", "--n-gpu-layers", "0"],
        ),
        (
            ["--ctx-size 1024", "--n-gpu-layers 0"],
            ParameterFormatEnum.EQUAL,
            ["--ctx-size=1024", "--n-gpu-layers=0"],
        ),
        # Test mixed formats with conversion
        (
            ["--ctx-size=1024", "--n-gpu-layers 0"],
            ParameterFormatEnum.SPACE,
            ["--ctx-size", "1024", "--n-gpu-layers", "0"],
        ),
        (
            ["--ctx-size 1024", "--n-gpu-layers=0"],
            ParameterFormatEnum.EQUAL,
            ["--ctx-size=1024", "--n-gpu-layers=0"],
        ),
        # Test parameters with multiple values
        (
            ['--arg "value1 value2 value3"'],
            ParameterFormatEnum.SPACE,
            ["--arg", "value1 value2 value3"],
        ),
        (
            ['--arg "value1 value2 value3"'],
            ParameterFormatEnum.EQUAL,
            ["--arg=value1 value2 value3"],
        ),
        # Test negative number values
        (
            ["--temperature -0.5"],
            ParameterFormatEnum.SPACE,
            ["--temperature", "-0.5"],
        ),
        (
            ["--temperature -0.5"],
            ParameterFormatEnum.EQUAL,
            ["--temperature=-0.5"],
        ),
        (
            ["--temperature=-0.5"],
            ParameterFormatEnum.SPACE,
            ["--temperature", "-0.5"],
        ),
        # ----- issue #5200: multiple --key in a single slot -----
        # Space form, no enforced target format: keep original separators.
        (
            ["--gpu-memory-utilization 0.8 --max-model-len 8192"],
            None,
            ["--gpu-memory-utilization", "0.8", "--max-model-len", "8192"],
        ),
        # Equal form previously folded into the first key's value.
        (
            ["--gpu-memory-utilization=0.8 --max-model-len=8192"],
            None,
            ["--gpu-memory-utilization=0.8", "--max-model-len=8192"],
        ),
        # Mixed forms; parameter_format=SPACE normalizes them all.
        (
            ["--gpu-memory-utilization 0.8 --max-model-len=8192"],
            ParameterFormatEnum.SPACE,
            ["--gpu-memory-utilization", "0.8", "--max-model-len", "8192"],
        ),
        # Mixed forms; parameter_format=EQUAL normalizes them all.
        (
            ["--gpu-memory-utilization 0.8 --max-model-len=8192"],
            ParameterFormatEnum.EQUAL,
            ["--gpu-memory-utilization=0.8", "--max-model-len=8192"],
        ),
        # Flag-only mixed with valued params in one slot.
        (
            ["--enable-prefix-caching --tp 8 --trust-remote-code"],
            ParameterFormatEnum.SPACE,
            ["--enable-prefix-caching", "--tp", "8", "--trust-remote-code"],
        ),
        # Negative value inside a multi-param slot stays bound to its key.
        (
            ["--temperature -0.5 --top-p 0.9"],
            ParameterFormatEnum.EQUAL,
            ["--temperature=-0.5", "--top-p=0.9"],
        ),
        # JSON value (equal form) followed by another --key — JSON masking
        # keeps the JSON intact across the split.
        (
            [
                '--compilation-config={"cudagraph_mode": "FULL_DECODE_ONLY"} '
                '--max-model-len 65536'
            ],
            None,
            [
                '--compilation-config={"cudagraph_mode": "FULL_DECODE_ONLY"}',
                '--max-model-len',
                '65536',
            ],
        ),
        # Quoted JSON value (space form) followed by another --key.
        (
            ['--speculative-config \'{"num_speculative_tokens": 2}\' --tp 8'],
            None,
            [
                '--speculative-config',
                '{"num_speculative_tokens": 2}',
                '--tp',
                '8',
            ],
        ),
        # Shell line-continuation in the middle of a single slot.
        (
            ["--tp 2 \\\n--max-model-len 8192"],
            ParameterFormatEnum.SPACE,
            ["--tp", "2", "--max-model-len", "8192"],
        ),
        # vLLM --lora-modules with multiple JSON values (flat-token form).
        # parameter_format=EQUAL must NOT fold multi-value into --key=v1=v2;
        # the cluster stays in space form because equal form is unsafe here.
        (
            [
                "--lora-modules",
                '{"name": "x1", "path": "/p1"}',
                '{"name": "x2", "path": "/p2"}',
            ],
            ParameterFormatEnum.EQUAL,
            [
                "--lora-modules",
                '{"name": "x1", "path": "/p1"}',
                '{"name": "x2", "path": "/p2"}',
            ],
        ),
        # Same multi-value cluster, SPACE target — passes through.
        (
            [
                "--lora-modules",
                '{"name": "x1"}',
                '{"name": "x2"}',
            ],
            ParameterFormatEnum.SPACE,
            [
                "--lora-modules",
                '{"name": "x1"}',
                '{"name": "x2"}',
            ],
        ),
    ],
)
def test_flatten_backend_param_with_format_conversion(
    backend_parameters, parameter_format, expected
):
    backend = CustomServer.__new__(CustomServer)
    backend._model = types.SimpleNamespace(backend_parameters=backend_parameters)

    # Mock the inference backend with parameter_format configuration
    if parameter_format is not None:
        inference_backend = types.SimpleNamespace(parameter_format=parameter_format)
        backend.inference_backend = inference_backend
    else:
        backend.inference_backend = None

    assert backend._flatten_backend_param() == expected


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        (None, None, []),
        ([], "0.15.2", []),
        ([], "0.16.0", ["--disable-access-log-for-endpoints", "/metrics"]),
        (
            ["--disable-access-log-for-endpoints=/health,/metrics"],
            "0.16.0",
            [],
        ),
        (
            ["--disable-access-log-for-endpoints", "/health,/metrics"],
            "0.16.0",
            [],
        ),
    ],
)
def test_vllm_access_log_arguments(backend_parameters, backend_version, expected):
    assert (
        get_vllm_access_log_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        (None, None, []),
        ([], "0.5.8", []),
        ([], "0.5.8.post1", ["--uvicorn-access-log-exclude-prefixes", "/metrics"]),
        (
            ["--uvicorn-access-log-exclude-prefixes=/health"],
            "0.5.8.post1",
            [],
        ),
        (
            ["--uvicorn-access-log-exclude-prefixes", "/health"],
            "0.5.8.post1",
            [],
        ),
    ],
)
def test_sglang_access_log_arguments(backend_parameters, backend_version, expected):
    assert (
        get_sglang_access_log_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        # Unknown version: do not inject (we cannot version-gate it).
        (None, None, []),
        # Below the v0.9.0.1 cutoff: skipped (V1 silently dropped the field).
        ([], "0.9.0", []),
        # At/after the cutoff: injected.
        ([], "0.9.0.1", ["--enable-prompt-tokens-details"]),
        ([], "0.10.0", ["--enable-prompt-tokens-details"]),
        # User explicitly opted in: do not duplicate.
        (["--enable-prompt-tokens-details"], "0.10.0", []),
        # User explicitly opted out: respect their choice.
        (["--no-enable-prompt-tokens-details"], "0.10.0", []),
        # Prefix-caching flags are not GPUStack's responsibility — left to the user.
        (["--enable-prefix-caching"], "0.10.0", ["--enable-prompt-tokens-details"]),
    ],
)
def test_vllm_cache_report_arguments(backend_parameters, backend_version, expected):
    assert (
        get_vllm_cache_report_arguments(backend_parameters, backend_version) == expected
    )


@pytest.mark.parametrize(
    "backend_parameters, backend_version, expected",
    [
        # Unknown version: do not inject (we cannot version-gate it).
        (None, None, []),
        # Below the v0.3.4 cutoff: skipped.
        ([], "0.3.3", []),
        # At/after the cutoff: injected.
        ([], "0.3.4", ["--enable-cache-report"]),
        ([], "0.5.8.post1", ["--enable-cache-report"]),
        # User already passed it: do not duplicate.
        (["--enable-cache-report"], "0.5.8.post1", []),
    ],
)
def test_sglang_cache_report_arguments(backend_parameters, backend_version, expected):
    assert (
        get_sglang_cache_report_arguments(backend_parameters, backend_version)
        == expected
    )


@pytest.mark.parametrize(
    "backend_version, expected_error, expected_message",
    [
        (None, RuntimeError, "workload setup reached"),
        ("0.5.4", ValueError, "do not support Diffusion models"),
        ("0.5.4-custom", RuntimeError, "workload setup reached"),
        ("0.5.6", RuntimeError, "workload setup reached"),
    ],
)
def test_sglang_diffusion_checks_only_known_backend_versions(
    backend_version, expected_error, expected_message
):
    backend = SGLangServer.__new__(SGLangServer)
    backend.is_diffusion = True
    backend._model = types.SimpleNamespace(
        backend_version=backend_version, image_name=None
    )

    def reach_workload_setup(*args):
        raise RuntimeError("workload setup reached")

    backend._override_entrypoint = reach_workload_setup

    with pytest.raises(expected_error, match=expected_message):
        backend._create_workload(None, None, None, [], {}, "sglang/nightly")


@pytest.mark.parametrize(
    "backend_version, supports_bounds",
    [
        ("0.5.9", True),
        ("0.5.10rc0", True),
        ("0.5.10", False),
        ("0.5.21", False),
        pytest.param(None, False, id="unknown"),
        pytest.param("custom", False, id="unparseable"),
    ],
)
@pytest.mark.parametrize(
    "min_length, max_length",
    [(1, 10), (2, 4), (2, None), (None, 4), (None, None), (0, 0)],
    ids=[
        "ui-defaults",
        "both-bounds",
        "min-only",
        "max-only",
        "unbounded",
        "zero-bounds",
    ],
)
def test_sglang_ngram_match_length_arguments(
    backend_version, supports_bounds, min_length, max_length, caplog
):
    backend = SGLangServer.__new__(SGLangServer)
    backend._model = types.SimpleNamespace(
        backend_version=backend_version,
        backend_parameters=[],
        speculative_config=SpeculativeConfig(
            enabled=True,
            algorithm="ngram",
            num_draft_tokens=8,
            ngram_min_match_length=min_length,
            ngram_max_match_length=max_length,
        ),
    )
    backend._pretrained_config = types.SimpleNamespace(architectures=[])

    expected = [
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-num-draft-tokens",
        "8",
    ]
    if supports_bounds and max_length:
        expected.extend(["--speculative-ngram-max-match-window-size", str(max_length)])
    if supports_bounds and min_length:
        expected.extend(["--speculative-ngram-min-match-window-size", str(min_length)])
    expected.extend(["--speculative-num-steps", "5", "--speculative-eagle-topk", "4"])
    with caplog.at_level(logging.WARNING, logger="gpustack.worker.backends.sglang"):
        assert backend._get_speculative_arguments() == expected

    if (min_length or max_length) and not supports_bounds:
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert f"ngram_min_match_length={min_length}" in message
        assert f"ngram_max_match_length={max_length}" in message
        assert "--speculative-ngram-min-match-window-size" in message
        assert "--speculative-ngram-max-match-window-size" in message
        assert f"backend version: {backend_version or 'unknown'}" in message
        assert "SGLang's own n-gram defaults" in message
    else:
        assert not caplog.records


CACHE_ENV_BACKENDS = [
    (VLLMServer, "VLLM_CACHE_ROOT", "vllm"),
    (SGLangServer, "SGLANG_CACHE_DIR", "sglang"),
]


@pytest.mark.parametrize("backend_class, variable, subdirectory", CACHE_ENV_BACKENDS)
def test_set_cache_env_defaults_to_config_cache_dir(
    tmp_path, backend_class, variable, subdirectory
):
    backend = backend_class.__new__(backend_class)
    backend._config = types.SimpleNamespace(cache_dir=str(tmp_path))

    env = {}
    backend._set_cache_env(env, variable, subdirectory)

    expected = tmp_path / subdirectory
    assert env[variable] == str(expected)
    assert expected.is_dir()

    # Without a config there is nothing to point at.
    backend._config = None
    env = {}
    backend._set_cache_env(env, variable, subdirectory)
    assert env == {}


@pytest.mark.parametrize("backend_class, variable, subdirectory", CACHE_ENV_BACKENDS)
def test_set_cache_env_respects_user_override(
    tmp_path, backend_class, variable, subdirectory
):
    backend = backend_class.__new__(backend_class)
    backend._config = types.SimpleNamespace(cache_dir=str(tmp_path))

    env = {variable: "/custom/cache"}
    backend._set_cache_env(env, variable, subdirectory)

    assert env[variable] == "/custom/cache"
    # Default cache dir should not be created when the user overrode it.
    assert not (tmp_path / subdirectory).exists()


def _vllm_backend_with_kv_cache(
    extended_kv_cache,
    cache_config=None,
    device_info=("cuda", None, None),
    backend_version=None,
    vram_claim=0,
    gpu_count=1,
):
    backend = VLLMServer.__new__(VLLMServer)
    backend._model = types.SimpleNamespace(
        extended_kv_cache=extended_kv_cache,
        backend_version=backend_version,
        image_name=None,
    )
    backend._model_instance = types.SimpleNamespace(cache_config=cache_config)
    backend._worker = types.SimpleNamespace(ip="10.0.0.7")
    backend._get_device_info = lambda: device_info
    backend._get_total_vram_claim = lambda: vram_claim
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(index=i) for i in range(gpu_count)
    ]
    return backend


def _vllm_backend_claiming(worker_id, instance_worker_id, vram, subordinates=()):
    """A backend reading its VRAM claim off the instance record, from the
    perspective of one of the workers the instance is placed on."""
    backend = VLLMServer.__new__(VLLMServer)
    backend._worker = types.SimpleNamespace(id=worker_id)
    backend._model_instance = types.SimpleNamespace(
        worker_id=instance_worker_id,
        computed_resource_claim=types.SimpleNamespace(vram=vram),
        distributed_servers=types.SimpleNamespace(
            subordinate_workers=[
                types.SimpleNamespace(
                    worker_id=sub_worker_id,
                    computed_resource_claim=types.SimpleNamespace(vram=sub_vram),
                )
                for sub_worker_id, sub_vram in subordinates
            ]
        ),
    )
    return backend


def _shared_kv_cache_config():
    return ExtendedKVCacheConfig(
        enabled=True,
        mode=KVCacheModeEnum.SHARED,
        cache_service_id=3,
        ram_size=8,
        chunk_size=256,
    )


def test_vllm_shared_kv_cache_injected_applies_snapshot_env_and_args():
    snapshot = CacheConfigSnapshot(
        cache_service_id=3,
        env={"MOONCAKE_CONFIG_PATH": "/etc/mooncake.json"},
        args=["--kv-transfer-config", '{"kv_connector":"MooncakeConnector"}'],
        injected=True,
    )
    backend = _vllm_backend_with_kv_cache(
        _shared_kv_cache_config(), cache_config=snapshot
    )

    env = {}
    backend._set_lmcache_env(env)
    # Snapshot env is applied verbatim; local CPU cache sizing does not fire.
    assert env == {"MOONCAKE_CONFIG_PATH": "/etc/mooncake.json"}
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE" not in env
    assert "LMCACHE_CHUNK_SIZE" not in env

    # The snapshot is applied verbatim — provider-specific knowledge
    # (e.g. LMCache's --shutdown-timeout) lives in the catalog args.
    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"MooncakeConnector"}',
    ]


@pytest.mark.parametrize(
    "cache_config",
    [
        None,
        CacheConfigSnapshot(
            cache_service_id=3, injected=False, reason="Cache service is not running."
        ),
    ],
)
def test_vllm_shared_kv_cache_not_injected_starts_without_cache(cache_config):
    backend = _vllm_backend_with_kv_cache(
        _shared_kv_cache_config(), cache_config=cache_config
    )

    env = {}
    backend._set_lmcache_env(env)
    assert env == {}

    assert backend._build_extended_kv_cache_arguments(None) == []


def test_vllm_shared_kv_cache_has_no_worker_vendor_gate():
    """Shared mode applies the snapshot on any accelerator: the catalog's
    framework-scoped integrations are the single accelerator gate (an
    unsupported framework never gets an injected snapshot), so the worker
    must not second-guess it and drop half the injection."""
    snapshot = CacheConfigSnapshot(
        cache_service_id=3,
        env={"PYTHONHASHSEED": "0"},
        args=["--kv-transfer-config", '{"kv_connector":"LMCacheMPConnector"}'],
        injected=True,
    )
    backend = _vllm_backend_with_kv_cache(
        _shared_kv_cache_config(), cache_config=snapshot
    )
    backend._get_device_info = lambda: ("cann", None, None)

    env = {}
    backend._set_lmcache_env(env)
    assert env == {"PYTHONHASHSEED": "0"}
    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheMPConnector"}',
    ]


def test_vllm_legacy_local_kv_cache_behavior_unchanged():
    # A config without an explicit mode is local: LMCache env sizing and the
    # LMCache connector args apply as before.
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_size=4, chunk_size=256)
    )

    env = {}
    backend._set_lmcache_env(env)
    assert env == {
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "4",
    }

    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
    ]


def test_vllm_local_kv_cache_ram_ratio_sizes_from_vram_claim():
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_ratio=1.5),
        vram_claim=8 * 1024**3,
    )

    env = {}
    backend._set_lmcache_env(env)
    assert env == {"LMCACHE_MAX_LOCAL_CPU_SIZE": "12"}


def test_vllm_ascend_local_kv_cache_wires_npu_offload_connector():
    """Ascend has no LMCache build, so local mode carries its sizing in
    vllm-ascend's own connector instead of the LMCache variables. The
    configured size is the worker's, so 4 GiB over 2 cards is 2 GiB a rank."""
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_size=4, chunk_size=256),
        device_info=("cann", None, "Ascend910B3"),
        backend_version="0.23.0",
        gpu_count=2,
    )

    env = {}
    backend._set_lmcache_env(env)
    assert env == {}

    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":2147483648}}',
    ]


def test_vllm_ascend_local_kv_cache_ram_ratio_sizes_from_vram_claim():
    """The ratio scales this worker's claim, and the per-rank share carries
    it independently of the engine's parallelism."""
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_ratio=2.0),
        device_info=("cann", None, "Ascend910B3"),
        backend_version="0.23.0",
        vram_claim=6 * 1024**3,
        gpu_count=4,
    )

    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":3221225472}}',
    ]


def test_vllm_ascend_local_kv_cache_without_a_size_leaves_the_capacity_default():
    """Neither size nor ratio: the connector still runs, on its own default,
    the way LMCache does on the CUDA path."""
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_ratio=None, ram_size=None),
        device_info=("cann", None, "Ascend910B3"),
        backend_version="0.23.0",
    )

    assert backend._build_extended_kv_cache_arguments(None) == [
        "--kv-transfer-config",
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{}}',
    ]


@pytest.mark.parametrize("worker_id", [1, 2])
def test_vllm_total_vram_claim_is_the_claim_of_the_worker_reading_it(worker_id):
    backend = _vllm_backend_claiming(
        worker_id=worker_id,
        instance_worker_id=1,
        vram={0: 4 * 1024**3, 1: 4 * 1024**3},
        subordinates=[(2, {0: 2 * 1024**3})],
    )

    expected = 8 * 1024**3 if worker_id == 1 else 2 * 1024**3
    assert backend._get_total_vram_claim() == expected


def test_vllm_total_vram_claim_tolerates_a_claim_without_vram():
    """computed_resource_claim.vram is optional; a deployment sized by
    ram_size must not trip over an instance that has none."""
    backend = _vllm_backend_claiming(worker_id=1, instance_worker_id=1, vram=None)

    assert backend._get_total_vram_claim() == 0


@pytest.mark.parametrize(
    "device_info, backend_version",
    [
        # 310P carries no KV offload component in the runner image.
        (("cann", None, "Ascend310P3"), "0.23.0"),
        # The NPU connector predates neither: it lands in vLLM Ascend 0.21.0.
        (("cann", None, "Ascend910B3"), "0.20.0"),
    ],
)
def test_vllm_ascend_local_kv_cache_unsupported_target_injects_nothing(
    device_info, backend_version
):
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_size=4),
        device_info=device_info,
        backend_version=backend_version,
    )

    env = {}
    backend._set_lmcache_env(env)
    assert env == {}

    assert backend._build_extended_kv_cache_arguments(None) == []


def test_vllm_local_kv_cache_on_other_vendor_injects_nothing():
    backend = _vllm_backend_with_kv_cache(
        ExtendedKVCacheConfig(enabled=True, ram_size=4),
        device_info=("musa", None, None),
    )

    assert backend._build_extended_kv_cache_arguments(None) == []


def test_sglang_shared_kv_cache_disables_hicache_arguments():
    backend = SGLangServer.__new__(SGLangServer)
    backend._model = types.SimpleNamespace(extended_kv_cache=_shared_kv_cache_config())

    assert backend._get_hicache_arguments() == []


def test_sglang_local_kv_cache_hicache_arguments_unchanged():
    backend = SGLangServer.__new__(SGLangServer)
    backend._model = types.SimpleNamespace(
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, ram_size=8, chunk_size=64, ram_ratio=None
        )
    )

    assert backend._get_hicache_arguments() == [
        "--enable-hierarchical-cache",
        "--page-size",
        "64",
        "--hicache-size",
        "8",
    ]


def test_sglang_local_kv_cache_has_no_accelerator_gate():
    """SGLang's hierarchical cache is engine-native on every framework it
    ships for — on CANN it picks the Ascend IO backend and memory layout
    itself — so the flags carry no accelerator condition, unlike the vLLM
    connectors."""
    backend = SGLangServer.__new__(SGLangServer)
    backend._model = types.SimpleNamespace(
        extended_kv_cache=ExtendedKVCacheConfig(enabled=True, ram_ratio=3.0)
    )
    backend._get_device_info = lambda: ("cann", None, "Ascend910B3")

    assert backend._get_hicache_arguments() == [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "3.0",
    ]


def test_vllm_command_args_include_late_system_flags_as_injected():
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        model_name="llm",
        distributed_servers=None,
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
        mounted_loras=None,
    )
    backend._model = types.SimpleNamespace(
        name="llm",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
        backend_version=None,
        image_name=None,
        categories=[],
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_speculative_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]

    arguments, injected = backend._build_command_args(port=4000, is_distributed=False)

    assert arguments[-6:] == [
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--served-model-name",
        "llm",
    ]
    assert injected == [
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--served-model-name",
        "llm",
    ]


def test_vllm_command_args_exclude_user_backend_parameters_from_injected():
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        model_name="llm",
        distributed_servers=None,
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
        mounted_loras=None,
    )
    backend._model = types.SimpleNamespace(
        name="llm",
        backend=BackendEnum.VLLM,
        backend_parameters=["--host", "0.0.0.0", "--temperature", "0.2"],
        backend_version=None,
        image_name=None,
        categories=[],
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_speculative_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]

    arguments, injected = backend._build_command_args(port=4000, is_distributed=False)

    assert "--temperature" in arguments
    assert "--temperature" not in injected
    assert "--host" not in injected
    assert injected == ["--port", "4000", "--served-model-name", "llm"]


@pytest.mark.parametrize(
    "run_command, served_model_args",
    [
        (None, []),
        ("--model-path {{model_path}}", []),
        (
            "--model-path {{model_path}} --served-model-name custom-name",
            ["--served-model-name", "custom-name"],
        ),
        (
            "--model-path {{model_path}} --served-model-name=custom-name",
            ["--served-model-name=custom-name"],
        ),
    ],
)
def test_sglang_command_args_include_model_and_late_system_flags_as_injected(
    run_command, served_model_args
):
    backend = SGLangServer.__new__(SGLangServer)
    backend.inference_backend = InferenceBackend(
        backend_name=BackendEnum.SGLANG,
        default_version="0.5.21",
        version_configs={},
    )
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        gpu_indexes=[],
        model_name="llm-deployment",
        port=4000,
        ports=[4000],
        computed_resource_claim=None,
        mounted_loras=None,
        distributed_servers=None,
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=[],
        backend_version=None,
        run_command=run_command,
        env={"GPUSTACK_DISABLE_METRICS": "1"},
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_model_architecture = lambda: []
    backend._get_speculative_arguments = lambda: []
    backend._get_hicache_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(index=0, vendor="NVIDIA", arch_family=None)
    ]

    arguments, injected = backend._build_command_args(
        port=4000,
        is_distributed=False,
        is_distributed_leader=False,
    )

    expected = [
        "--model-path",
        "/models/llm",
        *served_model_args,
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
    ]
    if not served_model_args:
        expected.extend(["--served-model-name", "llm-deployment"])

    assert arguments == expected
    assert injected == expected


@pytest.mark.parametrize("alias", ["tp", "dp"])
def test_sglang_short_alias_prevents_conflicting_auto_parallelism_arguments(alias):
    model_instance = types.SimpleNamespace(gpu_indexes=[0, 1])

    assert (
        get_sglang_auto_parallelism_arguments(
            [f"--{alias}", "2"], model_instance, is_distributed=False
        )
        == []
    )


def test_vox_box_command_args_return_injected_parameters():
    backend = VoxBoxServer.__new__(VoxBoxServer)
    backend.inference_backend = None
    backend._model_path = "/models/audio"
    backend._config = types.SimpleNamespace(data_dir="/var/lib/gpustack")
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(gpu_indexes=[1])
    backend._model = types.SimpleNamespace(backend_parameters=[], backend_version=None)

    _, injected = backend._build_command_args(port=4000)

    assert injected == [
        "--model",
        "/models/audio",
        "--data-dir",
        "/var/lib/gpustack",
        "--host",
        "192.168.50.10",
        "--port",
        "4000",
        "--device",
        "cuda:1",
    ]


def test_custom_command_args_return_injected_parameters_after_entrypoint():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        ports=[4000], gpu_indexes=None, gpu_type=None, distributed_servers=None
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=["--temperature", "0.2"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="python -m custom.launch --model-path {{model_path}} --port {{port}}",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: (
            "python -m custom.launch --model-path /models/custom --port 4000"
        )
    )

    arguments, injected = backend._build_command_args()

    assert arguments[-2:] == ["--temperature", "0.2"]
    assert injected == ["--model-path", "/models/custom", "--port", "4000"]


def test_custom_command_args_include_short_flags_as_injected():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        ports=[4000], gpu_indexes=None, gpu_type=None, distributed_servers=None
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=["-u", "1"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="custom-server -s 0.0.0.0 -t 4",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: "custom-server -s 0.0.0.0 -t 4"
    )

    _, injected = backend._build_command_args()

    assert injected == ["-s", "0.0.0.0", "-t", "4"]


def test_injected_parameters_start_at_zero_with_explicit_container_entrypoint():
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        ports=[4000], gpu_indexes=None, gpu_type=None, distributed_servers=None
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=["-u", "1"],
        backend_version=None,
        env={},
        name="custom-model",
        run_command="-m /models/custom -t 4",
    )
    backend.inference_backend = types.SimpleNamespace(
        replace_command_param=lambda **_: "-m /models/custom -t 4"
    )

    _, injected = backend._build_command_args(entrypoint=["llama-server"])

    assert injected == ["-m", "/models/custom", "-t", "4"]


@pytest.mark.parametrize(
    "default_entrypoint, version_entrypoint, default_run_command, expected_entrypoint, expected_injected",
    [
        (
            "llama-server",
            None,
            "-m {{model_path}} -p {{port}}",
            ["llama-server"],
            ["-m", "/models/custom", "-p", "4000"],
        ),
        (
            "unused-entrypoint",
            "python -m custom.launch",
            "--model-path {{model_path}} --port {{port}}",
            ["python", "-m", "custom.launch"],
            ["--model-path", "/models/custom", "--port", "4000"],
        ),
    ],
)
def test_custom_backend_configured_entrypoint_injected_parameters(
    default_entrypoint,
    version_entrypoint,
    default_run_command,
    expected_entrypoint,
    expected_injected,
):
    backend = CustomServer.__new__(CustomServer)
    backend._model_path = "/models/custom"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        ports=[4000], gpu_indexes=None, gpu_type=None, distributed_servers=None
    )
    backend._model = types.SimpleNamespace(
        backend_parameters=["--user-param", "1"],
        backend_version="cpu",
        env={},
        name="custom-model",
        run_command=None,
    )
    backend.inference_backend = InferenceBackend(
        backend_name="custom-entrypoint-backend",
        default_version="cpu",
        default_entrypoint=default_entrypoint,
        default_run_command=default_run_command,
        version_configs=VersionConfigDict(
            root={
                "cpu": VersionConfig(
                    image_name="custom/backend:cpu",
                    entrypoint=version_entrypoint,
                    custom_framework="cpu",
                )
            }
        ),
    )

    entrypoint = backend.inference_backend.get_container_entrypoint("cpu")
    arguments, injected = backend._build_command_args(entrypoint=entrypoint)

    assert entrypoint == expected_entrypoint
    assert arguments[-2:] == ["--user-param", "1"]
    assert injected == expected_injected


def _gpu_placeholder_backend() -> InferenceBackend:
    return InferenceBackend(
        backend_name="gpu-placeholder-backend",
        default_version="v1",
        default_run_command="serve {{model_path}} --gpus {{gpu_count}} --gpu-ids {{gpu_ids}} --port {{port}}",
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(
                    image_name="custom/backend:v1",
                    custom_framework="cuda",
                )
            }
        ),
    )


@pytest.mark.parametrize(
    "gpu_count, gpu_ids, expected_gpus, expected_gpu_ids",
    [
        (None, None, "", ""),
        (0, [], "0", ""),
        (1, [2], "1", "2"),
        (3, [0, 1, 2], "3", "0,1,2"),
    ],
)
def test_replace_command_param_substitutes_gpu_placeholders(
    gpu_count, gpu_ids, expected_gpus, expected_gpu_ids
):
    backend = _gpu_placeholder_backend()

    command = backend.replace_command_param(
        version="v1",
        model_path="/models/custom",
        port=4000,
        gpu_count=gpu_count,
        gpu_ids=gpu_ids,
    )

    assert command == (
        f"serve /models/custom --gpus {expected_gpus} "
        f"--gpu-ids {expected_gpu_ids} --port 4000"
    )


def test_replace_command_param_substitutes_all_placeholders_together():
    backend = InferenceBackend(
        backend_name="all-placeholder-backend",
        default_version="v1",
        default_run_command=(
            "serve {{model_path}} --port {{port}} --served-model-name {{model_name}} "
            "--host {{worker_ip}} --gpus {{gpu_count}} --gpu-ids {{gpu_ids}}"
        ),
        version_configs=VersionConfigDict(
            root={
                "v1": VersionConfig(
                    image_name="custom/backend:v1",
                    custom_framework="cuda",
                )
            }
        ),
    )

    command = backend.replace_command_param(
        version="v1",
        model_path="/models/custom",
        port=4000,
        worker_ip="192.168.50.10",
        model_name="custom-model",
        gpu_count=2,
        gpu_ids=[0, 1],
    )

    assert command == (
        "serve /models/custom --port 4000 --served-model-name custom-model "
        "--host 192.168.50.10 --gpus 2 --gpu-ids 0,1"
    )


@pytest.mark.parametrize(
    "command, command_args, command_script, expected_command, expected_args",
    [
        # No script + non-empty command (vLLM/SGLang main path): merge into
        # command so the image ENTRYPOINT is fully overridden.
        (
            ["vllm", "serve"],
            ["/models/llm", "--port", "8000"],
            None,
            ["vllm", "serve", "/models/llm", "--port", "8000"],
            None,
        ),
        # No script + command is None (vox_box/ascend with no entrypoint):
        # keep args as-is, must not raise TypeError on None + list.
        (
            None,
            ["vox-box", "start", "--model", "/models/audio"],
            None,
            None,
            ["vox-box", "start", "--model", "/models/audio"],
        ),
        # No script + empty command: same as None.
        ([], ["mindieservice_daemon"], None, None, ["mindieservice_daemon"]),
        # Script + non-empty command: script becomes entrypoint, command is
        # prepended to its args.
        (
            ["vllm", "serve"],
            ["/models/llm"],
            "setup.sh",
            None,
            ["vllm", "serve", "/models/llm"],
        ),
        # Script + command is None: just pass the args through under the script.
        (None, ["vox-box", "start"], "setup.sh", None, ["vox-box", "start"]),
    ],
)
def test_override_entrypoint(
    command, command_args, command_script, expected_command, expected_args
):
    result_command, result_args = InferenceServer._override_entrypoint(
        command, command_args, command_script
    )

    assert result_command == expected_command
    assert result_args == expected_args


@pytest.mark.parametrize(
    "arguments, entrypoint, expected_start_index",
    [
        # Empty arguments.
        ([], None, 0),
        # vLLM/SGLang style: executable carried in command (not args here),
        # first option marks the start.
        (["vllm", "serve", "/models/llm", "--port", "8000"], None, 3),
        # python -m module ...: skip the launcher and module, start at 3.
        (["python", "-m", "vllm", "--host", "x"], None, 3),
        # With an explicit container entrypoint, the python -m heuristic is
        # skipped; the first dash-prefixed token wins ("-m" at index 1).
        (["python", "-m", "vllm", "--host", "x"], ["llama-server"], 1),
        # Entrypoint set, model path then first option.
        (["/models/llm", "--port", "8000"], ["vllm", "serve"], 1),
        # First token already an option.
        (["--port", "8000"], None, 0),
    ],
)
def test_get_backend_parameter_start_index(arguments, entrypoint, expected_start_index):
    assert (
        InferenceServer._get_backend_parameter_start_index(arguments, entrypoint)
        == expected_start_index
    )


def _write_adapter_config(path, data) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _lora(lora_name, path):
    return types.SimpleNamespace(lora_name=lora_name, path=path)


def _arg_value(arguments, key):
    """Return the value following ``key`` in a flat argument list."""
    for i, token in enumerate(arguments):
        if token == key:
            return arguments[i + 1]
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    return None


def test_read_lora_max_rank_basic(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"r": 64})
    assert read_lora_max_rank([adapter]) == 64


def test_read_lora_max_rank_prefers_rank_pattern(tmp_path):
    adapter = _write_adapter_config(
        tmp_path / "a", {"r": 16, "rank_pattern": {"layers.0.attn": 128}}
    )
    assert read_lora_max_rank([adapter]) == 128


def test_read_lora_max_rank_across_multiple_adapters(tmp_path):
    a = _write_adapter_config(tmp_path / "a", {"r": 16})
    b = _write_adapter_config(tmp_path / "b", {"r": 64})
    assert read_lora_max_rank([a, b]) == 64


def test_read_lora_max_rank_missing_or_broken_returns_none(tmp_path):
    missing = str(tmp_path / "missing")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "adapter_config.json").write_text("{not json", encoding="utf-8")
    no_rank = _write_adapter_config(tmp_path / "no_rank", {"base_model_name": "x"})
    assert read_lora_max_rank([missing, str(broken), no_rank]) is None
    assert read_lora_max_rank([None, ""]) is None


def test_vllm_injects_max_lora_rank(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"r": 64})
    arguments = []
    extend_vllm_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        "base",
        backend_parameters=None,
    )
    assert _arg_value(arguments, "--max-lora-rank") == "64"


def test_vllm_rounds_up_to_allowed_choice(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"r": 24})
    arguments = []
    extend_vllm_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        "base",
        backend_parameters=None,
    )
    assert _arg_value(arguments, "--max-lora-rank") == "32"


def test_vllm_respects_user_max_lora_rank(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"r": 64})
    arguments = []
    extend_vllm_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        "base",
        backend_parameters=["--max-lora-rank", "128"],
    )
    # User value lives in backend_parameters and is appended later; we must not
    # inject our own into arguments.
    assert "--max-lora-rank" not in arguments


def test_vllm_skips_when_rank_unreadable(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"base_model_name": "base"})
    arguments = []
    extend_vllm_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        "base",
        backend_parameters=None,
    )
    assert "--max-lora-rank" not in arguments


def test_sglang_injects_raw_max_lora_rank(tmp_path):
    # SGLang has no fixed choice set: the raw max rank is used as-is.
    adapter = _write_adapter_config(tmp_path / "a", {"r": 24})
    arguments = []
    extend_sglang_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        backend_parameters=None,
    )
    assert _arg_value(arguments, "--max-lora-rank") == "24"


@pytest.mark.parametrize(
    'value', ["/models/worker's tokenizer", '/models/tokenizer --lora-paths=decoy']
)
def test_sglang_lora_probe_preserves_existing_argv(tmp_path, value):
    adapter = _write_adapter_config(tmp_path / 'a', {'r': 24})
    arguments = [f'--tokenizer-path={value}']
    extend_sglang_mounted_lora_arguments(
        arguments, [_lora('base:a', adapter)], backend_parameters=None
    )
    assert arguments[0] == f'--tokenizer-path={value}'
    assert _arg_value(arguments, '--lora-paths') == f'a={adapter}'


def test_sglang_respects_user_max_lora_rank(tmp_path):
    adapter = _write_adapter_config(tmp_path / "a", {"r": 64})
    arguments = []
    extend_sglang_mounted_lora_arguments(
        arguments,
        [_lora("base:a", adapter)],
        backend_parameters=["--max-lora-rank", "128"],
    )
    assert "--max-lora-rank" not in arguments


def test_round_up_vllm_lora_rank():
    assert _round_up_vllm_lora_rank(8) == 8
    assert _round_up_vllm_lora_rank(24) == 32
    assert _round_up_vllm_lora_rank(64) == 64
    assert _round_up_vllm_lora_rank(200) == 256
    assert _round_up_vllm_lora_rank(1024) == 1024  # beyond known set: pass through


def _make_versioned_runner(backend_version, docker_image):
    """Build a minimal BackendVersionedRunner-shaped stub for _resolve_image."""
    platform_entry = types.SimpleNamespace(docker_image=docker_image)
    service_version = types.SimpleNamespace(
        version="0.10.2", platforms=[platform_entry]
    )
    service = types.SimpleNamespace(versions=[service_version])
    variant = types.SimpleNamespace(services=[service])
    return types.SimpleNamespace(version=backend_version, variants=[variant])


@pytest.mark.parametrize(
    "runtime_version, expected_image",
    [
        # Host runtime matches a runner exactly: pick it.
        ("12.8", "gpustack/runner:cuda12.8-vllm0.10.2"),
        # Host runtime lower than every runner: never cross major, fall back to
        # the newest runner sharing the host major (12.9, not the oldest 12.8).
        ("12.6", "gpustack/runner:cuda12.9-vllm0.10.2"),
        # No runner shares the host major: fall back to the oldest available.
        ("11.5", "gpustack/runner:cuda12.8-vllm0.10.2"),
        # Host runtime is higher than all: pick the newest that is <= it.
        ("13.5", "gpustack/runner:cuda13.0-vllm0.10.2"),
        # Runtime version undetectable: fall back to the latest.
        (None, "gpustack/runner:cuda13.0-vllm0.10.2"),
    ],
)
def test_resolve_image_fallback_matches_host_major(
    runtime_version, expected_image, monkeypatch
):
    import gpustack.worker.backends.base as base_module

    # gpustack-runner returns backend versions newest-first.
    runner = types.SimpleNamespace(
        versions=[
            _make_versioned_runner("13.0", "gpustack/runner:cuda13.0-vllm0.10.2"),
            _make_versioned_runner("12.9", "gpustack/runner:cuda12.9-vllm0.10.2"),
            _make_versioned_runner("12.8", "gpustack/runner:cuda12.8-vllm0.10.2"),
        ]
    )
    # merged_backend_runners takes the override rows positionally, then the filters.
    monkeypatch.setattr(
        base_module, "merged_backend_runners", lambda *_, **__: [runner]
    )

    server = VLLMServer.__new__(VLLMServer)
    server._model = types.SimpleNamespace(
        image_name=None, backend="vllm", backend_version=None
    )
    server.inference_backend = None
    server._get_device_info = lambda: ("cuda", runtime_version, None)
    # The real fetch needs a clientset, and its failure is fatal by design.
    server._fetch_runner_overrides = lambda: []

    image_name, _ = server._resolve_image()
    assert image_name == expected_image


@pytest.mark.parametrize(
    "backend_version, expected_service_version, expected_with_deprecated",
    [
        # Auto: no version filter, deprecated runners stay hidden.
        (None, None, False),
        # Blank version is stored by legacy/migrated data and API clients. It is
        # shown as "Auto" in the UI and has to resolve like None, not like an
        # exact version filter that matches no runner.
        ("", None, False),
        # An explicit version is still passed through verbatim.
        ("0.10.2", "0.10.2", True),
    ],
)
def test_resolve_image_treats_blank_backend_version_as_auto(
    backend_version, expected_service_version, expected_with_deprecated, monkeypatch
):
    import gpustack.worker.backends.base as base_module

    captured = {}

    # merged_backend_runners takes the override rows positionally, then the filters.
    def fake_merged_backend_runners(_overrides, **kwargs):
        captured.update(kwargs)
        return [
            types.SimpleNamespace(
                versions=[
                    _make_versioned_runner(
                        "13.0", "gpustack/runner:cuda13.0-vllm0.10.2"
                    )
                ]
            )
        ]

    monkeypatch.setattr(
        base_module, "merged_backend_runners", fake_merged_backend_runners
    )

    server = VLLMServer.__new__(VLLMServer)
    server._model = types.SimpleNamespace(
        image_name=None, backend="vllm", backend_version=backend_version
    )
    server.inference_backend = None
    server._get_device_info = lambda: ("cuda", "13.0", None)
    # The real fetch needs a clientset, and its failure is fatal by design.
    server._fetch_runner_overrides = lambda: []

    image_name, _ = server._resolve_image()

    assert image_name == "gpustack/runner:cuda13.0-vllm0.10.2"
    assert captured["service_version"] == expected_service_version
    assert captured["with_deprecated"] is expected_with_deprecated


class _StubServer(InferenceServer):
    """Concrete InferenceServer so base methods can be exercised in isolation."""

    def start(self):  # pragma: no cover - not used by these tests
        raise NotImplementedError


@pytest.mark.parametrize("text_length,expected", [(None, 65536), (8192, 32768)])
def test_worker_context_length_uses_root_only_when_text_length_is_missing(
    text_length, expected
):
    config = types.SimpleNamespace(
        max_position_embeddings=32768,
        rope_scaling={"type": "linear", "factor": 2},
        text_config=types.SimpleNamespace(
            num_attention_heads=16,
            max_position_embeddings=text_length,
            rope_scaling={"type": "linear", "factor": 4},
        ),
    )
    server = _StubServer.__new__(_StubServer)
    server._get_pretrained_config = lambda: config
    assert server._derive_max_model_len() == expected


@pytest.mark.parametrize(
    "image,expected",
    [
        ("gpustack/runner:cuda12.9-vllm0.10.2", "12.9"),
        (
            "registry.example.com/gpustack/runner:cuda12.8-vllm0.10.2-linux-amd64",
            "12.8",
        ),
        (
            "registry.example.com/cuda11.8/gpustack/runner:cuda12.8-vllm0.10.2-linux-amd64",
            "12.8",
        ),
        ("gpustack/runner:cuda13.0-sglang0.4", "13.0"),
        ("gpustack/runner:rocm6.2-vllm0.10.2", None),
        ("myregistry/custom-vllm:latest", None),
        ("", None),
    ],
)
def test_parse_image_cuda_version(image, expected):
    assert _parse_image_cuda_version(image) == expected


def _make_cuda_compat_server(host_cuda, image, backend="cuda", model_env=None):
    server = _StubServer.__new__(_StubServer)
    server._model = types.SimpleNamespace(env=model_env)
    server._get_device_info = lambda: (backend, host_cuda, None)
    server._resolve_image = lambda: (image, None)
    return server


@pytest.mark.parametrize(
    "host_cuda,image,backend,expected",
    [
        # Image minor higher than host driver, same major -> route through MVC.
        ("12.8", "gpustack/runner:cuda12.9-vllm0.10.2", "cuda", True),
        # Equal minor: cuda-compat is never activated, nothing to do.
        ("12.8", "gpustack/runner:cuda12.8-vllm0.10.2", "cuda", False),
        # Host newer than image: plain backward compatibility, nothing to do.
        ("12.8", "gpustack/runner:cuda12.6-vllm0.10.2", "cuda", False),
        # Cross major is not MVC-safe; must not trigger.
        ("12.8", "gpustack/runner:cuda13.0-vllm0.10.2", "cuda", False),
        # Non-CUDA vendor.
        ("6.2", "gpustack/runner:rocm6.3-vllm0.10.2", "rocm", False),
        # Host CUDA version undetected.
        (None, "gpustack/runner:cuda12.9-vllm0.10.2", "cuda", False),
        # Custom image whose tag does not encode a CUDA version.
        ("12.8", "myregistry/custom:latest", "cuda", False),
    ],
)
def test_should_disable_cuda_compat(monkeypatch, host_cuda, image, backend, expected):
    # This case matrix exercises the version condition; force the opt-in switch on.
    monkeypatch.setattr("gpustack.envs.ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY", True)
    server = _make_cuda_compat_server(host_cuda, image, backend)
    assert server._should_disable_cuda_compat() is expected


@pytest.mark.parametrize(
    "global_enabled,model_env,expected",
    [
        # Enabled globally, no model override -> triggers.
        (True, None, True),
        # Disabled globally (the default), no model override -> off.
        (False, None, False),
        # Model-level override wins over the global default (opt in this model).
        (False, {_KNOB: "true"}, True),
        # Model-level override wins over the global default (opt out this model).
        (True, {_KNOB: "false"}, False),
    ],
)
def test_cuda_compat_switch_is_multi_level(
    monkeypatch, global_enabled, model_env, expected
):
    # Host/image are fixed to a triggering pair so only the switch varies.
    monkeypatch.setattr(
        "gpustack.envs.ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY", global_enabled
    )
    server = _make_cuda_compat_server(
        "12.8",
        "gpustack/runner:cuda12.9-vllm0.10.2",
        model_env=model_env,
    )
    assert server._should_disable_cuda_compat() is expected


def _make_host_ipc_server(cache_config=None, model_env=None):
    server = _StubServer.__new__(_StubServer)
    server._model = types.SimpleNamespace(env=model_env)
    server._model_instance = types.SimpleNamespace(cache_config=cache_config)
    return server


@pytest.mark.parametrize(
    "cache_config,model_env,global_env,expected",
    [
        # No shared cache attached -> private /dev/shm (shm_size applies).
        (None, None, None, False),
        # Attached and injected -> host IPC for the CUDA-IPC transfer path.
        (types.SimpleNamespace(injected=True), None, None, True),
        # Attached but degraded (not injected) -> nothing to transfer.
        (types.SimpleNamespace(injected=False), None, None, False),
        # Model env overrides the derivation either way.
        (
            types.SimpleNamespace(injected=True),
            {"GPUSTACK_HOST_IPC": "false"},
            None,
            False,
        ),
        (None, {"GPUSTACK_HOST_IPC": "true"}, None, True),
        # Global env overrides the derivation (e.g. PodSecurity clusters)
        # but yields to the model env.
        (types.SimpleNamespace(injected=True), None, "false", False),
        (None, {"GPUSTACK_HOST_IPC": "true"}, "false", True),
    ],
)
def test_host_ipc_follows_cache_attachment(
    monkeypatch, cache_config, model_env, global_env, expected
):
    monkeypatch.setattr("gpustack.envs.HOST_IPC", global_env)
    server = _make_host_ipc_server(cache_config=cache_config, model_env=model_env)
    assert server._host_ipc_enabled() is expected


def _script_server(should_disable_compat, cache_files=None):
    server = _StubServer.__new__(_StubServer)
    server._should_disable_cuda_compat = lambda: should_disable_compat
    server._model_instance = types.SimpleNamespace(
        cache_config=(
            CacheConfigSnapshot(cache_service_id=5, injected=True, files=cache_files)
            if cache_files
            else None
        )
    )
    return server


def test_serving_command_script_bakes_in_compat_removal():
    script = _script_server(True)._get_serving_command_script({})
    assert script is not None
    assert "rm -rf /usr/local/cuda/compat" in script
    assert "ldconfig" in script
    # Non-root containers can't remove compat; the script warns instead of failing silently.
    assert "Warning: failed to remove /usr/local/cuda/compat" in script
    assert script.rstrip().endswith('exec "$@"')


def test_serving_command_script_writes_cache_injection_files():
    """A shared cache connector that reads a config file (e.g. Mooncake's
    MOONCAKE_CONFIG_PATH JSON) gets it written by the serving script
    before the engine starts, verbatim via a quoted heredoc."""
    files = {"/tmp/gpustack-mooncake.json": '{"mode": "standalone-store"}'}
    script = _script_server(False, cache_files=files)._get_serving_command_script({})
    assert script is not None
    assert "cat > '/tmp/gpustack-mooncake.json' <<'GPUSTACK_CACHE_FILE_EOF'" in script
    assert '{"mode": "standalone-store"}' in script
    assert script.rstrip().endswith('exec "$@"')


def test_serving_command_script_none_without_triggers():
    assert _script_server(False)._get_serving_command_script({}) is None
    # Explicit opt-out wins even when compat removal was requested.
    assert (
        _script_server(True)._get_serving_command_script(
            {"GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED": "1"}
        )
        is None
    )


def test_serving_command_script_pypi_has_no_compat_block():
    script = _script_server(False)._get_serving_command_script(
        {"PYPI_PACKAGES_INSTALL": "some-package"}
    )
    assert script is not None
    assert "PYPI_PACKAGES_INSTALL" in script
    assert "/usr/local/cuda/compat" not in script


def test_configured_env_injects_disable_require_only(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _StubServer.__new__(_StubServer)
    server._model = types.SimpleNamespace(env=None)
    server._should_disable_cuda_compat = lambda: True

    env = server._get_configured_env()

    assert env["NVIDIA_DISABLE_REQUIRE"] == "1"
    assert "GPUSTACK_DISABLE_CUDA_COMPAT" not in env


def test_configured_env_respects_user_override(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _StubServer.__new__(_StubServer)
    # User explicitly pins NVIDIA_DISABLE_REQUIRE; we must not clobber it.
    server._model = types.SimpleNamespace(env={"NVIDIA_DISABLE_REQUIRE": "0"})
    server._should_disable_cuda_compat = lambda: True

    env = server._get_configured_env()

    assert env["NVIDIA_DISABLE_REQUIRE"] == "0"


def test_configured_env_no_injection_when_not_triggered(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.backends.base.filter_env_vars", lambda _env: {}
    )
    server = _StubServer.__new__(_StubServer)
    server._model = types.SimpleNamespace(env=None)
    server._should_disable_cuda_compat = lambda: False

    env = server._get_configured_env()

    assert "NVIDIA_DISABLE_REQUIRE" not in env


def _gpu(vendor: str, arch_family=None):
    return types.SimpleNamespace(vendor=vendor, arch_family=arch_family)


@pytest.mark.parametrize(
    "name, devices, expected",
    [
        # The regression this guards: `all()` is vacuously true over nothing,
        # so a model instance that named no device -- one scheduled by GPU
        # type rather than by device index -- used to be served as Ascend
        # 310P, and got `--enforce-eager --dtype float16` injected on top of
        # whatever accelerator it actually landed on.
        ("no device selected", [], False),
        ("single 310P", [_gpu("ascend", "Ascend310P1")], True),
        (
            "several 310P",
            [_gpu("ascend", "Ascend310P1"), _gpu("ascend", "Ascend310P3")],
            True,
        ),
        ("Ascend but not 310P", [_gpu("ascend", "Ascend910B4")], False),
        ("NVIDIA", [_gpu("nvidia")], False),
        (
            "310P mixed with another Ascend generation",
            [_gpu("ascend", "Ascend310P1"), _gpu("ascend", "Ascend910B4")],
            False,
        ),
    ],
)
def test_is_ascend_310p(name, devices, expected):
    actual = is_ascend_310p(devices)
    assert actual == expected, f"case {name} expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    "name, devices, expected",
    [
        ("no device selected", [], False),
        ("single Ascend", [_gpu("ascend", "Ascend910B4")], True),
        (
            "several Ascend",
            [_gpu("ascend", "Ascend910B4"), _gpu("ascend", "Ascend310P1")],
            True,
        ),
        ("NVIDIA", [_gpu("nvidia")], False),
        (
            "Ascend mixed with NVIDIA",
            [_gpu("ascend", "Ascend910B4"), _gpu("nvidia")],
            False,
        ),
    ],
)
def test_is_ascend(name, devices, expected):
    actual = is_ascend(devices)
    assert actual == expected, f"case {name} expected {expected}, but got {actual}"


def _make_versioned_args_server(run_command=None, backend_version=None):
    server = _StubServer.__new__(_StubServer)
    server._model = types.SimpleNamespace(
        run_command=run_command,
        backend_version=backend_version,
        env=None,
    )
    server._model_path = "/models/qwen"
    server._model_instance = types.SimpleNamespace(port=8000, model_name="qwen")
    server._worker = types.SimpleNamespace(ip="10.0.0.1")
    server._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(index=1),
        types.SimpleNamespace(index=0),
    ]
    backend = InferenceBackend(
        backend_name=BackendEnum.VLLM,
        default_version="0.11.0",
        version_configs={},
    )
    backend.version_configs = VersionConfigDict(
        root={
            "0.11.0": VersionConfig(
                image_name="runner:0.11.0", built_in_frameworks=["cuda"]
            ),
            "0.11.0-custom": VersionConfig(
                image_name="own:0.11.0",
                run_command="--from-version {{model_path}} --port {{port}}",
            ),
        }
    )
    server.inference_backend = backend
    return server


def test_a_run_command_on_the_model_replaces_the_initial_arguments():
    """The model's own run_command wins over both the backend's default
    arguments and a version config's command, and its placeholders resolve.

    vLLM and SGLang supply the executable through the container entrypoint, so
    a run_command for them carries arguments only.
    """
    default_args = ["--model", "/models/qwen", "--port", "8000"]

    # No custom command anywhere: the backend keeps its own arguments.
    server = _make_versioned_args_server()
    assert server.build_versioned_command_args(default_args) == default_args

    # A model command applies without a backend_version to hang it on.
    server = _make_versioned_args_server(
        run_command="--model {{model_path}} --port {{port}} --host {{worker_ip}} --gpus {{gpu_ids}}"
    )
    assert server.build_versioned_command_args(default_args) == [
        "--model",
        "/models/qwen",
        "--port",
        "8000",
        "--host",
        "10.0.0.1",
        "--gpus",
        "0,1",
    ]

    # A version config command is shared by every deployment on that version,
    # so the model's own command outranks it.
    server = _make_versioned_args_server(
        run_command="--from-model {{model_path}}",
        backend_version="0.11.0-custom",
    )
    assert server.build_versioned_command_args(default_args) == [
        "--from-model",
        "/models/qwen",
    ]

    # Without a model command that version config still applies.
    server = _make_versioned_args_server(backend_version="0.11.0-custom")
    assert server.build_versioned_command_args(default_args) == [
        "--from-version",
        "/models/qwen",
        "--port",
        "8000",
    ]


# --- the two writers of --kv-transfer-config meet in the built command ------ #


def _vllm_backend_for_composition(role, cache_args, pd_args):
    """A vLLM server whose two connector producers both contribute.

    Built around `_build_command_args` rather than the composer, because the
    bug this guards was in neither of them: the composition ran at a point
    where only one producer had contributed, so it was a no-op and both flags
    reached the container. Testing the composer alone cannot see that.
    """
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = None
    backend._model_path = "/models/llm"
    backend._worker = types.SimpleNamespace(ip="192.168.50.10")
    backend._model_instance = types.SimpleNamespace(
        model_name="llm",
        distributed_servers=None,
        gpu_indexes=[],
        ports=[4000],
        computed_resource_claim=None,
        mounted_loras=None,
        cache_config=None,
        role=role,
    )
    backend._model = types.SimpleNamespace(
        name="llm",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
        backend_version=None,
        image_name=None,
        categories=[],
        extended_kv_cache=None,
        speculative_config=None,
    )
    backend._derive_max_model_len = lambda: None
    backend._get_speculative_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: [
        types.SimpleNamespace(vendor="NVIDIA", arch_family=None)
    ]
    # The cache contributes with the engine arguments; the PD connector
    # arrives much later, with the user's parameters. That gap is the point.
    backend._build_extended_kv_cache_arguments = lambda ctx: list(cache_args)
    backend._flatten_backend_param = lambda: list(pd_args)
    return backend


_CACHE_JSON = '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
_PD_JSON = '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'


def _composed(arguments):
    assert arguments.count("--kv-transfer-config") == 1, "one flag, one value"
    value = json.loads(arguments[arguments.index("--kv-transfer-config") + 1])
    assert value["kv_connector"] == "MultiConnector"
    return [c["kv_connector"] for c in value["kv_connector_extra_config"]["connectors"]]


def test_a_prefill_command_asks_the_pd_connector_before_cache():
    backend = _vllm_backend_for_composition(
        "prefill",
        ["--kv-transfer-config", _CACHE_JSON],
        ["--kv-transfer-config", _PD_JSON],
    )

    arguments, _ = backend._build_command_args(port=4000, is_distributed=False)

    assert _composed(arguments) == ["NixlConnector", "LMCacheMPConnector"]


def test_a_decode_command_asks_its_own_connector_first():
    backend = _vllm_backend_for_composition(
        "decode",
        ["--kv-transfer-config", _CACHE_JSON],
        ["--kv-transfer-config", _PD_JSON],
    )

    arguments, _ = backend._build_command_args(port=4000, is_distributed=False)

    assert _composed(arguments) == ["NixlConnector", "LMCacheMPConnector"]


@pytest.mark.parametrize(
    "cache_args,pd_args",
    [
        (["--kv-transfer-config", _CACHE_JSON], []),
        ([], ["--kv-transfer-config", _PD_JSON]),
        ([], []),
    ],
)
def test_a_single_writer_leaves_the_command_alone(cache_args, pd_args):
    """This runs on every vLLM launch, so the un-composed cases must not move."""
    backend = _vllm_backend_for_composition("prefill", cache_args, pd_args)

    arguments, _ = backend._build_command_args(port=4000, is_distributed=False)

    assert (
        arguments.count("--kv-transfer-config")
        == len(cache_args) // 2 + len(pd_args) // 2
    )
    assert "MultiConnector" not in " ".join(arguments)


def _router_server(monkeypatch, engine_backend, captured=None):
    """A server as it exists mid-launch for a managed router.

    `_model` is already projected to the custom backend — that is what makes it
    launch a command instead of an engine — while `_model_spec` still holds the
    group's engine and version. The backend row it was handed is None, because
    `custom` has no row.
    """
    import gpustack.worker.backends.base as base_module

    def fake_merged(*_, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        return []

    monkeypatch.setattr(base_module, "merged_backend_runners", fake_merged)

    server = VLLMServer.__new__(VLLMServer)
    server._model = types.SimpleNamespace(
        image_name=None, backend="custom", backend_version=None
    )
    server._model_spec = types.SimpleNamespace(
        image_name=None,
        backend="vLLM",
        backend_version="0.20.2-ascend-pd-custom",
        owner_principal_id=None,
    )
    server.inference_backend = None
    server._get_device_info = lambda: ("cann", "9.0", "910b")
    server._fetch_runner_overrides = lambda: []
    server._engine_inference_backend = lambda spec: engine_backend
    return server


def test_resolve_image_for_a_router_reads_the_engine_spec_not_its_own_model(
    monkeypatch,
):
    """The custom version's image lives only on the engine's backend row.

    Resolving against the router's own projected model asks for a `custom`
    service at no version, which matches neither the row nor the runner
    catalog: the unresolved `{{runner_image}}` reaches docker as an invalid
    reference.
    """
    asked = []

    engine_backend = types.SimpleNamespace(
        backend_name="vLLM",
        get_image_name=lambda version: (
            asked.append(version),
            ("gpustack/runner:cann9.0-910b-vllm0.20.2-router", version),
        )[1],
    )
    server = _router_server(monkeypatch, engine_backend)

    image_name, version = server._resolve_image(
        backend="cann",
        spec=server._model_spec,
        inference_backend=engine_backend,
    )

    assert image_name == "gpustack/runner:cann9.0-910b-vllm0.20.2-router"
    assert version == "0.20.2-ascend-pd-custom"
    assert asked == ["0.20.2-ascend-pd-custom"]


def test_resolve_image_queries_the_runner_catalog_for_the_engine_service(
    monkeypatch,
):
    """The spec has to reach the catalog query too, not just the backend row.

    A group whose engine version *is* in the catalog never touches the row, so
    the fallback path has to be looking up `vllm` — `custom` is not a runner
    service and matches nothing.
    """
    captured = {}
    server = _router_server(monkeypatch, None, captured=captured)

    server._resolve_image(
        backend="cann",
        spec=server._model_spec,
        inference_backend=None,
    )

    assert captured["service"] == "vllm"
    assert captured["service_version"] == "0.20.2-ascend-pd-custom"


def test_resolve_image_without_a_spec_still_reads_this_server(monkeypatch):
    """The override is opt-in: every other backend's call must be unchanged."""
    captured = {}
    server = _router_server(monkeypatch, None, captured=captured)

    server._resolve_image()

    assert captured["service"] == "custom"


def test_resolve_image_honours_an_explicit_none_backend_row(monkeypatch):
    """`None` means "this model has no row", not "use mine".

    Told to use the server's own row instead, a router would resolve the
    image of whatever backend it was handed — which for the fallback custom
    row is the router's own command, not the engine's image.
    """
    own_row = types.SimpleNamespace(
        backend_name="custom",
        get_image_name=lambda version: ("wrong/image:from-the-routers-own-row", "x"),
    )
    server = _router_server(monkeypatch, None)
    server.inference_backend = own_row

    image_name, _ = server._resolve_image(
        spec=server._model_spec, inference_backend=None
    )

    assert image_name is None


def _rendered_router(**overrides):
    fields = {
        "image_name": "gpustack/runner:x",
        "run_command": "vllm-router",
        "env": {},
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "overrides, expected_fragment",
    [
        ({"image_name": "{{runner_image}}"}, "image_name={{runner_image}}"),
        ({"run_command": "vllm-router --prefill {{peers}}"}, "{{peers}}"),
        (
            {"env": {"UCX_NET_DEVICES": "{{net_device}}"}},
            "UCX_NET_DEVICES={{net_device}}",
        ),
    ],
)
def test_refuse_unrendered_router_names_the_value(overrides, expected_fragment):
    from gpustack.worker.backends.base import _refuse_unrendered_router
    from gpustack.worker.pd_injection import PDInjectionError

    instance = types.SimpleNamespace(name="qwen3-pd-router-abc12")

    with pytest.raises(PDInjectionError) as excinfo:
        _refuse_unrendered_router(_rendered_router(**overrides), instance)

    assert expected_fragment in str(excinfo.value)
    assert "qwen3-pd-router-abc12" in str(excinfo.value)


def test_refuse_unrendered_router_passes_a_fully_rendered_router():
    from gpustack.worker.backends.base import _refuse_unrendered_router

    _refuse_unrendered_router(
        _rendered_router(
            image_name="quay.io/gpustack/runner:cann9.0-910b-vllm0.20.2-router",
            run_command="vllm-router --prefill http://192.168.13.3:40056",
            env={"UCX_NET_DEVICES": "bond1"},
        ),
        types.SimpleNamespace(name="qwen3-pd-router-abc12"),
    )


_PD_CONNECTOR = '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'


def _backend_reporting_injected_parameters(parameter_format):
    """An `InferenceServer` with just enough on it to answer the report call."""
    backend = VLLMServer.__new__(VLLMServer)
    backend.inference_backend = (
        types.SimpleNamespace(parameter_format=parameter_format)
        if parameter_format is not None
        else None
    )
    backend._pd_arguments = lambda: ["--kv-transfer-config", _PD_CONNECTOR]
    return backend


@pytest.mark.parametrize(
    "parameter_format, user_parameters",
    [
        (
            ParameterFormatEnum.EQUAL,
            [f"--kv-transfer-config={_PD_CONNECTOR}", "--max-model-len=8192"],
        ),
        (
            None,
            ["--kv-transfer-config", _PD_CONNECTOR, "--max-model-len", "8192"],
        ),
    ],
)
def test_the_pd_connector_is_reported_as_injected_in_either_parameter_format(
    parameter_format, user_parameters
):
    """The connector GPUStack added is never attributed to the user.

    The role's arguments ride in front of the user's through
    `_flatten_backend_param`, which also reshapes the whole stream to the
    backend's `parameter_format` — so what comes back here is the prefix in
    that shape, and the report has to recognise it in either one.
    """
    backend = _backend_reporting_injected_parameters(parameter_format)
    arguments = ["vllm", "serve", "/models/llm"] + user_parameters + ["--port", "4000"]

    injected = backend._get_injected_backend_parameters(arguments, user_parameters)

    assert "--kv-transfer-config" in " ".join(injected)
    assert "--max-model-len" not in " ".join(injected)


@pytest.mark.parametrize("legacy_snapshot", [False, True])
@pytest.mark.parametrize(
    "should_disable_compat,env,expect_script",
    [
        (False, {}, False),
        (True, {}, True),
        (True, {"GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED": "1"}, False),
    ],
)
def test_cache_package_metadata_cannot_guard_engine_start(
    legacy_snapshot, should_disable_compat, env, expect_script
):
    snapshot_data = {
        "cache_service_id": 5,
        "injected": True,
        "files": {},
        "required_packages": {"lmcache": ">=99999"},
    }
    server = _script_server(should_disable_compat)
    server._model_instance.cache_config = (
        types.SimpleNamespace(**snapshot_data)
        if legacy_snapshot
        else CacheConfigSnapshot.model_validate(snapshot_data)
    )

    script = server._get_serving_command_script(env)

    if expect_script:
        assert "rm -rf /usr/local/cuda/compat" in script
        assert "GPUSTACK_CACHE_PACKAGES_EOF" not in script
        assert "PD cache requires" not in script
    else:
        assert script is None


@pytest.mark.parametrize("version", ["0.25.3", "0.26.0", "0.25.3-custom", None])
def test_local_nixl_explicit_engine_versions_can_start(version):
    from gpustack.schemas.models import DisaggregationSpec

    backend = _vllm_backend_with_kv_cache(ExtendedKVCacheConfig(enabled=True))
    backend._model.disaggregation = DisaggregationSpec(mode="vllm-nixl")
    backend._model.backend_version = version
    assert backend._build_extended_kv_cache_arguments(None)


@pytest.mark.parametrize("version", ["0.20.2", "0.23.0", "0.20.2-custom"])
def test_local_ascend_pd_keeps_requested_cache_outside_engine_range(version):
    from gpustack.schemas.models import DisaggregationSpec

    backend = _vllm_backend_with_kv_cache(ExtendedKVCacheConfig(enabled=True))
    backend._model.disaggregation = DisaggregationSpec(mode="vllm-ascend-mooncake")
    backend._model.backend_version = version
    args = backend._build_ascend_local_kv_cache_arguments("Ascend910B3")
    assert "SimpleCPUOffloadConnector" in args[1]


@pytest.mark.parametrize(
    "version,image,warning,informational",
    [
        ("0.20.2", None, True, False),
        ("0.20.2-custom", None, False, True),
        ("0.20.2+custom", None, False, True),
        ("0.20.2rc1", None, False, True),
        ("0.20.2", "custom/ascend:latest", False, True),
        ("0.23.0", None, False, False),
    ],
)
def test_ascend_pd_old_builtin_cache_version_is_advisory(
    version, image, warning, informational, caplog
):
    from gpustack.schemas.models import DisaggregationSpec

    backend = _vllm_backend_with_kv_cache(ExtendedKVCacheConfig(enabled=True))
    backend._model.disaggregation = DisaggregationSpec(mode="vllm-ascend-mooncake")
    backend._model.backend_version = version
    backend._model.image_name = image
    caplog.set_level(logging.INFO)
    assert backend._build_ascend_local_kv_cache_arguments("Ascend910B3")
    assert ("PD engine-version ranges are advisory" in caplog.text) is warning
    assert (
        "PD KV cache offload compatibility is unverified" in caplog.text
    ) is informational
    if informational:
        assert all(record.levelno == logging.INFO for record in caplog.records)


def test_ascend_pd_unsupported_device_still_disables_cache(caplog):
    from gpustack.schemas.models import DisaggregationSpec

    backend = _vllm_backend_with_kv_cache(ExtendedKVCacheConfig(enabled=True))
    backend._model.disaggregation = DisaggregationSpec(mode="vllm-ascend-mooncake")
    backend._model.backend_version = "0.20.2"
    assert backend._build_ascend_local_kv_cache_arguments("Ascend310P3") == []
    assert "310P" in caplog.text


def test_managed_router_records_the_same_peers_it_renders(monkeypatch):
    from unittest.mock import MagicMock
    from gpustack.schemas.models import (
        DisaggregationSpec,
        Model,
        ModelInstance,
        PortBand,
        RoleSpec,
        SourceEnum,
    )
    from gpustack.utils.pd_router_membership import router_peer_snapshot
    import gpustack.worker.pd_router as renderer

    model = Model(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        disaggregation=DisaggregationSpec(mode="vllm-mooncake"),
        roles=[RoleSpec(name="router")],
    )
    peer = types.SimpleNamespace(
        id=1,
        model_id=1,
        role="prefill",
        group_id="g",
        worker_id=7,
        port=8000,
        ports=[8000],
        distributed_servers=None,
        state="running",
        named_ports={"bootstrap": PortBand(base=9000)},
        draining_since=None,
        restart_count=2,
        last_restart_time=None,
    )
    draining = types.SimpleNamespace(**{**vars(peer), "id": 2, "draining_since": "now"})
    server = _StubServer.__new__(_StubServer)
    server._model_instance = ModelInstance(
        name="router",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        role="router",
        group_id="g",
        id=3,
    )
    server._clientset = MagicMock()
    server._clientset.model_instances.list.return_value.items = [peer, draining]
    server._clientset.workers.get.return_value = types.SimpleNamespace(ip="10.0.0.1")
    server._template_variables = MagicMock(return_value={})
    server._pd_template_variables = MagicMock(return_value={})
    server._update_model_instance = MagicMock()
    apply = MagicMock(
        return_value=types.SimpleNamespace(run_command="router", image_name="image")
    )
    monkeypatch.setattr(renderer, "apply_managed_router", apply)
    server._apply_managed_router(model)
    assert apply.call_args.kwargs["peers"] == {
        "prefill": [renderer.PeerAddress("10.0.0.1", 8000, {"bootstrap": 9000})]
    }
    assert "intra_node_data_parallel_size" not in apply.call_args.kwargs["variables"]
    server._update_model_instance.assert_called_once_with(
        3, router_peer_snapshot=router_peer_snapshot([peer], "g", {7: "10.0.0.1"})
    )


@pytest.mark.parametrize('worker_index', [0, 1])
@pytest.mark.parametrize('vendor', ['NVIDIA', 'Huawei'])
def test_external_cross_worker_rank_commands_and_device_slices(worker_index, vendor):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    backend, plan = _external_backend(vendor=vendor)
    mi = backend._model_instance
    mi.worker_ip = '10.0.0.7'
    mi.gpu_indexes = [1, 3, 5, 7]
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3, 5, 7]
        )
    ]
    mi.named_ports['_vllm_external_store'] = PortBand(base=40100, count=2)
    backend._model.backend_parameters = [
        '--data-parallel-external-lb',
        '--dp=2',
        '--tp=4',
        '--dpl=2',
    ]
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'
    manifest = backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe']
    )
    config = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith('tasks.json')
        )
    )
    for rank, task in enumerate(config['tasks']):
        argv = task['argv'][6:]

        def value(key):
            return argv[argv.index(key) + 1]

        assert value('--nnodes') == '4'
        assert value('--node-rank') == str(rank * 2 + worker_index)
        assert value('--master-port') == '40100'
        assert value('--master-addr') == '10.0.0.7'
        assert value('--data-parallel-rpc-port') == '40010'
        assert value('--data-parallel-size-local') == '1'
        assert ('--headless' in argv) == bool(worker_index)
        assert value('--api-server-count') == ('0' if worker_index else '1')
        assert json.loads(task['argv'][5]) == [rank * 2, rank * 2 + 1]
        placement = manifest.dp_rank_placements[rank]
        assert placement.gpu_indexes == ([1, 3] if rank == 0 else [5, 7])
        assert placement.node_rank == worker_index
        assert placement.api_port == (None if worker_index else 40000 + rank)
    assert {e.name: e.value for e in plan.containers[0].envs}[
        'VLLM_DP_MASTER_IP'
    ] == '10.0.0.7'


@pytest.mark.parametrize('worker_index', range(4))
def test_external_disjoint_dp_groups_share_global_store(worker_index):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    backend, plan = _external_backend()
    mi = backend._model_instance
    mi.worker_ip, mi.gpu_indexes = '10.0.0.7', [1, 3, 5, 7]
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=7 + index,
            worker_ip=f'10.0.0.{7 + index}',
            gpu_indexes=[1, 3, 5, 7],
        )
        for index in range(1, 4)
    ]
    mi.named_ports['_vllm_external_store'] = PortBand(base=40100, count=2)
    backend._model.backend_parameters = [
        '--data-parallel-external-lb',
        '--dp=2',
        '--tp=8',
        '--dpl=1',
    ]
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'

    backend._configure_external_workload(plan, ['vllm', 'serve'], ['/models/moe'])

    config = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith('tasks.json')
        )
    )
    assert len(config['tasks']) == 1
    argv = config['tasks'][0]['argv']
    assert argv[argv.index('--node-rank') + 1] == str(worker_index)
    assert argv[argv.index('--master-addr') + 1] == '10.0.0.7'
    assert argv[argv.index('--master-port') + 1] == '40100'


def test_managed_backend_version_conflict_stops_preparation_before_model_write():
    from unittest.mock import MagicMock

    backend, _ = _external_backend()
    backend._model.backend_version = None
    backend._model_spec = new_model(1, 'moe', huggingface_repo_id='org/moe')
    backend._clientset = MagicMock()
    backend._update_model_instance = MagicMock(
        side_effect=RuntimeError('version conflict')
    )
    with pytest.raises(RuntimeError, match='version conflict'):
        backend._update_model_backend_service_version('0.32.0')
    backend._clientset.models.update.assert_not_called()


@pytest.mark.parametrize(
    'parameters, mode',
    [
        (['-dpm', '-dpr', '2'], 'multi_port_external'),
        (
            ['--data-parallel-multi-port-external-lb', '--data-parallel-start-rank=0'],
            'multi_port_external',
        ),
        (['-dpe'], 'external'),
    ],
)
def test_multi_port_external_mode_preserves_start_rank_alias(parameters, mode):
    assert vllm_lb_mode(parameters) == mode


@pytest.mark.parametrize('vendor', ['NVIDIA', 'Huawei'])
@pytest.mark.parametrize('shutdown_flag', ['--shutdown-timeout', '--shutdown_timeout'])
def test_multi_port_external_plan_uses_vllm_launcher_and_contiguous_apis(
    vendor, shutdown_flag
):
    backend, plan = _external_backend(vendor=vendor)
    backend._model.backend_parameters = ['-dpm', '--dp=2', '--tp=2']
    backend._model_instance.gpu_indexes = [1, 3, 5, 7]
    backend._model_instance.named_ports['_vllm_external_api'] = PortBand(
        base=40000, count=2
    )
    backend._model_instance.named_ports['_vllm_multi_port_external_supervisor'] = (
        PortBand(base=40030)
    )
    backend._override_entrypoint = lambda command, args, script: (command, args)
    manifest = backend._configure_multi_port_external_workload(
        plan,
        ['vllm', 'serve'],
        ['/models/moe', *backend._model.backend_parameters, f'{shutdown_flag}=60'],
    )
    argv = plan.containers[0].execution.command + plan.containers[0].execution.args
    assert '--data-parallel-multi-port-external-lb' in argv
    assert '--data-parallel-external-lb' not in argv
    assert plan.containers[0].files == []
    assert plan.termination_grace_period_seconds == 75
    assert [p.dp_rank for p in manifest.dp_rank_placements] == [0, 1]
    assert [p.api_port for p in manifest.dp_rank_placements] == [40000, 40001]
    assert [p.gpu_indexes for p in manifest.dp_rank_placements] == [[1, 3], [5, 7]]
    assert argv[argv.index('--data-parallel-supervisor-port') + 1] == '40030'
    assert argv[argv.index('--data-parallel-size-local') + 1] == '2'


@pytest.mark.parametrize('worker_index', [0, 1])
def test_multi_port_external_cross_worker_apis_have_real_global_ranks(worker_index):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    backend, plan = _external_backend()
    backend._model.backend_parameters = ['-dpm', '--dp=4', '--tp=2']
    mi = backend._model_instance
    mi.gpu_indexes = [1, 3, 5, 7]
    mi.worker_ip = '10.0.0.7'
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3, 5, 7]
        )
    ]
    mi.named_ports['_vllm_external_api'] = PortBand(base=40000, count=4)
    mi.named_ports['_vllm_multi_port_external_supervisor'] = PortBand(base=40030)
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'
    manifest = backend._configure_multi_port_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *backend._model.backend_parameters]
    )
    argv = plan.containers[0].execution.command
    assert argv[argv.index('--port') + 1] == str(40000 + 2 * worker_index)
    assert argv[argv.index('--data-parallel-start-rank') + 1] == str(2 * worker_index)
    assert '--headless' not in argv
    assert [p.dp_rank for p in manifest.dp_rank_placements] == [
        2 * worker_index,
        2 * worker_index + 1,
    ]
    env = {e.name: e.value for e in plan.containers[0].envs}
    assert env['VLLM_DP_MASTER_PORT'] == '40020'
    assert env['VLLM_DP_MASTER_IP'] == '10.0.0.7'


@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi_port_external'])
@pytest.mark.parametrize(
    'parameters',
    [
        ['--ssl-keyfile=key.pem'],
        ['--ssl-certfile=cert.pem'],
        ['--uds', '/tmp/model-api.sock'],
        ['--uds=/tmp/model-api.sock'],
    ],
)
def test_managed_rank_http_contract_rejects_incompatible_listeners(mode, parameters):
    backend, plan = _external_backend()
    backend._model.backend_parameters[0] = {
        'external': '--data-parallel-external-lb',
        'hybrid': '--data-parallel-hybrid-lb',
        'multi_port_external': '--data-parallel-multi-port-external-lb',
    }[mode]
    configure = {
        'external': backend._configure_external_workload,
        'hybrid': backend._configure_hybrid_workload,
        'multi_port_external': backend._configure_multi_port_external_workload,
    }[mode]
    with pytest.raises(ValueError, match='routes and health checks.*HTTP'):
        configure(
            plan,
            ['vllm', 'serve'],
            ['/models/moe', *backend._model.backend_parameters, *parameters],
        )


@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi_port_external'])
@pytest.mark.parametrize('form', ['equal', 'space'])
@pytest.mark.parametrize(
    'value',
    [
        "You're helpful. {{ messages[0].content }}",
        'text --data-parallel-hybrid-lb --tp=8 --uds=/tmp/model.sock',
    ],
)
def test_managed_launch_preserves_template_argv(mode, form, value):
    backend, plan = _external_backend()
    parameters = [f'--data-parallel-{mode.replace("_", "-")}-lb', '--dp=2', '--tp=2']
    backend._model.backend_parameters = [
        *parameters,
        f'--chat-template={shlex.quote(value)}',
    ]
    backend._model_instance.gpu_indexes = [1, 3, 5, 7]
    backend._model_instance.named_ports['_vllm_external_api'] = PortBand(
        base=40000, count=1 if mode == 'hybrid' else 2
    )
    backend._model_instance.named_ports['_vllm_multi_port_external_supervisor'] = (
        PortBand(base=40030)
    )
    template = (
        [f'--chat-template={value}'] if form == 'equal' else ['--chat-template', value]
    )
    configure = {
        'external': backend._configure_external_workload,
        'hybrid': backend._configure_hybrid_workload,
        'multi_port_external': backend._configure_multi_port_external_workload,
    }[mode]
    configure(plan, ['vllm', 'serve'], ['/models/moe', *parameters, *template])
    container = plan.containers[0]
    if mode == 'external':
        tasks = json.loads(
            next(
                file.content
                for file in container.files
                if file.path.endswith('tasks.json')
            )
        )['tasks']
        commands = [task['argv'] for task in tasks]
    else:
        commands = [container.execution.command + (container.execution.args or [])]
    for argv in commands:
        assert template[0] in argv
        if form == 'space':
            assert argv[argv.index('--chat-template') + 1] == value


@pytest.mark.parametrize('parameter', ['-dpm', '--data-parallel-start-rank=0'])
def test_external_command_cannot_switch_to_another_load_balancing_mode(parameter):
    backend, plan = _external_backend()
    with pytest.raises(ValueError, match='selected External-LB mode'):
        backend._configure_external_workload(
            plan, ['vllm', 'serve'], ['/models/moe', parameter]
        )


def test_multi_port_external_custom_command_retains_supervisor_parameters():
    backend, plan = _external_backend()
    backend._model.backend_parameters[0] = '-dpm'
    mi = backend._model_instance
    mi.named_ports['_vllm_external_api'] = PortBand(base=40000, count=2)
    mi.named_ports['_vllm_multi_port_external_supervisor'] = PortBand(base=40030)
    manifest = backend._configure_multi_port_external_workload(
        plan,
        [
            'vllm',
            'serve',
            '/models/moe',
            *backend._model.backend_parameters,
            '--dp-supervisor-probe-interval-s=7',
        ],
        [],
    )
    assert len(manifest.dp_rank_placements) == 2
    assert '--dp-supervisor-probe-interval-s=7' in plan.containers[0].execution.command


@pytest.mark.asyncio
@pytest.mark.parametrize('role', ['prefill', 'decode'])
async def test_new_pd_external_member_builds_distinct_rank_plans(monkeypatch, role):
    from unittest.mock import AsyncMock, MagicMock
    from gpustack.schemas.models import (
        Model,
        ModelInstance,
        RoleSpec,
        DisaggregationSpec,
        role_effective_model,
    )
    from gpustack.server import controllers

    backend, plan = _external_backend()
    parameters = list(backend._model.backend_parameters)
    model = Model(
        id=1,
        name='pd',
        source='huggingface',
        huggingface_repo_id='org/moe',
        owner_principal_id=1,
        backend=BackendEnum.VLLM,
        backend_version='0.32.0',
        backend_parameters=['--tp=16'],
        disaggregation=DisaggregationSpec(mode='vllm-nixl'),
        roles=[
            RoleSpec(name=name, replicas=1, backend_parameters=parameters)
            for name in ['prefill', 'decode']
        ]
        + [RoleSpec(name='router', replicas=1)],
    )
    monkeypatch.setattr(
        controllers, 'get_draft_model_source', AsyncMock(return_value=None)
    )
    row = await controllers._build_instance_create(
        MagicMock(),
        model,
        next(r for r in model.roles if r.name == role),
        'g1',
        'digest',
    )
    assert row.distributed_servers.workloads == []
    mi = ModelInstance.model_validate(row.model_dump())
    mi.id, mi.worker_id, mi.worker_ip = 1, 7, '10.0.0.7'
    mi.restart_count = 3
    mi.gpu_indexes = [1, 3, 5, 7]
    mi.port, mi.ports = 40000, backend._model_instance.ports
    mi.named_ports = {
        **backend._model_instance.named_ports,
        'kv_side_channel': PortBand(base=40100, count=2),
    }
    backend._model_instance = mi
    backend._model_spec = model
    backend._model = role_effective_model(model, role)
    backend._model_path = '/models/moe'
    backend._pd_injection_resolved = False
    backend._pd_injection_cache = None
    backend._pd_template_variables = lambda: {'net_device': 'eth0'}
    manifest = backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *parameters]
    )
    tasks = json.loads(
        next(
            f.content
            for f in plan.containers[0].files
            if f.path == '/gpustack/tasks.json'
        )
    )['tasks']
    assert [p.api_port for p in manifest.dp_rank_placements] == [40000, 40001]
    for rank, task in enumerate(tasks):
        argv = task['argv']
        descriptor = json.loads(argv[argv.index('--kv-transfer-config') + 1])
        assert descriptor['engine_id'] == f'gpustack-1-r3-dp{rank}'
        assert descriptor['kv_role'] == (
            'kv_producer' if role == 'prefill' else 'kv_consumer'
        )
        assert task['env']['VLLM_NIXL_SIDE_CHANNEL_PORT'] == '40100'
    assert model.backend_parameters == ['--tp=16']
    assert backend._model.backend_parameters == parameters


@pytest.mark.parametrize(
    'flag', ['--kv-transfer-config', '--kv-transfer-config=', '--kv_transfer_config=']
)
def test_pd_rank_final_user_kv_json_and_generated_files_are_isolated(flag):
    from gpustack.worker.pd_injection import PDInjection

    backend, plan = _external_backend()
    backend._model_instance.role = 'prefill'
    backend._model.env = {}
    backend._template_variables = lambda **kwargs: kwargs
    backend._pd_injection = lambda **kwargs: PDInjection(
        env={'PD_CONFIG': '/etc/pd.json'},
        files={'/etc/pd.json': json.dumps({'rank': kwargs['dp_rank']})},
    )
    descriptor = {
        'kv_connector': 'MultiConnector',
        'engine_id': 'user-engine',
        'kv_connector_extra_config': {
            'connectors': [
                {'kv_connector': 'NixlConnector', 'engine_id': 'user-child'},
                {'kv_connector': 'LMCacheConnectorV1'},
            ]
        },
    }
    encoded = json.dumps(descriptor)
    user_args = [flag + encoded] if flag.endswith('=') else [flag, encoded]
    backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *user_args]
    )
    files = {f.path: f.content for f in plan.containers[0].files}
    tasks = json.loads(files['/gpustack/tasks.json'])['tasks']
    for rank, task in enumerate(tasks):
        index = next(
            i
            for i, token in enumerate(task['argv'])
            if token.split('=', 1)[0].replace('_', '-') == '--kv-transfer-config'
        )
        token = task['argv'][index]
        raw = token.split('=', 1)[1] if '=' in token else task['argv'][index + 1]
        actual = json.loads(raw)
        assert actual['engine_id'] == f'gpustack-1-r3-dp{rank}'
        assert all(
            'engine_id' not in child
            for child in actual['kv_connector_extra_config']['connectors']
        )
        config_path = task['env']['PD_CONFIG']
        assert config_path == f'/gpustack/pd/dp{rank}/etc/pd.json'
        assert json.loads(files[config_path]) == {'rank': rank}
    assert json.loads(encoded) == descriptor


@pytest.mark.parametrize('pd_role', [None, 'prefill', 'decode'])
def test_external_dp1_dense_stays_one_complete_engine(pd_role):
    backend, plan = _external_backend(dp=1)
    backend._get_pretrained_config = lambda: types.SimpleNamespace(hidden_size=4096)
    if pd_role:
        from gpustack.worker.pd_injection import PDInjection

        backend._model_instance.role = pd_role
        backend._model.env = {}
        backend._template_variables = lambda **kwargs: kwargs
        backend._pd_injection = lambda **kwargs: PDInjection(
            args=['--kv-transfer-config', json.dumps({'kv_connector': 'NixlConnector'})]
        )
    manifest = backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/dense']
    )
    assert len(manifest.dp_rank_placements) == 1
    assert manifest.dp_rank_placements[0].gpu_indexes == [1, 3]


def test_managed_pd_dense_tp_across_workers_keeps_one_engine_identity():
    from gpustack.schemas.models import ModelInstanceSubordinateWorker
    from gpustack.worker.pd_injection import PDInjection

    identities = []
    for worker_index in (0, 1):
        backend, plan = _external_backend(dp=1)
        backend._model.backend_parameters = [
            '--data-parallel-external-lb',
            '--dp=1',
            '--tp=4',
        ]
        backend._get_pretrained_config = lambda: types.SimpleNamespace(hidden_size=4096)
        backend._model.env = {}
        backend._template_variables = lambda **kwargs: kwargs
        backend._pd_injection = lambda **kwargs: PDInjection(
            args=['--kv-transfer-config', '{"kv_connector":"NixlConnector"}']
        )
        mi = backend._model_instance
        mi.role, mi.worker_ip, mi.gpu_indexes = 'prefill', '10.0.0.7', [1, 3]
        mi.distributed_servers.subordinate_workers = [
            ModelInstanceSubordinateWorker(
                worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3]
            )
        ]
        mi.named_ports['_vllm_external_store'] = PortBand(base=40100, count=1)
        backend._worker.id = 7 + worker_index
        backend._worker.ip = f'10.0.0.{7 + worker_index}'
        manifest = backend._configure_external_workload(
            plan, ['vllm', 'serve'], ['/models/dense']
        )
        tasks = json.loads(
            next(
                f.content
                for f in plan.containers[0].files
                if f.path == '/gpustack/tasks.json'
            )
        )['tasks']
        assert len(tasks) == 1
        argv = tasks[0]['argv']
        identities.append(
            json.loads(argv[argv.index('--kv-transfer-config') + 1])['engine_id']
        )
        placement = manifest.dp_rank_placements[0]
        assert placement.node_rank == worker_index
        assert placement.api_port == (None if worker_index else 40000)
        assert ('--headless' in argv) is bool(worker_index)
    assert identities == ['gpustack-1-r3-dp0'] * 2


def _managed_pd_backend_case(mode, env=None, *, hybrid=False):
    from gpustack.schemas.models import (
        Model,
        RoleSpec,
        DisaggregationSpec,
        role_effective_model,
    )

    backend, plan = _external_backend(
        vendor='Huawei' if mode == 'vllm-ascend-mooncake' else 'NVIDIA'
    )
    parameters = list(backend._model.backend_parameters)
    if hybrid:
        parameters[0] = '--data-parallel-hybrid-lb'
        backend._model_instance.named_ports['_vllm_external_api'] = PortBand(base=40000)
    model = Model(
        id=1,
        name='pd',
        source='huggingface',
        huggingface_repo_id='org/moe',
        owner_principal_id=1,
        backend=BackendEnum.VLLM,
        env=env or {},
        disaggregation=DisaggregationSpec(mode=mode),
        roles=[
            RoleSpec(name=name, backend_parameters=parameters)
            for name in ['prefill', 'decode']
        ],
    )
    backend._model_spec = model
    backend._model = role_effective_model(model, 'prefill')
    backend._model_instance.role, backend._model_instance.gpu_indexes = 'prefill', [
        1,
        3,
        5,
        7,
    ]
    backend._model_instance.named_ports.update(
        {
            'kv_side_channel': PortBand(base=40100, count=2),
            'kv_port': PortBand(base=40200, count=4),
        }
    )
    backend._model_path = '/models/moe'
    backend._pd_injection_resolved, backend._pd_injection_cache = False, None
    backend._pd_template_variables = lambda: {'net_device': 'eth0'}
    return backend, plan


@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize(
    'mode,key,value',
    [
        ('vllm-nixl', 'VLLM_NIXL_SIDE_CHANNEL_PORT', '49999'),
        ('vllm-nixl', 'VLLM_NIXL_SIDE_CHANNEL_HOST', '10.99.0.1'),
        ('vllm-ascend-mooncake', 'HCCL_IF_IP', '10.99.0.1'),
    ],
)
def test_managed_pd_rejects_reserved_endpoint_env_conflicts(mode, key, value, hybrid):
    backend, plan = _managed_pd_backend_case(mode, {key: value}, hybrid=hybrid)
    configure = (
        backend._configure_hybrid_workload
        if hybrid
        else backend._configure_external_workload
    )
    with pytest.raises(ValueError, match='reserved connector'):
        configure(plan, ['vllm', 'serve'], ['/models/moe'])


@pytest.mark.parametrize('port', [None, 40200, 49999])
def test_managed_ascend_final_kv_json_uses_reserved_port(port):
    backend, plan = _managed_pd_backend_case(
        'vllm-ascend-mooncake', {'HCCL_SOCKET_IFNAME': 'bond0'}
    )
    descriptor = {'kv_connector': 'MooncakeHybridConnector', 'kv_role': 'kv_producer'}
    if port is not None:
        descriptor['kv_port'] = port
    args = ['/models/moe', '--kv-transfer-config', json.dumps(descriptor)]
    if port == 49999:
        with pytest.raises(ValueError, match='reserved connector'):
            backend._configure_external_workload(plan, ['vllm', 'serve'], args)
        return
    backend._configure_external_workload(plan, ['vllm', 'serve'], args)
    tasks = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith('tasks.json')
        )
    )['tasks']
    for task in tasks:
        argv = task['argv']
        assert (
            json.loads(argv[argv.index('--kv-transfer-config') + 1])['kv_port'] == 40200
        )
        assert task['env']['HCCL_SOCKET_IFNAME'] == 'bond0'


@pytest.mark.parametrize(
    'parameters,expected',
    [
        (['--dp-size=4'], True),
        (['--enable-dp-attention', '--dp', '2'], True),
        (['--ep-size', '8'], False),
        (['--ep-size', '8', '--dp-size', '2'], True),
        (['--enable-dp-attention', '--dp-size', '1'], False),
        (['--tp', '8'], False),
        (['--dp-size=1'], False),
        (['--policy=random'], False),
        (['--dp=4', '--data-parallel-size=1'], False),
    ],
)
def test_sglang_smg_mode_uses_effective_parallelism(parameters, expected):
    from gpustack.utils.model_instance_workers import sglang_smg_mode

    model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=parameters,
    )
    assert sglang_smg_mode(model) is expected
    model.disaggregation = True
    assert not sglang_smg_mode(model)


@pytest.mark.parametrize(
    'parameters,expected,policy',
    [
        (['--dp=4'], ['--dp=4'], 'cache_aware'),
        (
            ['--policy=random', '--policy round_robin', '--dp=4'],
            ['--dp=4'],
            'round_robin',
        ),
        (
            [
                '--schedule-policy=lpm',
                '--load-balance-method=auto',
                '--policy',
                'manual',
            ],
            ['--schedule-policy=lpm', '--load-balance-method=auto'],
            'manual',
        ),
    ],
)
def test_sglang_smg_policy_is_native_and_nonmutating(parameters, expected, policy):
    from gpustack.worker.backends.sglang import split_sglang_smg_parameters

    original = list(parameters)
    assert split_sglang_smg_parameters(parameters, {'sglang-router': '0.3.2'}) == (
        expected,
        policy,
    )
    assert parameters == original


@pytest.mark.parametrize('tokenized', [False, True])
@pytest.mark.parametrize(
    'value',
    [
        '/models/chat templates/template.jinja',
        "/models/worker's templates/template.jinja",
        'text --policy=random --host=wrong',
    ],
)
def test_sglang_policy_preserves_opaque_values(value, tokenized):
    from gpustack.worker.backends.sglang import split_sglang_smg_parameters

    native = f'--chat-template={value}'
    parameters = [
        '--dp=4',
        '--policy=manual',
        native if tokenized else f'--chat-template={shlex.quote(value)}',
    ]
    original = list(parameters)
    kwargs = {'tokenized': True} if tokenized else {}
    assert split_sglang_smg_parameters(
        parameters, {'sglang-router': '0.3.2'}, **kwargs
    ) == (['--dp=4', native], 'manual')
    assert parameters == original


@pytest.mark.parametrize(
    'parameters,dependencies',
    [
        (['--policy'], {'sglang-router': '0.3.2'}),
        (['--policy='], {'sglang-router': '0.3.2'}),
        (['--policy', '--dp=4'], {'sglang-router': '0.3.2'}),
        (['--policy=consistent_hashing'], {'sglang-router': '0.3.2'}),
        ([], None),
        ([], {'sglang-router': '0.5.0'}),
    ],
)
def test_sglang_smg_policy_rejects_unknown_capabilities(parameters, dependencies):
    from gpustack.worker.backends.sglang import split_sglang_smg_parameters

    with pytest.raises(ValueError):
        split_sglang_smg_parameters(parameters, dependencies)


def test_resolved_image_capabilities_follow_selected_runner_and_reset(monkeypatch):
    import gpustack.worker.backends.base as base_module

    runner = _make_versioned_runner('13.0', 'selected-image')
    runner.variants[0].services[0].versions[0].platforms[0].dependencies = {
        'sglang-router': '0.3.2'
    }
    monkeypatch.setattr(
        base_module,
        'merged_backend_runners',
        lambda *_, **__: [types.SimpleNamespace(versions=[runner])],
    )
    server = SGLangServer.__new__(SGLangServer)
    server._model = types.SimpleNamespace(
        image_name=None, backend='sglang', backend_version=None
    )
    server.inference_backend = None
    server._get_device_info = lambda: ('cuda', '13.0', None)
    server._fetch_runner_overrides = lambda: []
    assert server._resolve_image()[0] == 'selected-image'
    assert server._resolved_image_dependencies == {'sglang-router': '0.3.2'}
    server._model.image_name = 'custom-image'
    assert server._resolve_image()[0] == 'custom-image'
    assert server._resolved_image_dependencies is None


def test_ordinary_sglang_legacy_launcher_does_not_forward_smg_policy():
    server = SGLangServer.__new__(SGLangServer)
    server._model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=['--dp=4', '--policy=random'],
    )
    server._model_instance = new_model_instance(1, 'sg', model_id=1, worker_id=1)
    with pytest.raises(ValueError, match='paired SMG'):
        server._start()


def _managed_sglang_backend(worker_id=1):
    from unittest.mock import MagicMock

    backend = SGLangServer.__new__(SGLangServer)
    backend._model = new_model(
        1,
        'sg',
        huggingface_repo_id='org/sg',
        backend='SGLang',
        backend_parameters=['--dp=4', '--policy=manual'],
    )
    backend._model_instance = new_model_instance(1, 'sg', model_id=1, worker_id=1)
    backend._model_instance.worker_ip = '10.0.0.1'
    backend._model_instance.restart_count = 3
    backend._model_instance.port = 40000
    backend._model_instance.ports = [40000, 40001, 40002, 40003]
    backend._model_instance.named_ports = {
        '_sglang_engine_api': PortBand(base=40000),
        '_sglang_connecting': PortBand(base=40001),
        '_sglang_smg_api': PortBand(base=40002),
        '_sglang_smg_metrics': PortBand(base=40003),
    }
    backend._model_instance.distributed_servers = DistributedServers(workloads=[])
    if worker_id != 1:
        backend._model_instance.distributed_servers.subordinate_workers = [
            ModelInstanceSubordinateWorker(
                worker_id=worker_id, worker_ip='10.0.0.2', gpu_indexes=[0]
            )
        ]
    backend._worker = types.SimpleNamespace(id=worker_id, ip=f'10.0.0.{worker_id}')
    backend._smg_image_dependencies = {'sglang-router': '0.3.2'}
    backend._get_configured_resources = lambda: {'nvidia.com/gpu': '0'}
    backend._get_configured_mounts = lambda: []
    backend._get_configured_ports = lambda: []
    backend._get_container_env_config = lambda env: types.SimpleNamespace(
        user=0, group=0, shm_size_gib=1
    )
    backend._host_ipc_enabled = lambda: False
    backend._transform_workload_plan = lambda plan, **kwargs: plan
    backend._workload_plan_callback = MagicMock()
    return backend


@pytest.mark.parametrize('worker_id,count', [(1, 2), (2, 1)])
def test_managed_sglang_prepares_engine_and_primary_cpu_smg(
    monkeypatch, worker_id, count
):
    from gpustack.schemas.models import ROUTER_DEFAULT_CPU, ROUTER_DEFAULT_MEMORY

    backend = _managed_sglang_backend(worker_id)
    monkeypatch.setattr(
        'gpustack.worker.backends.sglang.create_workload',
        lambda *args: pytest.fail('preparer created runtime'),
    )
    backend._create_workload(
        backend._model_instance.get_deployment_metadata(worker_id),
        ['sglang', 'serve'],
        None,
        ['--dp=4', '--port=40000'],
        {'CUDA_VISIBLE_DEVICES': '0', 'NCCL_DEBUG': 'INFO'},
        'same-engine-image',
    )
    manifests, plans = backend._workload_plan_callback.call_args.args
    assert len(plans) == len(manifests) == count
    assert manifests[0].component.kind == 'sglang_engine'
    assert manifests[0].component.api_port == (40000 if worker_id == 1 else None)
    assert manifests[0].component.dp_size == (4 if worker_id == 1 else None)
    assert not manifests[0].dp_rank_placements
    assert plans[0].containers[0].resources == {'nvidia.com/gpu': '0'}
    assert manifests[0].component.image == 'same-engine-image'
    assert manifests[0].component.router_version == '0.3.2'

    assert '--policy' not in ' '.join(
        [
            *(plans[0].containers[0].execution.command or []),
            *(plans[0].containers[0].execution.args or []),
        ]
    )
    if worker_id == 1:
        smg = plans[1].containers[0]
        assert smg.image == 'same-engine-image'
        assert smg.resources == {
            'cpu': ROUTER_DEFAULT_CPU,
            'memory': ROUTER_DEFAULT_MEMORY,
        }
        assert [(env.name, env.value) for env in smg.envs] == [
            ('TORCH_DEVICE_BACKEND_AUTOLOAD', '0'),
            ('ASCEND_VISIBLE_DEVICES', 'void'),
            ('NVIDIA_VISIBLE_DEVICES', 'void'),
        ]
        assert not smg.mounts
        assert not plans[1].instance_type and not plans[1].host_ipc
        argv = [*(smg.execution.command or []), *smg.execution.args]
        assert argv[:3] == ['python3', '-m', 'sglang_router.launch_router']
        assert argv[argv.index('--worker-urls') + 1] == 'http://10.0.0.1:40000'
        assert argv[argv.index('--policy') + 1] == 'manual'
        assert '--dp-aware' in argv
        assert {port.internal for port in smg.ports} == {40002, 40003}
        assert not {40002, 40003} & {
            port.internal for port in plans[0].containers[0].ports
        }
    assert pickle.loads(pickle.dumps((manifests, plans)))[0] == manifests


@pytest.mark.parametrize('dependencies', [None, {}, {'sglang-router': 'unknown'}])
def test_managed_sglang_preparation_rejects_unknown_image_capability(
    monkeypatch, dependencies
):
    backend = _managed_sglang_backend()
    backend._smg_image_dependencies = dependencies
    monkeypatch.setattr(
        'gpustack.worker.backends.sglang.create_workload',
        lambda *args: pytest.fail('preparer created runtime'),
    )
    with pytest.raises(ValueError, match='trusted'):
        backend._create_workload(
            backend._model_instance.get_deployment_metadata(1),
            ['sglang', 'serve'],
            None,
            [],
            {},
            'unknown-image',
        )
    backend._workload_plan_callback.assert_not_called()


@pytest.mark.parametrize(
    'parameters,error',
    [
        (
            [
                '--dp=4',
                '--policy=manual',
                '--schedule-policy=lpm',
                '--load-balance-method=auto',
            ],
            False,
        ),
        (['--dp=4', '--host=wrong'], True),
        (['--dp=4', '--port=9999'], True),
    ],
)
def test_managed_sglang_engine_argv_splits_policy_and_fences_addresses(
    parameters, error
):
    backend = _managed_sglang_backend()
    backend._model.backend_parameters = list(parameters)
    backend._model_path = '/models/sg'
    backend.inference_backend = None
    backend._derive_max_model_len = lambda: None
    backend._get_model_architecture = lambda: []
    backend._get_speculative_arguments = lambda: []
    backend._get_hicache_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: []
    backend._flatten_backend_param = lambda: list(parameters)
    if error:
        with pytest.raises(ValueError, match='conflicts'):
            backend._build_command_args(40000, False, False)
    else:
        argv, injected = backend._build_command_args(40000, False, False)
        assert '--policy=manual' not in argv
        assert '--schedule-policy=lpm' in argv and '--load-balance-method=auto' in argv
    assert backend._model.backend_parameters == parameters


@pytest.mark.parametrize(
    'parameter_format', [None, ParameterFormatEnum.SPACE, ParameterFormatEnum.EQUAL]
)
@pytest.mark.parametrize('equal', [False, True])
@pytest.mark.parametrize(
    'value',
    [
        '/models/chat templates/template.jinja',
        "/models/worker's templates/template.jinja",
        'text --policy=random --host=wrong',
    ],
)
def test_sglang_build_command_keeps_real_flattened_values(
    parameter_format, equal, value
):
    import argparse

    backend = _managed_sglang_backend()
    option = (
        f'--chat-template={shlex.quote(value)}'
        if equal
        else f'--chat-template {shlex.quote(value)}'
    )
    parameters = ['--dp=4', '--policy=manual', option]
    backend._model.backend_parameters = list(parameters)
    backend._model_path = '/models/sg'
    backend._model_spec = backend._model
    backend._pd_injection_resolved, backend._pd_injection_cache = True, None
    backend.inference_backend = None
    backend._parameter_format = lambda: parameter_format
    backend._derive_max_model_len = lambda: None
    backend._get_model_architecture = lambda: []
    backend._get_speculative_arguments = lambda: []
    backend._get_hicache_arguments = lambda: []
    backend._get_selected_gpu_devices = lambda: []

    argv, _ = backend._build_command_args(40000, False, False)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--chat-template')
    parsed, _ = parser.parse_known_args(argv)
    assert parsed.chat_template == value
    assert not any(token.partition('=')[0] == '--policy' for token in argv)
    assert argv[argv.index('--host') + 1] == '10.0.0.1'
    assert backend._model.backend_parameters == parameters


@pytest.mark.parametrize(
    'value', ["worker's template", 'text --host=wrong --port=9999']
)
def test_sglang_address_validation_reads_existing_argv(value):
    backend = _managed_sglang_backend()
    backend._validate_managed_engine_parameters(
        [f'--chat-template={value}'], False, False, 40000
    )


def test_accelerator_free_plan_conversion_skips_model_instance_type(monkeypatch):
    import gpustack.worker.backends.base as base_module

    backend = _StubServer.__new__(_StubServer)
    backend._model_instance = new_model_instance(1, 'engine', 1)
    backend._model = types.SimpleNamespace(
        gpu_type_selector=types.SimpleNamespace(type='gpu-instance-type')
    )
    backend._config, backend._fallback_registry = None, None
    monkeypatch.setattr(
        base_module, 'transform_workload_plan', lambda config, plan, registry: plan
    )
    cpu = WorkloadPlan(name='smg', containers=[])
    engine = WorkloadPlan(name='engine', containers=[])
    assert (
        backend._transform_workload_plan(cpu, accelerator_free=True).instance_type
        is None
    )
    assert backend._transform_workload_plan(engine).instance_type == 'gpu-instance-type'


def test_smg_recovery_plan_is_pure_cpu_and_preserves_resolved_image():
    from gpustack.schemas.models import ModelInstanceComponent
    from gpustack.worker.backends.sglang import build_sglang_smg_workload

    component = ModelInstanceComponent(
        kind='sglang_smg', api_port=41002, metrics_port=41003, attempt=2
    )
    plan = build_sglang_smg_workload(
        'instance-r3-smg-a2',
        'mirror/engine:chosen',
        'cache_aware',
        component,
        '10.0.0.1',
        'http://10.0.0.1:41000',
    )
    assert plan.name == 'instance-r3-smg-a2'
    assert plan.instance_type is None
    container = plan.containers[0]
    assert container.image == 'mirror/engine:chosen'
    assert [(env.name, env.value) for env in container.envs] == [
        ('TORCH_DEVICE_BACKEND_AUTOLOAD', '0'),
        ('ASCEND_VISIBLE_DEVICES', 'void'),
        ('NVIDIA_VISIBLE_DEVICES', 'void'),
    ]
    assert not container.mounts
    assert set(container.resources) == {'cpu', 'memory'}
    assert container.execution.command == [
        'python3',
        '-m',
        'sglang_router.launch_router',
    ]
    assert (
        container.execution.args[container.execution.args.index('--worker-urls') + 1]
        == 'http://10.0.0.1:41000'
    )


@pytest.mark.parametrize('handed_off', [True, False])
def test_sglang_child_preparer_failure_respects_engine_handoff(handed_off):
    from unittest.mock import MagicMock, patch
    from gpustack.schemas.models import DistributedServers, ModelInstanceWorkload
    from gpustack.worker.backends.sglang import SGLangServer
    from tests.utils.model import new_model_instance

    server = object.__new__(SGLangServer)
    mi = new_model_instance(1, 'sg', 1, worker_id=1)
    mi.distributed_servers = DistributedServers(workloads=[])
    server._model_instance = mi
    server._worker = types.SimpleNamespace(id=1)
    server._clientset = MagicMock()
    current = mi.model_copy(deep=True)
    if handed_off:
        current.distributed_servers.workloads = [
            ModelInstanceWorkload(
                name='engine',
                worker_id=1,
                state='running',
                component=dict(
                    kind='sglang_engine', api_port=41000, dp_size=2, ready=True
                ),
            )
        ]
        current.distributed_servers.orchestration_mode = "managed"
    server._clientset.model_instances.get.return_value = current
    error = BrokenPipeError('preparation pipe closed')
    with patch.object(InferenceServer, '_handle_error', side_effect=error) as parent:
        with pytest.raises(BrokenPipeError):
            server._handle_error(error)
    assert parent.call_count == int(not handed_off)


@pytest.mark.parametrize('conflict', [None, 'dist-init-addr', 'nccl-port'])
def test_managed_single_node_sglang_dpa_uses_reserved_internal_ports(conflict):
    from gpustack.utils.command import find_parameter

    backend = _managed_sglang_backend()
    backend._model.backend_parameters.append('--enable-dp-attention')
    band = backend._model_instance.named_ports['_sglang_connecting']
    band.count = 14
    args = backend._get_multinode_arguments(True)
    assert find_parameter(args, ['dist-init-addr']) == '127.0.0.1:40001'
    assert find_parameter(args, ['nccl-port']) == '40008'
    parameters = [f'--{conflict}=49999'] if conflict else []
    if conflict:
        with pytest.raises(ValueError, match='reserved allocation'):
            backend._validate_managed_engine_parameters(parameters, False, True, 40000)
    else:
        backend._validate_managed_engine_parameters([], False, True, 40000)


def test_managed_sglang_dpa_refuses_to_expand_a_published_old_layout():
    backend = _managed_sglang_backend()
    backend._model.backend_parameters.append('--enable-dp-attention')
    with pytest.raises(ValueError, match='recreate the instance'):
        backend._get_multinode_arguments(True)


def test_legacy_single_node_sglang_dpa_keeps_native_port_selection():
    backend = _managed_sglang_backend()
    backend._model_instance.distributed_servers.workloads = None
    backend._model_instance.distributed_servers.orchestration_mode = "legacy"
    backend._model.backend_parameters.append('--enable-dp-attention')
    assert backend._get_multinode_arguments(True) == []


@pytest.mark.parametrize('worker_index', [0, 1])
@pytest.mark.parametrize('vendor', ['NVIDIA', 'Huawei'])
@pytest.mark.parametrize('shutdown_flag', ['--shutdown-timeout', '--shutdown_timeout'])
def test_hybrid_native_launcher_keeps_one_shared_api_and_local_engine_pool(
    worker_index, vendor, shutdown_flag
):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    backend, plan = _external_backend(vendor=vendor)
    plan.containers[0].envs = [
        ContainerEnv(name='VLLM_CACHE_ROOT', value='/cache/vllm')
    ]
    mi = backend._model_instance
    mi.worker_ip = '10.0.0.7'
    mi.gpu_indexes = [1, 3, 5, 7]
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3, 5, 7]
        )
    ]
    mi.ports = [40000, 40010]
    mi.named_ports = {
        '_vllm_external_api': PortBand(base=40000),
        '_vllm_external_rpc': PortBand(base=40010),
    }
    backend._model.backend_parameters = [
        '--data-parallel-hybrid-lb',
        '--dp=4',
        '--tp=2',
        '--api-server-count=3',
    ]
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'
    manifest = backend._configure_hybrid_workload(
        plan,
        ['vllm', 'serve'],
        [
            '/models/moe',
            *backend._model.backend_parameters,
            '--port=1234',
            f'{shutdown_flag}=60',
        ],
    )
    argv = plan.containers[0].execution.command + (
        plan.containers[0].execution.args or []
    )
    assert argv[:3] == ['vllm', 'serve', '/models/moe']
    assert '--data-parallel-hybrid-lb' in argv
    assert '--data-parallel-external-lb' not in argv
    assert '--api-server-count=3' in argv
    assert argv[argv.index('--data-parallel-size-local') + 1] == '2'
    assert argv[argv.index('--data-parallel-start-rank') + 1] == str(worker_index * 2)
    assert argv[argv.index('--port') + 1] == '40000'
    assert argv[argv.index('--host') + 1] == backend._worker.ip
    assert '--headless' not in argv
    assert manifest.component.kind == 'vllm_hybrid'
    assert manifest.component.api_port == 40000
    assert manifest.component.dp_size == 2
    assert manifest.dp_rank_placements == []
    assert not plan.containers[0].files
    assert {item.name: item.value for item in plan.containers[0].envs}[
        'VLLM_CACHE_ROOT'
    ] == '/cache/vllm'
    assert plan.termination_grace_period_seconds == 75


@pytest.mark.parametrize('worker_index', [0, 1, 2, 3])
@pytest.mark.parametrize('vendor', ['NVIDIA', 'Huawei'])
def test_hybrid_cross_worker_single_fragment_has_true_api_owner(worker_index, vendor):
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    backend, plan = _external_backend(dp=1, vendor=vendor)
    cache_root = '/cache/vllm'
    plan.containers[0].envs = [ContainerEnv(name='VLLM_CACHE_ROOT', value=cache_root)]
    mi = backend._model_instance
    mi.worker_ip = '10.0.0.7'
    mi.gpu_indexes = [1, 3]
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=7 + i, worker_ip=f'10.0.0.{7 + i}', gpu_indexes=[1, 3]
        )
        for i in (1, 2, 3)
    ]
    mi.ports = [40000, 40010, 40020, 40100]
    mi.named_ports = {
        '_vllm_external_api': PortBand(base=40000),
        '_vllm_external_rpc': PortBand(base=40010),
        '_vllm_mp_connecting': PortBand(base=40020, count=10),
        '_vllm_external_store': PortBand(base=40100, count=2),
    }
    backend._model.backend_parameters = [
        '--data-parallel-hybrid-lb',
        '--dp=2',
        '--tp=4',
        '--api-server-count=2',
    ]
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'
    manifest = backend._configure_hybrid_workload(
        plan,
        ['vllm', 'serve'],
        ['/models/moe', *backend._model.backend_parameters, '--shutdown-timeout=60'],
    )
    assert plan.termination_grace_period_seconds == 75
    if worker_index % 2:
        config = json.loads(
            next(
                f.content
                for f in plan.containers[0].files
                if f.path.endswith('tasks.json')
            )
        )
        assert len(config['tasks']) == 1
        key, pool = (
            ('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
            if vendor == 'NVIDIA'
            else ('ASCEND_RT_VISIBLE_DEVICES', '1,3')
        )
        actual = _run_device_task_environment(
            config['tasks'][0], {**os.environ, key: pool, 'VLLM_CACHE_ROOT': cache_root}
        )
        assert actual[key] == pool
        assert actual['VLLM_CACHE_ROOT'] == os.path.join(
            cache_root, 'isolated_devices', '2'
        )
        argv = config['tasks'][0]['argv'][6:]
        assert '--headless' in argv
        assert '--data-parallel-hybrid-lb' not in argv
        assert manifest.component.api_port is None
        assert manifest.component.dp_size is None
    else:
        argv = plan.containers[0].execution.command + (
            plan.containers[0].execution.args or []
        )
        assert '--headless' not in argv
        assert '--data-parallel-hybrid-lb' in argv
        assert manifest.component.api_port == 40000
        assert manifest.component.dp_size == 1
    assert {item.name: item.value for item in plan.containers[0].envs}[
        'VLLM_CACHE_ROOT'
    ] == cache_root
    assert argv[argv.index('--nnodes') + 1] == '4'
    assert argv[argv.index('--node-rank') + 1] == str(worker_index)
    assert argv[argv.index('--master-addr') + 1] == '10.0.0.7'
    assert argv[argv.index('--master-port') + 1] == '40100'
    assert argv[argv.index('--data-parallel-start-rank') + 1] == str(worker_index // 2)


@pytest.mark.parametrize('equal', [False, True])
@pytest.mark.parametrize('composed', [False, True])
@pytest.mark.parametrize('port', [None, 40200, 49999])
def test_hybrid_pd_validates_final_connector_json_without_fixing_engine_identity(
    equal, composed, port
):
    backend, plan = _managed_pd_backend_case('vllm-ascend-mooncake', hybrid=True)
    connector = {
        'kv_connector': 'MooncakeHybridConnector',
        'kv_role': 'kv_producer',
    }
    if port is not None:
        connector['kv_port'] = port
    descriptor = (
        {
            'kv_connector': 'MultiConnector',
            'kv_connector_extra_config': {
                'connectors': [
                    connector,
                    {'kv_connector': 'SimpleCPUOffloadConnector'},
                ],
            },
        }
        if composed
        else connector
    )
    raw = json.dumps(descriptor)
    arguments = ['/models/moe'] + (
        ['--kv-transfer-config=' + raw] if equal else ['--kv-transfer-config', raw]
    )
    backend._pd_engine_identity = lambda *args: pytest.fail(
        'shared API pinned engine identity'
    )
    if port == 49999:
        with pytest.raises(ValueError, match='reserved connector'):
            backend._configure_hybrid_workload(plan, ['vllm', 'serve'], arguments)
        return
    manifest = backend._configure_hybrid_workload(plan, ['vllm', 'serve'], arguments)
    argv = plan.containers[0].execution.command + (
        plan.containers[0].execution.args or []
    )
    actual_raw = next(
        (
            arg.split('=', 1)[1]
            for arg in argv
            if arg.startswith('--kv-transfer-config=')
        ),
        None,
    )
    if actual_raw is None:
        actual_raw = argv[argv.index('--kv-transfer-config') + 1]
    actual = json.loads(actual_raw)
    connector['kv_port'] = 40200
    assert actual == descriptor
    assert manifest.component.dp_size == 2
    assert {env.name: env.value for env in plan.containers[0].envs}[
        'HCCL_IF_IP'
    ] == '10.0.0.7'


@pytest.mark.parametrize('mode', ['vllm-nixl', 'vllm-ascend-mooncake'])
@pytest.mark.parametrize('worker_index', [0, 1])
def test_hybrid_pd_cross_worker_api_and_compute_tasks_keep_local_connector_address(
    mode, worker_index
):
    from gpustack.schemas.models import (
        ModelInstanceSubordinateWorker,
        role_effective_model,
    )

    backend, plan = _managed_pd_backend_case(mode, hybrid=True)
    for role in backend._model_spec.roles:
        role.backend_parameters = ['--data-parallel-hybrid-lb', '--dp=2', '--tp=4']
    backend._model = role_effective_model(backend._model_spec, 'prefill')
    instance = backend._model_instance
    instance.worker_ip = '10.0.0.7'
    instance.gpu_indexes = [1, 3]
    instance.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=7 + index, worker_ip=f'10.0.0.{7 + index}', gpu_indexes=[1, 3]
        )
        for index in (1, 2, 3)
    ]
    instance.named_ports['_vllm_external_store'] = PortBand(base=40300, count=2)
    instance.named_ports['kv_port'] = PortBand(base=40200, count=8)
    devices = backend._get_selected_gpu_devices()
    backend._get_selected_gpu_devices = lambda: devices[:2]
    backend._worker.id = 7 + worker_index
    backend._worker.ip = f'10.0.0.{7 + worker_index}'
    manifest = backend._configure_hybrid_workload(
        plan, ['vllm', 'serve'], ['/models/moe']
    )
    container = plan.containers[0]
    if worker_index:
        tasks = json.loads(
            next(
                file.content
                for file in container.files
                if file.path.endswith('tasks.json')
            )
        )['tasks']
        assert len(tasks) == 1
        argv = tasks[0]['argv']
    else:
        argv = container.execution.command + (container.execution.args or [])
    descriptor = json.loads(argv[argv.index('--kv-transfer-config') + 1])
    assert 'engine_id' not in descriptor
    assert argv[argv.index('--data-parallel-size') + 1] == '2'
    assert argv[argv.index('--data-parallel-size-local') + 1] == '1'
    env = {item.name: item.value for item in container.envs}
    key = (
        'HCCL_IF_IP'
        if mode == 'vllm-ascend-mooncake'
        else 'VLLM_NIXL_SIDE_CHANNEL_HOST'
    )
    assert env[key] == backend._worker.ip
    assert manifest.component.api_port == (40000 if worker_index == 0 else None)


@pytest.mark.parametrize(
    'flag,alias,mode',
    [
        ('--data-parallel-external-lb', '-dpe', 'external'),
        ('--data-parallel-multi-port-external-lb', '-dpm', 'multi_port_external'),
        ('--data-parallel-hybrid-lb', '-dph', 'hybrid'),
    ],
)
@pytest.mark.parametrize('enabled', [False, True])
def test_managed_lb_modes_use_the_last_boolean_value(flag, alias, mode, enabled):
    parameters = [
        f'{flag}={str(not enabled).lower()}',
        f'{alias}={str(enabled).lower()}',
    ]
    assert vllm_lb_mode(parameters) == (mode if enabled else 'internal')
    parameters = [f'{flag}=false', alias]
    assert vllm_lb_mode(parameters) == mode


@pytest.mark.parametrize("backend_class", [SGLangServer, VLLMServer])
@pytest.mark.parametrize("injected", [True, False])
def test_shared_cache_env_resolves_runtime_variables_without_mutating_snapshot(
    backend_class, injected, monkeypatch
):
    from gpustack.schemas.cache_providers import (
        CACHE_ENV_RUNTIME_VARIABLES,
        CacheProvider,
        render_injection,
        validate_injection_templates,
    )

    # One registry entry must support validation and both rendering phases.
    monkeypatch.setitem(CACHE_ENV_RUNTIME_VARIABLES, "worker_name", "name")
    provider = CacheProvider(
        name="runtime-env",
        custom_version=True,
        inference_backend_integrations=[
            {
                "backend": "SGLang" if backend_class is SGLangServer else "vLLM",
                "injection": {
                    "env": {
                        "CACHE_CLIENT_ADDRESS": "{{worker_ip}}:{{port}}",
                        "CACHE_CLIENT_NAME": "{{worker_name}}",
                        "CACHE_MODE": "remote",
                    },
                },
            }
        ],
    )
    assert validate_injection_templates(provider) == []
    env, _, _ = render_injection(
        provider.inference_backend_integrations[0],
        {"worker_ip": "leader-ip", "worker_name": "leader-name", "port": 50051},
    )
    assert env["CACHE_CLIENT_NAME"] == "{{worker_name}}"
    snapshot = CacheConfigSnapshot(cache_service_id=3, injected=injected, env=env)
    before = snapshot.model_dump()
    monkeypatch.setattr(InferenceServer, "_get_configured_env", lambda self: {})
    for ip in ("10.0.0.7", "10.0.0.8"):
        backend = backend_class.__new__(backend_class)
        backend._worker = types.SimpleNamespace(ip=ip, name=f"worker-{ip}")
        backend._model = types.SimpleNamespace(
            extended_kv_cache=_shared_kv_cache_config()
        )
        backend._model_instance = types.SimpleNamespace(cache_config=snapshot)
        if backend_class is SGLangServer:
            backend._set_cache_env = lambda *args: None
            backend._get_selected_gpu_devices = lambda: []
            env = backend._get_configured_env(is_distributed=False)
        else:
            env = {}
            backend._set_lmcache_env(env)
        if injected:
            assert env["CACHE_CLIENT_ADDRESS"] == f"{ip}:50051"
            assert env["CACHE_CLIENT_NAME"] == f"worker-{ip}"
            assert env["CACHE_MODE"] == "remote"
        else:
            assert "CACHE_CLIENT_ADDRESS" not in env
            assert "CACHE_CLIENT_NAME" not in env
            assert "CACHE_MODE" not in env
    assert snapshot.model_dump() == before


@pytest.mark.parametrize('alias', ['--attn-dp-size', '--attention-data-parallel-size'])
@pytest.mark.parametrize('ep', [False, True])
@pytest.mark.parametrize('equal', [False, True])
def test_sglang_dpa_syntax_has_consistent_ranks_and_reserved_ports(alias, ep, equal):
    from gpustack.utils.model_instance_workers import (
        sglang_parallelism,
        sglang_smg_mode,
    )
    from gpustack.worker.backends.sglang import sglang_component_connecting_port_count
    from gpustack.policies.candidate_selectors.sglang_resource_fit_selector import (
        SGLangResourceFitSelector,
    )

    backend = _managed_sglang_backend()
    parameters = ['--tp=8', f'{alias}=8'] if equal else ['--tp', '8', alias, '8']
    if ep:
        parameters += ['--ep=8']
    backend._model.backend_parameters = parameters
    assert sglang_parallelism(parameters) == (8, 8 if ep else 1, True)
    assert sglang_smg_mode(backend._model)
    assert (
        sglang_component_connecting_port_count(backend._model, backend._model_instance)
        == 14
    )
    assert SGLangResourceFitSelector.get_world_size_from_backend_parameters(
        backend._model
    ) == (8, ['tp'])
    backend._model_instance.named_ports['_sglang_connecting'].count = 14
    captured = []
    backend._workload_plan_callback = lambda manifests, plans: captured.extend(
        manifests
    )
    plan = WorkloadPlan(
        name='engine',
        containers=[
            Container(
                name='engine', image='test/sglang', execution=ContainerExecution()
            )
        ],
    )
    backend._prepare_component_workloads(plan, 'test/sglang')
    assert captured[0].component.dp_size == 8


@pytest.mark.parametrize('role', ['prefill', 'decode'])
@pytest.mark.parametrize('worker_index', [0, 1])
def test_multi_port_nixl_pd_inherits_connector_and_reports_global_rank_apis(
    role, worker_index
):
    from gpustack.schemas.models import (
        ModelInstanceSubordinateWorker,
        role_effective_model,
    )

    backend, plan = _managed_pd_backend_case(
        'vllm-nixl', env={'MODEL_API_PORT': '{{port}}'}
    )
    for spec in backend._model_spec.roles:
        spec.backend_parameters = [
            '--data-parallel-multi-port-external-lb',
            '--dp=4',
            '--tp=2',
            '--dpl=2',
        ]
    backend._model = role_effective_model(backend._model_spec, role)
    mi = backend._model_instance
    mi.role, mi.worker_ip = role, '10.0.0.7'
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3, 5, 7]
        )
    ]
    mi.named_ports['_vllm_external_api'] = PortBand(base=40000, count=4)
    mi.named_ports['_vllm_multi_port_external_supervisor'] = PortBand(base=40030)
    mi.named_ports['kv_side_channel'].count = 4
    backend._worker.id, backend._worker.ip = (
        7 + worker_index,
        f'10.0.0.{7 + worker_index}',
    )
    backend._override_entrypoint = lambda command, args, script: (command, args)
    manifest = backend._configure_multi_port_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *backend._model.backend_parameters]
    )
    container = plan.containers[0]
    argv = container.execution.command + (container.execution.args or [])
    descriptor = json.loads(argv[argv.index('--kv-transfer-config') + 1])
    assert descriptor['kv_connector'] == 'NixlConnector'
    assert descriptor['kv_role'] == (
        'kv_producer' if role == 'prefill' else 'kv_consumer'
    )
    assert 'engine_id' not in descriptor
    env = {item.name: item.value for item in container.envs}
    assert env['VLLM_NIXL_SIDE_CHANNEL_HOST'] == backend._worker.ip
    assert env['VLLM_NIXL_SIDE_CHANNEL_PORT'] == '40100'
    assert env['MODEL_API_PORT'] == str(40000 + worker_index * 2)
    assert [(p.dp_rank, p.api_port) for p in manifest.dp_rank_placements] == [
        (rank, 40000 + rank) for rank in range(worker_index * 2, worker_index * 2 + 2)
    ]


def test_external_mooncake_rank_tasks_have_distinct_bootstrap_ports():
    backend, plan = _managed_pd_backend_case('vllm-mooncake')
    backend._model_instance.named_ports['bootstrap'] = PortBand(base=40300, count=2)
    backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *backend._model.backend_parameters]
    )
    tasks = json.loads(
        next(
            file.content
            for file in plan.containers[0].files
            if file.path.endswith('tasks.json')
        )
    )['tasks']
    assert [task['env']['VLLM_MOONCAKE_BOOTSTRAP_PORT'] for task in tasks] == [
        '40300',
        '40301',
    ]


@pytest.mark.parametrize('alias', ['--attn-dp-size', '--attention-data-parallel-size'])
def test_sglang_resolved_old_runner_refuses_new_dpa_before_container_creation(
    monkeypatch, alias
):
    backend = _managed_sglang_backend()
    backend._model.backend_parameters = ['--tp=8', f'{alias}=8']
    backend._get_deployment_metadata = lambda: types.SimpleNamespace(distributed=False)
    backend._get_configured_env = lambda **kwargs: {}

    def resolve_image():
        backend._model.backend_version = '0.5.21'
        return 'gpustack/runner:sglang0.5.21'

    backend._get_configured_image = resolve_image
    monkeypatch.setattr(
        'gpustack.worker.backends.sglang.create_workload',
        lambda *args: pytest.fail('unsupported Runner created a container'),
    )
    with pytest.raises(ValueError, match='0.5.22'):
        backend._start()


@pytest.mark.parametrize('alias', ['--attn-dp-size', '--attention-data-parallel-size'])
@pytest.mark.parametrize('value', ['0', '-1', 'wrong'])
def test_sglang_dpa_requires_a_positive_rank_count(alias, value):
    from gpustack.utils.model_instance_workers import sglang_parallelism

    with pytest.raises(ValueError, match='positive integer'):
        sglang_parallelism([alias, value])


@pytest.mark.parametrize('worker_index', [0, 1])
def test_external_mooncake_cross_worker_bootstrap_ports_follow_global_ranks(
    worker_index,
):
    from gpustack.schemas.models import (
        ModelInstanceSubordinateWorker,
        role_effective_model,
    )

    backend, plan = _managed_pd_backend_case('vllm-mooncake')
    for role in backend._model_spec.roles:
        role.backend_parameters = ['--data-parallel-external-lb', '--dp=4', '--tp=2']
    backend._model = role_effective_model(backend._model_spec, 'prefill')
    mi = backend._model_instance
    mi.worker_ip = '10.0.0.7'
    mi.distributed_servers.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=8, worker_ip='10.0.0.8', gpu_indexes=[1, 3, 5, 7]
        )
    ]
    mi.named_ports['bootstrap'] = PortBand(base=40300, count=4)
    mi.ports = [40000, 40001, 40002, 40003, 40010, 40020]
    backend._worker.id, backend._worker.ip = (
        7 + worker_index,
        f'10.0.0.{7 + worker_index}',
    )
    backend._configure_external_workload(
        plan, ['vllm', 'serve'], ['/models/moe', *backend._model.backend_parameters]
    )
    tasks = json.loads(
        next(
            f.content for f in plan.containers[0].files if f.path.endswith('tasks.json')
        )
    )['tasks']
    assert [t['env']['VLLM_MOONCAKE_BOOTSTRAP_PORT'] for t in tasks] == [
        str(40300 + rank) for rank in range(worker_index * 2, worker_index * 2 + 2)
    ]


def test_multi_port_pd_refuses_ascend_even_with_a_nixl_mode():
    backend, plan = _managed_pd_backend_case('vllm-nixl')
    for role in backend._model_spec.roles:
        role.backend_parameters = [
            '--data-parallel-multi-port-external-lb',
            '--dp=2',
            '--tp=2',
        ]
    from gpustack.schemas.models import role_effective_model

    backend._model = role_effective_model(backend._model_spec, 'prefill')
    devices = backend._get_selected_gpu_devices()
    for device in devices:
        device.type = 'cann'
    backend._get_selected_gpu_devices = lambda: devices
    with pytest.raises(ValueError, match='only CUDA NIXL'):
        backend._configure_multi_port_external_workload(
            plan, ['vllm', 'serve'], ['/models/moe', *backend._model.backend_parameters]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('role', ['prefill', 'decode'])
async def test_new_pd_multi_port_members_start_with_managed_workloads(
    monkeypatch, role
):
    from unittest.mock import AsyncMock, MagicMock
    from gpustack.schemas.models import Model, RoleSpec, DisaggregationSpec
    from gpustack.server import controllers

    model = Model(
        id=1,
        name='pd',
        source='huggingface',
        huggingface_repo_id='org/moe',
        backend='vLLM',
        disaggregation=DisaggregationSpec(mode='vllm-nixl'),
        roles=[
            RoleSpec(
                name=n,
                backend_parameters=[
                    '--data-parallel-multi-port-external-lb',
                    '--dp=2',
                    '--tp=1',
                ],
            )
            for n in ['prefill', 'decode']
        ],
    )
    monkeypatch.setattr(
        controllers, 'get_draft_model_source', AsyncMock(return_value=None)
    )
    row = await controllers._build_instance_create(
        MagicMock(),
        model,
        next(r for r in model.roles if r.name == role),
        'g1',
        'digest',
    )
    assert row.distributed_servers is not None
    assert row.distributed_servers.workloads == []


@pytest.mark.parametrize('manual_dp', ['0', '1'])
@pytest.mark.parametrize(
    'parameters',
    [
        ['--data-parallel-external-lb', '--data-parallel-rank=1'],
        ['--data-parallel-hybrid-lb', '--data-parallel-start-rank=1'],
        ['--data-parallel-external-lb', '--data-parallel-start-rank=1'],
    ],
)
def test_legacy_vllm_arguments_preserve_user_dp_group(manual_dp, parameters):
    backend = _vllm_backend_for_composition(None, [], [])
    backend._model.env = {'GPUSTACK_MANUAL_DP': manual_dp}
    backend._model.backend_parameters = [
        *parameters,
        '--tp=8',
        '--dp=2',
        '--dpl=1',
        '--data-parallel-address=10.0.0.9',
        '--data-parallel-rpc-port=13345',
        '--master-addr=10.0.0.9',
        '--master-port=29500',
    ]
    backend._flatten_backend_param = lambda: list(backend._model.backend_parameters)
    arguments, injected = backend._build_command_args(port=4000, is_distributed=False)
    for parameter in backend._model.backend_parameters:
        assert arguments.count(parameter) == 1
        assert parameter not in injected
    assert '--data-parallel-rank' not in injected
    assert '--nnodes' not in injected


@pytest.mark.parametrize(
    'option,value',
    [
        ('--chat-template', 'hello world'),
        ('--tokenizer', '/models/my tokenizer'),
    ],
)
@pytest.mark.parametrize('equal', [False, True])
def test_managed_command_preserves_tokenized_option_values(option, value, equal):
    from gpustack.utils.command import flatten_to_argv

    raw = [f"{option}='{value}'"] if equal else [option, value]
    arguments = ['/models/moe', *flatten_to_argv(raw)]
    assert VLLMServer._external_command(['vllm', 'serve'], arguments) == [
        'vllm',
        'serve',
        *arguments,
    ]
