"""Shipped presets and Custom load shapes retain their execution semantics."""

import json
from pathlib import Path

import pytest
import yaml

from gpustack.routes.benchmark_profiles import get_builtin_profiles_config_file_path
from gpustack.routes.benchmarks import _validate_load_config
from gpustack.schemas.benchmark import (
    Benchmark,
    BenchmarkCreate,
    BenchmarkLoadModeEnum,
    SLO_THRESHOLDS,
    benchmark_load_mode,
)
from gpustack.worker.benchmark.runner import BenchmarkRunner


def _presets():
    config = yaml.safe_load(Path(get_builtin_profiles_config_file_path()).read_text())
    return config['profiles']


def _benchmark(**fields):
    payload = {
        'name': 'profile-test',
        'model_instance_name': 'instance',
        'model_name': 'model',
        'dataset_name': 'Random',
        'dataset_input_tokens': 128,
        'dataset_output_tokens': 128,
        'dataset_seed': 42,
        'total_requests': 123,
        'max_seconds': 9,
        **fields,
    }
    validated = BenchmarkCreate(**payload)
    _validate_load_config(validated)
    return Benchmark(id=1, **validated.model_dump())


def _args(benchmark):
    runner = object.__new__(BenchmarkRunner)
    runner._benchmark = benchmark
    runner._model_name = 'model'
    runner._model_endpoint = 'http://127.0.0.1:8000'
    runner._model_path = '/models/model'
    runner._model_backend_parameters = []
    runner._benchmark_dir = '/benchmarks'
    runner._api_url = 'http://127.0.0.1:9000/v2/benchmarks/1/state'
    runner._api_key = 'test-token'
    runner._progress_insecure_skip_tls_verify = False
    return runner._build_command_args()


@pytest.mark.parametrize('preset', _presets(), ids=lambda preset: preset['name'])
def test_shipped_presets_forward_their_axis_budget_and_actual_slo(preset):
    fields = {**preset, 'profile': preset['name']}
    fields.pop('name')
    benchmark = _benchmark(**fields)
    args = _args(benchmark)

    assert benchmark_load_mode(benchmark) is BenchmarkLoadModeEnum.AUTO_TUNE
    assert '--auto-tune' in args
    expected_axis = 'rate' if preset['load_type'] == 'fixed_rate' else 'concurrency'
    assert args[args.index('--axis') + 1] == expected_axis
    for field in ('lower_bound', 'upper_bound', 'max_points', 'max_total_seconds'):
        assert (
            float(args[args.index('--' + field.replace('_', '-')) + 1]) == preset[field]
        )
    for threshold in SLO_THRESHOLDS:
        if threshold.attr in preset:
            assert float(args[args.index(threshold.flag) + 1]) == preset[threshold.attr]
        else:
            assert getattr(benchmark, threshold.attr) is None
            assert threshold.flag not in args
    assert '--max-requests' not in args and '--max-seconds' not in args
    assert '--profile' not in args and '--stages' not in args
    assert args[args.index('--data') + 1] == (
        f"prompt_tokens={preset['dataset_input_tokens']},"
        f"output_tokens={preset['dataset_output_tokens']}"
    )


@pytest.mark.parametrize('axis', ['fixed_rate', 'concurrency'])
@pytest.mark.parametrize('mode', list(BenchmarkLoadModeEnum))
def test_custom_shapes_preserve_load_axis_and_per_stage_constraints(axis, mode):
    rate = 0.5 if axis == 'fixed_rate' else 4
    stages = [
        {'rate': rate, 'max_requests': 12},
        {'rate': rate * 2, 'max_seconds': 3.5},
    ]
    fields = {'profile': 'Custom', 'load_type': axis, 'request_rate': 4}
    if mode is BenchmarkLoadModeEnum.AUTO_TUNE:
        fields.update(auto_tune=True, lower_bound=rate, upper_bound=16)
    elif mode is BenchmarkLoadModeEnum.STAGES:
        fields['stages'] = stages
    benchmark = _benchmark(**fields)
    args = _args(benchmark)

    assert benchmark_load_mode(benchmark) is mode
    if mode is BenchmarkLoadModeEnum.SINGLE:
        assert args[args.index('--profile') + 1] == (
            'constant' if axis == 'fixed_rate' else 'concurrent'
        )
        assert args[args.index('--rate') + 1] == '4'
        assert args[args.index('--max-requests') + 1] == '123'
        assert args[args.index('--max-seconds') + 1] == '9.0'
    else:
        assert args[args.index('--axis') + 1] == (
            'rate' if axis == 'fixed_rate' else 'concurrency'
        )
        assert '--max-requests' not in args and '--max-seconds' not in args
        assert '--profile' not in args and '--rate' not in args
        if mode is BenchmarkLoadModeEnum.STAGES:
            assert json.loads(args[args.index('--stages') + 1]) == stages
        else:
            assert '--auto-tune' in args and '--stages' not in args
