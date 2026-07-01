import pytest

from gpustack.utils.vllm_topology import (
    subordinates_serve_api,
    resolve_data_parallel_load_balance_mode,
    matched_manual_distributed_params,
)


@pytest.mark.parametrize(
    "parameters, expected",
    [
        # hybrid-lb flag.
        (['--data-parallel-hybrid-lb'], True),
        (['--data-parallel-size', '4', '--data-parallel-hybrid-lb'], True),
        # external-lb flag.
        (['--data-parallel-external-lb'], True),
        (['--data-parallel-size=4', '--data-parallel-external-lb'], True),
        # Explicit rank implies external-lb in vLLM, even rank 0 (falsy value).
        (['--data-parallel-rank', '0'], True),
        (['--data-parallel-rank', '3'], True),
        # Plain internal-lb DP must NOT be treated as per-rank API.
        (['--data-parallel-size', '4'], False),
        # Nothing relevant.
        (['--tp', '8'], False),
        (None, False),
    ],
)
def test_subordinates_serve_api(parameters, expected):
    assert subordinates_serve_api(parameters) is expected


@pytest.mark.parametrize(
    "parameters, expected",
    [
        # Raw flags map to their mode.
        (['--data-parallel-hybrid-lb'], "hybrid"),
        (['--data-parallel-external-lb'], "external"),
        (['--data-parallel-rank', '0'], "external"),
        # Defaults to internal.
        (['--data-parallel-size', '4'], "internal"),
        ([], "internal"),
    ],
)
def test_resolve_data_parallel_load_balance_mode(parameters, expected):
    assert resolve_data_parallel_load_balance_mode(parameters) == expected


@pytest.mark.parametrize(
    "parameters, expected",
    [
        # Hand-wired DP cluster signals, echoed back as --flags.
        (['--data-parallel-address=1.2.3.4'], ['--data-parallel-address']),
        (['--node-rank', '1'], ['--node-rank']),
        (['--data-parallel-start-rank=2'], ['--data-parallel-start-rank']),
        (['--data-parallel-rank', '0'], ['--data-parallel-rank']),
        # Multiple matches keep the declared order.
        (
            ['--node-rank', '1', '--data-parallel-address=1.2.3.4'],
            ['--data-parallel-address', '--node-rank'],
        ),
        # Plain DP sizing is not manual wiring.
        (['--data-parallel-size', '4'], []),
        (None, []),
    ],
)
def test_matched_manual_distributed_params(parameters, expected):
    assert matched_manual_distributed_params(parameters) == expected
