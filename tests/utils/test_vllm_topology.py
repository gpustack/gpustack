import pytest

from gpustack.utils.vllm_topology import (
    subordinates_serve_api,
    resolve_data_parallel_load_balance_mode,
    has_manual_distributed_params,
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
        # Hand-wired DP cluster signals.
        (['--data-parallel-address=1.2.3.4'], True),
        (['--node-rank', '1'], True),
        (['--data-parallel-start-rank=2'], True),
        (['--data-parallel-rank', '0'], True),
        # Plain DP sizing is not manual wiring.
        (['--data-parallel-size', '4'], False),
        (None, False),
    ],
)
def test_has_manual_distributed_params(parameters, expected):
    assert has_manual_distributed_params(parameters) is expected
