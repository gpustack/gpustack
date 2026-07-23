import pytest

from gpustack.utils.vllm_topology import (
    subordinates_serve_api,
    resolve_data_parallel_load_balance_mode,
    matched_manual_distributed_params,
    derive_dp_node_count,
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


@pytest.mark.parametrize(
    "backend, parameters, distributed, expected",
    [
        # vLLM external / hybrid-LB distributed -> DP node-per-instance.
        ("vLLM", ["--data-parallel-external-lb"], True, True),
        ("vLLM", ["--data-parallel-hybrid-lb"], True, True),
        # internal-LB, non-distributed, and non-vLLM backends are excluded.
        ("vLLM", ["--data-parallel-size", "8"], True, False),
        ("vLLM", ["--data-parallel-external-lb"], False, False),
        ("SGLang", ["--data-parallel-external-lb"], True, False),
    ],
)
def test_is_dp_node_per_instance(backend, parameters, distributed, expected):
    from gpustack.schemas.models import Model, is_dp_node_per_instance

    model = Model(
        name="m",
        backend=backend,
        backend_parameters=parameters,
        distributed_inference_across_workers=distributed,
    )
    assert is_dp_node_per_instance(model) is expected


@pytest.mark.parametrize(
    "parameters, expected",
    [
        # external-LB pins one rank per node -> replicas == dp.
        (['--data-parallel-external-lb', '--data-parallel-size=4'], 4),
        # hybrid-LB packs dpl ranks per node -> replicas == dp // dpl.
        (
            [
                '--data-parallel-hybrid-lb',
                '--data-parallel-size=8',
                '--data-parallel-size-local=2',
            ],
            4,
        ),
        # Not external/hybrid-LB (internal-LB / no LB flag / nothing) -> None.
        (['--data-parallel-size', '4'], None),
        (['--tp', '8'], None),
        (None, None),
    ],
)
def test_derive_dp_node_count(parameters, expected):
    assert derive_dp_node_count(parameters) == expected


@pytest.mark.parametrize(
    "parameters",
    [
        # external-LB needs dp > 1 to derive a node count.
        ['--data-parallel-external-lb'],
        ['--data-parallel-external-lb', '--data-parallel-size=1'],
        # hybrid-LB needs dpl (per-node rank count) and dp % dpl == 0.
        ['--data-parallel-hybrid-lb', '--data-parallel-size=8'],
        [
            '--data-parallel-hybrid-lb',
            '--data-parallel-size=7',
            '--data-parallel-size-local=2',
        ],
    ],
)
def test_derive_dp_node_count_insufficient_params_raise(parameters):
    with pytest.raises(ValueError):
        derive_dp_node_count(parameters)


def test_get_deployment_metadata_dp_node_per_instance():
    """DP node instances derive their distributed role from dp_rank alone (no
    subordinate workers, no follower-index name mutation)."""
    from gpustack.schemas.models import ModelInstance

    leader = ModelInstance(
        name="m-0", model_id=1, model_name="m", worker_id=7, dp_rank=0
    )
    meta = leader.get_deployment_metadata(7)
    assert (
        meta.distributed and meta.distributed_leader and not meta.distributed_follower
    )
    assert meta.dp_rank == 0 and meta.distributed_follower_index is None
    assert meta.name == "m-0"  # not mutated with -f suffix

    member = ModelInstance(
        name="m-2", model_id=1, model_name="m", worker_id=9, dp_rank=2
    )
    member_meta = member.get_deployment_metadata(9)
    assert member_meta.distributed_follower and not member_meta.distributed_leader
    assert member_meta.dp_rank == 2
    # Not served by this worker -> no metadata.
    assert member.get_deployment_metadata(999) is None
