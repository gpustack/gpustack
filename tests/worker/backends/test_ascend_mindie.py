import argparse

from gpustack.worker.backends.ascend_mindie import AscendMindIEParameters
import pytest


@pytest.mark.parametrize(
    "world_size, local_world_size, args, expected",
    [
        # The following cases are a forward derivation, which means that
        # the world size is not provided,
        # and is determined by input parameters.
        [
            -1,
            -1,
            ["--pipeline-parallel-size=2", "--tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                pipeline_parallel_size=2,
                tensor_parallel_size=8,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            -1,
            -1,
            ["--tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=8,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=8,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            -1,
            -1,
            ["--data-parallel-size=2", "--tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                data_parallel_size=2,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=16,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            -1,
            -1,
            ["--context-parallel-size=2", "--tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                context_parallel_size=2,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=16,
                data_parallel_size=1,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            -1,
            -1,
            ["--moe-expert-parallel-size=2", "--moe-tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                tensor_parallel_size=8,
                moe_expert_parallel_size=2,
                moe_tensor_parallel_size=8,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        # The following cases are a backward derivation, which means that
        # the world size is provided,
        # and provided partial parameters.
        [
            16,
            8,
            ["--pipeline-parallel-size=2"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                pipeline_parallel_size=2,
                tensor_parallel_size=8,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            16,
            8,
            ["--tensor-parallel-size=8"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=16,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            16,
            8,
            ["--data-parallel-size=2"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                data_parallel_size=2,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=16,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            16,
            8,
            ["--context-parallel-size=2"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                context_parallel_size=2,
                tensor_parallel_size=8,
                moe_tensor_parallel_size=16,
                data_parallel_size=1,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
        [
            16,
            8,
            ["--moe-expert-parallel-size=2"],
            AscendMindIEParameters(
                world_size=16,
                local_world_size=8,
                moe_expert_parallel_size=2,
                tensor_parallel_size=16,
                moe_tensor_parallel_size=8,
                max_prefill_tokens=8192,
                max_input_token_len=8192,
                max_iter_times=8192,
            ),
        ],
    ],
)
@pytest.mark.asyncio
async def test_ascend_mindie_parameters_parallelism_default(
    world_size, local_world_size, args, expected: AscendMindIEParameters
):
    actual = AscendMindIEParameters(
        world_size=world_size,
        local_world_size=local_world_size,
    )
    actual.from_args(args)
    assert actual == expected


@pytest.mark.parametrize(
    "world_size, local_world_size, args, exception_msg",
    [
        # The following cases are a forward derivation, which means that
        # the world size is not provided,
        # and is determined by input parameters.
        [
            -1,
            -1,
            ["--pipeline-parallel-size=-1"],
            "--pipeline-parallel-size must be greater than 0",
        ],
        [
            -1,
            -1,
            ["--tensor-parallel-size=3"],
            "--tensor-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--data-parallel-size=3"],
            "--data-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--context-parallel-size=3"],
            "--context-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--sequence-parallel-size=3"],
            "--sequence-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--moe-tensor-parallel-size=3"],
            "--moe-tensor-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--moe-expert-parallel-size=3"],
            "--moe-expert-parallel-size must be the power of 2",
        ],
        [
            -1,
            -1,
            ["--pipeline-parallel-size=2", "--data-parallel-size=4"],
            "--pipeline-parallel-size 2 and --data-parallel-size 4 are incompatible, set --pipeline-parallel-size to 1 or disable data parallelism",
        ],
        [
            -1,
            -1,
            ["--data-parallel-size=4", "--context-parallel-size=2"],
            "--data-parallel-size 4 and --context-parallel-size 2 are incompatible, set --data-parallel-size to 1 or disable context parallelism",
        ],
        [
            -1,
            -1,
            ["--sequence-parallel-size=4", "--tensor-parallel-size=2"],
            "--sequence-parallel-size 4 must be equal to --tensor-parallel-size 2",
        ],
        [
            -1,
            -1,
            [
                "--data-parallel-size=4",
                "--tensor-parallel-size=2",
            ],  # DP and TP are compatible
            "",  # No exception expected
        ],
        [
            -1,
            -1,
            [
                "--context-parallel-size=2",
                "--tensor-parallel-size=4",
            ],  # CP and TP are compatible
            "",  # No exception expected
        ],
        [
            -1,
            -1,
            [
                "--sequence-parallel-size=4",
                "--tensor-parallel-size=4",
            ],  # SP and TP are compatible
            "",  # No exception expected
        ],
        # The following cases are a backward derivation, which means that
        # the world size is provided,
        # and provided partial parameters.
        # These situations should not normally occur,
        # if they do, it means we have made the wrong choice in resource selection.
        [
            4,
            4,
            ["--pipeline-parallel-size=2", "--tensor-parallel-size=4"],
            "--pipeline-parallel-size 2 and --tensor-parallel-size 4 must be multiples of world size: 4",
        ],
        [
            16,
            4,
            ["--tensor-parallel-size=8"],
            "--tensor-parallel-size 8 must be less or equal to local world size: 4 or equal to world size: 16",
        ],
        [
            32,
            8,
            ["--data-parallel-size=2", "--tensor-parallel-size=8"],
            "--data-parallel-size 2 and --tensor-parallel-size 8 must be multiples of world size: 32",
        ],
        [
            32,
            8,
            ["--context-parallel-size=2", "--tensor-parallel-size=8"],
            "--context-parallel-size 2 and --tensor-parallel-size 8 must be multiples of world size: 32",
        ],
        [
            16,
            4,
            ["--moe-expert-parallel-size=4", "--moe-tensor-parallel-size=8"],
            "--moe-tensor-parallel-size 8 must be less or equal to local world size: 4 or equal to world size: 16",
        ],
        [
            16,
            8,
            ["--moe-expert-parallel-size=4", "--moe-tensor-parallel-size=8"],
            "--moe-expert-parallel-size 4and --moe-tensor-parallel-size 8 must be multiples of world size: 16",
        ],
        [
            32,
            8,
            ["--moe-tensor-parallel-size=8"],
            "--moe-tensor-parallel-size 8 must be equal to world size: 32",
        ],
    ],
)
@pytest.mark.asyncio
async def test_ascend_mindie_parameters_parallelism_violation(
    world_size,
    local_world_size,
    args,
    exception_msg: str,
):
    """
    Test AscendMindIEParameters.from_args for various parallelism violations.
    """
    if not exception_msg:
        # No exception expected
        params = AscendMindIEParameters(
            world_size=world_size,
            local_world_size=local_world_size,
        )
        params.from_args(args)
        return

    with pytest.raises(argparse.ArgumentTypeError, match=exception_msg):
        params = AscendMindIEParameters(
            world_size=world_size,
            local_world_size=local_world_size,
        )
        params.from_args(args)
