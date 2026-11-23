"""When a DP replica does not fit, one replica is sharded across the GPUs."""

import pytest

from gpustack.utils.vllm_topology import (
    MultinodeUserParallelism,
    auto_shard_override,
)

_GIB = 1024**3


def test_auto_shard_uses_tp_when_a_dp_replica_does_not_fit():
    """Two 1-GPU nodes whose sum covers the model, but neither node does."""
    override = auto_shard_override(
        [1, 1],
        [10 * _GIB, 10 * _GIB],
        16 * _GIB,
        tp_divisible=lambda size: size in (1, 2),
    )
    assert override == MultinodeUserParallelism(tp=2)


def test_auto_shard_keeps_dp_when_each_node_can_hold_a_replica():
    assert (
        auto_shard_override(
            [8, 8],
            [40 * _GIB, 40 * _GIB],
            20 * _GIB,
        )
        is None
    )


def test_auto_shard_requires_explicit_pipeline_when_full_tp_is_illegal():
    with pytest.raises(ValueError, match="pipeline-parallel layout explicitly"):
        auto_shard_override(
            [8, 8],
            [40 * _GIB, 40 * _GIB],
            70 * _GIB,
            tp_divisible=lambda size: size == 8,
        )


def test_auto_shard_ignores_an_explicit_user_layout():
    assert (
        auto_shard_override(
            [1, 1],
            [10 * _GIB, 10 * _GIB],
            16 * _GIB,
            MultinodeUserParallelism(tp=1, dp=2),
        )
        is None
    )


def test_auto_shard_rejects_uneven_gpu_counts_that_cannot_share_a_replica():
    with pytest.raises(ValueError, match="not equal"):
        auto_shard_override([2, 1], [40 * _GIB, 20 * _GIB], 30 * _GIB)


def test_auto_shard_rejects_a_node_that_cannot_hold_its_share():
    """Equal GPU counts, but the small node's reservation is under an even split."""
    with pytest.raises(ValueError, match="smallest reservation"):
        auto_shard_override(
            [1, 1],
            [80 * _GIB, 10 * _GIB],
            30 * _GIB,
            tp_divisible=lambda size: size in (1, 2),
        )


def test_auto_shard_keeps_tp_when_the_smaller_gpu_can_hold_its_share():
    """The smaller budget still covers an even share, so full-width TP is kept."""
    override = auto_shard_override(
        [1, 1],
        [14495514624, 11596411699],
        23188691062,
        tp_divisible=lambda size: size in (1, 2),
    )
    assert override == MultinodeUserParallelism(tp=2)
