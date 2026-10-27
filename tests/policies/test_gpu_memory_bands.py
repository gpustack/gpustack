"""Banding workers by the size of the GPU they carry.

The size a member's VRAM reservation is a fraction of is the card's *total*, so
that is what bands a fleet: two workers are interchangeable for one role when
their biggest card is the same size, whatever either card is busy with.
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.utils import (
    group_workers_by_gpu_memory_size,
    worker_largest_gpu_memory,
)

GIB = 1024**3


def _worker(worker_id: int, *gpu_gib: float):
    """A worker reporting one GPU per entry in `gpu_gib`."""
    return SimpleNamespace(
        id=worker_id,
        name=f"w{worker_id}",
        status=SimpleNamespace(
            gpu_devices=[
                SimpleNamespace(memory=SimpleNamespace(total=int(size * GIB)))
                for size in gpu_gib
            ]
        ),
    )


def _blind(worker_id: int):
    """A worker whose GPU telemetry says nothing about memory."""
    return SimpleNamespace(
        id=worker_id,
        name=f"w{worker_id}",
        status=SimpleNamespace(gpu_devices=[SimpleNamespace(memory=None)]),
    )


@pytest.mark.parametrize(
    "workers, expected",
    [
        pytest.param([], [], id="nothing-to-band"),
        pytest.param(
            [_worker(1, 48), _worker(2, 48), _worker(3, 48)],
            [[1, 2, 3]],
            id="one-size-is-one-band",
        ),
        pytest.param(
            [_worker(1, 48), _worker(2, 32), _worker(3, 32)],
            [[2, 3], [1]],
            id="two-sizes-smallest-band-first",
        ),
        pytest.param(
            [_worker(1, 48), _worker(2, 47.4)],
            [[2, 1]],
            id="boards-of-one-model-vary-within-tolerance",
        ),
        pytest.param(
            [_worker(1, 48), _worker(2, 40)],
            [[2], [1]],
            id="beyond-tolerance-is-a-different-card",
        ),
        pytest.param(
            [_worker(1, 24, 48), _worker(2, 48)],
            [[1, 2]],
            id="the-biggest-card-speaks-for-a-mixed-worker",
        ),
        pytest.param(
            [_worker(1, 48), _blind(2), _worker(3, 32)],
            [[3], [1]],
            id="an-unreadable-size-is-not-a-band",
        ),
        pytest.param(
            [_worker(1, 48), SimpleNamespace(id=2, name="w2", status=None)],
            [[1]],
            id="a-worker-with-no-telemetry-is-not-a-band",
        ),
    ],
)
def test_bands_hold_the_workers_of_one_card_size(workers, expected):
    bands = group_workers_by_gpu_memory_size(workers)

    assert [[w.id for w in band] for band in bands] == expected


@pytest.mark.parametrize(
    "worker, expected",
    [
        pytest.param(_worker(1, 24, 48, 24), 48 * GIB, id="the-biggest-of-several"),
        pytest.param(_worker(1, 32), 32 * GIB, id="the-only-one"),
        pytest.param(_blind(1), None, id="no-memory-reported"),
        pytest.param(_worker(1), None, id="no-gpu-reported"),
    ],
)
def test_a_worker_is_represented_by_its_biggest_card(worker, expected):
    assert worker_largest_gpu_memory(worker) == expected
