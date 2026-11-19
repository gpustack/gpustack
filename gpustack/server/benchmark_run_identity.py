"""Fence Benchmark reports to the execution that produced them."""

from typing import Optional

from gpustack.api.exceptions import ConflictException


def assert_benchmark_run(
    benchmark, submitted_run_id: Optional[str], *, allow_unscoped: bool = False
) -> None:
    """Reject stale writers while retaining unidentified legacy executions.

    Human stop actions can omit an execution token. An explicitly submitted
    token must always match, including on those actions.
    """
    expected = getattr(benchmark, 'run_id', None)
    if submitted_run_id == expected:
        return
    if submitted_run_id is None and allow_unscoped:
        return
    raise ConflictException(
        message='Benchmark execution changed; reload the current execution.',
        details={'code': 'benchmark_run_changed'},
    )
