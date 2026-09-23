"""`GET /v2/benchmarks?worker_id=N` must actually filter.

Every worker's 3-second state poll sends `worker_id`, believing it gets back
only its own runs. FastAPI drops an undeclared query parameter silently rather
than rejecting it, so a parameter missing from the route hands each worker the
whole cluster's RUNNING rows to reconcile.

For a run it does not own that reconcile is wrong at every step -- the workload
lives on the owning host, so the local runtime reports it missing -- and it ends
by patching the row to ERROR and tearing the workload down under the worker
that owns it, mid-run.

The assertions below are on `fields` rather than on returned rows because
`fields` is what both consumers of this filter read: `paginated_by_query` for
the poll and `Benchmark.streaming` for the watch. A fix that only reached the
poll would leave the watch cluster-wide, which is the mistake worth pinning.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.routes import benchmarks as route
from gpustack.schemas.benchmark import BenchmarkStateEnum


def _params(watch=False):
    return SimpleNamespace(watch=watch, page=1, perPage=100, order_by=None)


def _ctx():
    return SimpleNamespace(current_principal_id=None)


async def _call(**kwargs):
    """Run `_get_benchmarks` and hand back the `fields` it built."""
    captured = {}

    async def _fake_paginated(**call_kwargs):
        captured.update(call_kwargs)
        return SimpleNamespace(items=[])

    with (
        patch.object(
            route.Benchmark,
            "paginated_by_query",
            AsyncMock(side_effect=_fake_paginated),
        ),
        patch.object(route, "tenant_list_conditions", lambda *_a, **_k: []),
        patch.object(route, "async_session"),
    ):
        await route._get_benchmarks(ctx=_ctx(), params=_params(), **kwargs)

    return captured.get("fields", {})


@pytest.mark.asyncio
async def test_worker_id_reaches_the_query():
    fields = await _call(worker_id=4)

    assert fields.get("worker_id") == 4


@pytest.mark.asyncio
async def test_worker_id_is_absent_when_not_asked_for():
    """Omitting it must stay cluster-wide -- the UI's list has no worker filter."""
    fields = await _call()

    assert "worker_id" not in fields


@pytest.mark.asyncio
async def test_worker_id_combines_with_state():
    """The poll sends both. Dropping either one is what made D17 reachable:
    without `state` it reconciles finished runs, without `worker_id` it
    reconciles other hosts' runs."""
    fields = await _call(worker_id=3, state=BenchmarkStateEnum.RUNNING)

    assert fields.get("worker_id") == 3
    assert fields.get("state") == BenchmarkStateEnum.RUNNING


@pytest.mark.asyncio
async def test_worker_id_zero_is_a_filter_not_a_falsy_skip():
    """Guarded with `is not None`, so id 0 filters rather than silently
    widening to the whole cluster."""
    fields = await _call(worker_id=0)

    assert fields.get("worker_id") == 0


def test_the_route_declares_worker_id():
    """FastAPI ignores query parameters a route does not declare, which is how
    this went unnoticed for the life of the poll. Assert the signature itself."""
    import inspect

    sig = inspect.signature(route.get_benchmarks)

    assert "worker_id" in sig.parameters
