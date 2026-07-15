import asyncio

import pytest

import gpustack.server.metrics_collector as mc


@pytest.mark.asyncio
async def test_flush_gateway_metrics_loop_survives_error(monkeypatch):
    """A flush error must not escape the loop. If it did, the exception would
    propagate through the server's asyncio.gather and crash the whole process;
    the loop should log and retry on the next tick.
    """
    calls = {"flush": 0, "sleep": 0}

    async def _boom():
        calls["flush"] += 1
        raise RuntimeError("db unavailable")

    async def _sleep(_seconds):
        calls["sleep"] += 1
        # End the otherwise-infinite loop the way a real shutdown would.
        if calls["sleep"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(mc, "flush_gateway_metrics", _boom)
    monkeypatch.setattr("gpustack.server.metrics_collector.asyncio.sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await mc.flush_gateway_metrics_to_db()

    # Iteration 1: sleep ok -> flush raised -> swallowed; iteration 2: sleep
    # raises CancelledError, which propagates (clean shutdown).
    assert calls["flush"] == 1
