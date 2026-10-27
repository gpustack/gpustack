"""Where a benchmark report's monitoring link goes, and over which interval.

The client-side numbers say a run was slow; they cannot say where the time
went. Under PD that is three separate places -- the prefill queue, the decode
queue, and the KV transfer between them -- and only the PD dashboard shows all
three, which is also why a group must not be linked to the model dashboard
whose request counters double for it.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.routes import benchmarks as route
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.schemas.models import DisaggregationSpec, Model, PDModeEnum

_STARTED = datetime(2026, 9, 7, 11, 0, tzinfo=timezone.utc)
_ENDED = _STARTED + timedelta(minutes=30)


def _cfg():
    return SimpleNamespace(
        get_grafana_url=lambda: "http://grafana:3000",
        grafana_model_dashboard_uid="gpustack-model",
        grafana_pd_dashboard_uid="gpustack-pd",
    )


def _benchmark(**kwargs):
    return SimpleNamespace(
        id=1,
        model_id=1,
        cluster_id=1,
        created_at=kwargs.pop("created_at", _STARTED),
        updated_at=kwargs.pop("updated_at", _ENDED),
        state=kwargs.pop("state", BenchmarkStateEnum.COMPLETED),
        **kwargs,
    )


def _model(pd=False):
    return Model(
        id=1,
        name="m",
        cluster_id=1,
        replicas=1,
        disaggregation=(DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL) if pd else None),
    )


async def _redirect(benchmark, model, cfg=None):
    with (
        patch.object(route.Benchmark, "one_by_id", AsyncMock(return_value=benchmark)),
        patch.object(route, "assert_resource_visible", lambda *a, **k: None),
        patch.object(route.Model, "one_by_id", AsyncMock(return_value=model)),
        patch.object(
            route.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(name="x1")),
        ),
        patch.object(route, "get_global_config", lambda: cfg or _cfg()),
        patch.object(
            route, "resolve_grafana_base_url", lambda *_: "http://grafana:3000"
        ),
        patch("gpustack.server.pd_mode_catalog.get_pd_mode", lambda _: None),
    ):
        response = await route.get_benchmark_dashboard(
            session=None, ctx=None, id=1, request=None
        )
    return response.headers["location"]


@pytest.mark.asyncio
async def test_a_group_is_sent_to_the_pd_dashboard():
    location = await _redirect(_benchmark(), _model(pd=True))
    assert "/d/gpustack-pd/gpustack-pd" in location
    # Without this the reader gets request counters that are 2x reality.
    assert "var-counted_role=" in location


@pytest.mark.asyncio
async def test_a_plain_model_is_sent_to_the_model_dashboard():
    location = await _redirect(_benchmark(), _model())
    assert "/d/gpustack-model/gpustack-model" in location
    assert "var-model_name=m" in location


@pytest.mark.asyncio
async def test_the_window_is_the_run_not_now():
    # A report read a day later is about an interval that has passed; a link
    # opening on the last six hours shows an idle deployment.
    location = await _redirect(_benchmark(), _model())
    padded_from = int((_STARTED - route._DASHBOARD_PADDING).timestamp() * 1000)
    padded_to = int((_ENDED + route._DASHBOARD_PADDING).timestamp() * 1000)
    assert f"from={padded_from}" in location
    assert f"to={padded_to}" in location


@pytest.mark.asyncio
async def test_a_running_benchmark_keeps_the_right_edge_open():
    # The reader is watching it happen; a window ending seconds ago looks
    # frozen.
    location = await _redirect(_benchmark(state=BenchmarkStateEnum.RUNNING), _model())
    assert "to=now" in location


@pytest.mark.asyncio
async def test_an_unconfigured_grafana_says_so():
    from gpustack.api.exceptions import InternalServerErrorException

    cfg = SimpleNamespace(
        get_grafana_url=lambda: "",
        grafana_model_dashboard_uid="",
        grafana_pd_dashboard_uid="",
    )
    with pytest.raises(InternalServerErrorException):
        await _redirect(_benchmark(), _model(), cfg=cfg)


@pytest.mark.asyncio
async def test_a_deleted_model_is_refused_rather_than_guessed():
    from gpustack.api.exceptions import BadRequestException

    with pytest.raises(BadRequestException):
        await _redirect(_benchmark(), None)
