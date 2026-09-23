"""Which Grafana dashboard a model's monitoring link lands on.

A disaggregated group and a plain deployment are not the same subject. The
model dashboard is not merely less specific for a group -- its request counters
double, because every request under PD traverses both roles -- so sending a
group there hands the reader a number that is 2x reality with nothing saying
so.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import InternalServerErrorException
from gpustack.routes import models as route
from gpustack.schemas.models import DisaggregationSpec, Model, PDModeEnum


def _model(**kwargs) -> Model:
    return Model(
        name=kwargs.pop("name", "m"),
        cluster_id=kwargs.pop("cluster_id", 1),
        **kwargs,
    )


def _cfg(**overrides):
    base = {
        "grafana_model_dashboard_uid": "gpustack-model",
        "grafana_pd_dashboard_uid": "gpustack-pd",
        "get_grafana_url": lambda: "http://grafana:3000",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


async def _redirect(model, cfg=None, mode=None):
    with (
        patch.object(route, "_get_model", AsyncMock(return_value=model)),
        patch.object(route, "get_global_config", lambda: cfg or _cfg()),
        patch.object(
            route.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(name="x1")),
        ),
        patch.object(
            route, "resolve_grafana_base_url", lambda *_: "http://grafana:3000"
        ),
        # Patched at its source, not on a route module: the dashboard link is
        # built in `utils.grafana` now, shared with the benchmark report, and
        # it reads the catalog through a deferred import.
        patch("gpustack.server.pd_mode_catalog.get_pd_mode", lambda _: mode),
    ):
        response = await route.get_model_dashboard(
            session=None, ctx=None, id=1, request=None
        )
    return response.headers["location"]


@pytest.mark.asyncio
async def test_a_plain_model_goes_to_the_model_dashboard():
    location = await _redirect(_model())
    assert "/d/gpustack-model/gpustack-model" in location
    assert "var-model_name=m" in location
    assert "var-cluster_name=x1" in location
    # No role variable: the model dashboard has none, and a stray one would
    # ride into a saved URL that means nothing there.
    assert "counted_role" not in location


@pytest.mark.asyncio
async def test_a_disaggregated_group_goes_to_the_pd_dashboard():
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    location = await _redirect(model)
    assert "/d/gpustack-pd/gpustack-pd" in location
    assert "var-model_name=m" in location
    assert "var-cluster_name=x1" in location


@pytest.mark.asyncio
async def test_the_counted_role_follows_the_connector():
    """The dashboard defaults to `decode`, so a pushing connector has to say so
    or its transfer panels read a flat zero."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.SGLANG_MOONCAKE))
    pushes = SimpleNamespace(transfer_metrics=SimpleNamespace(read_from_role="prefill"))
    assert "var-counted_role=prefill" in await _redirect(model, mode=pushes)

    pulls = SimpleNamespace(transfer_metrics=SimpleNamespace(read_from_role="decode"))
    assert "var-counted_role=decode" in await _redirect(model, mode=pulls)


@pytest.mark.asyncio
async def test_an_unknown_mode_falls_back_to_decode():
    """A mode the catalog does not carry still gets a working link. Guessing
    wrong shows an empty panel, which beats no dashboard at all."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    assert "var-counted_role=decode" in await _redirect(model, mode=None)


@pytest.mark.asyncio
async def test_a_group_is_refused_when_only_the_pd_uid_is_missing():
    """The uid that matters is the one this model will use, not the model
    dashboard's -- a deployment that configured only the model uid must not
    redirect a group to a dashboard that does not exist."""
    model = _model(disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL))
    with pytest.raises(InternalServerErrorException):
        await _redirect(model, cfg=_cfg(grafana_pd_dashboard_uid=None))

    # The same deployment still serves a plain model.
    location = await _redirect(_model(), cfg=_cfg(grafana_pd_dashboard_uid=None))
    assert "/d/gpustack-model/" in location
