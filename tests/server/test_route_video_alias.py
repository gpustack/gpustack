"""The generic-proxy router's ``aliasNameMapping`` entry per route.

Generic proxy addresses a route by id (``/model/proxy/{id}``). A video
route's job follow-ups name the route by id as well, inside the self-routing
id the gateway wrote into the create response, and carry no model of their
own -- without the entry the follow-up resolves no model, matches no route
and is answered by the Server with an id it cannot read. The decision is
re-read off the fresh row: the category sync that turns a route into a video
route lands as a route update the event payload may predate.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.model_routes import ModelRoute
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import sync_gateway


def _route(**changes):
    fields = {
        "id": 7,
        "name": "r",
        "owner_principal_id": 1,
        "generic_proxy": False,
        "categories": [],
        **changes,
    }
    return ModelRoute(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_route, fresh_route, expected",
    [
        (_route(), _route(), False),
        (_route(generic_proxy=True), _route(generic_proxy=True), True),
        (_route(categories=["video"]), _route(categories=["video"]), True),
        # the payload predates the category sync
        (_route(), _route(categories=["llm", "video"]), True),
        (_route(categories=["video"]), _route(), False),
        # the route was deleted between the event and the reconcile
        (_route(categories=["video"]), None, False),
    ],
)
async def test_alias_entry_is_written_for_generic_proxy_or_video_routes(
    event_route, fresh_route, expected
):
    cfg = SimpleNamespace(
        gateway_namespace="higress-system",
        gateway_ingress_class="higress",
        get_namespace=lambda: "gpustack",
    )

    with (
        patch(
            "gpustack.server.controllers.ModelRoute.one_by_id",
            AsyncMock(return_value=fresh_route),
        ),
        patch(
            "gpustack.server.controllers.calculate_destinations",
            AsyncMock(return_value=([], [])),
        ),
        patch(
            "gpustack.server.controllers.Principal.one_by_id",
            AsyncMock(return_value=None),
        ),
        patch(
            "gpustack.server.controllers.effective_route_name",
            return_value="org/r",
        ),
        patch(
            "gpustack.server.controllers.dispatch_route_reconcile", AsyncMock()
        ) as dispatch,
        patch(
            "gpustack.server.controllers.mcp_handler.ensure_model_ingress",
            AsyncMock(),
        ),
        patch(
            "gpustack.server.controllers.ensure_route_generic_proxy_router_config",
            AsyncMock(),
        ) as ensure_alias,
    ):
        await sync_gateway(
            session=None,
            event=Event(type=EventType.UPDATED, data=event_route),
            cfg=cfg,
            model_route=event_route,
            networking_api=None,
            extensions_api=None,
            istio_networking_api=None,
        )

    assert dispatch.await_count == 1
    assert ensure_alias.await_count == 1
    kwargs = ensure_alias.await_args.kwargs
    assert kwargs["alias_enabled"] is expected
    assert kwargs["effective_name"] == "org/r"
    assert kwargs["model_route"].id == 7
