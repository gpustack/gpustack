"""The fallback plugin owns the whole fallback path: the mapper CR's
fallback rules (declared on the collector under the ``mapper`` owner, so
the LB rules on the same CR survive), the fallback ingress, and the
fallback EnvoyFilter."""

from unittest.mock import MagicMock, patch

import pytest

from gpustack.routes.plugins.fallback.plugin import fallback_plugin


class _Registry:
    def __init__(self, name):
        self.name = name

    def get_service_name(self):
        return self.name


class _Cfg:
    gateway_namespace = "higress-system"
    gateway_ingress_class = "higress"
    # the shape lb_module_available() reads; these tests run with the
    # bundled manifest, which resolves the module, so degraded-mode
    # tests monkeypatch lb_module_available directly
    gateway_plugin = {}
    gateway_plugin_server_url = None
    gateway_mode = "external"

    def get_namespace(self):
        return "gpustack"


class _Ctx:
    def __init__(
        self, route, fallback_destinations=None, delete=False, destinations=None
    ):
        self.cfg = _Cfg()
        self.session = None
        self.model_route = route
        self.ingress_name = "ai-route-route-1.internal"
        self.event_is_delete = delete
        self.extensions_api = object()
        self.istio_networking_api = object()
        self.collector = MagicMock()
        self.networking_api = object()
        self.effective_name = "org1/route"
        self.fallback_destinations = fallback_destinations or []
        self.destinations = destinations


def _route(fallback_codes=None):
    from gpustack.schemas.model_routes import (
        ModelRoute,
        ModelRouteTarget,
        TargetStateEnum,
    )

    target = ModelRouteTarget(
        id=1,
        name="t",
        route_id=1,
        route_name="r",
        state=TargetStateEnum.ACTIVE,
        fallback_status_codes=fallback_codes,
    )
    return ModelRoute(id=1, name="r", targets=0, ready_targets=0), target


async def _run(monkeypatch, targets, ctx):
    ingress_calls = []
    filter_calls = []

    async def fake_ingress(**kwargs):
        ingress_calls.append(kwargs)

    async def fake_filter(**kwargs):
        filter_calls.append(kwargs)

    import gpustack.routes.plugins.fallback.plugin as plugin_mod

    monkeypatch.setattr(plugin_mod, "_has_fallback_target", _has_targets(targets))
    with (
        patch(
            "gpustack.gateway.utils.ensure_model_ingress",
            side_effect=fake_ingress,
        ),
        patch(
            "gpustack.gateway.utils.ensure_fallback_filter",
            side_effect=fake_filter,
        ),
    ):
        await fallback_plugin.reconcile_route(ctx)
    return ingress_calls, filter_calls


def _has_targets(targets):
    async def inner(ctx):
        return any(
            t.fallback_status_codes and len(t.fallback_status_codes) > 0
            for t in targets
        )

    return inner


def _declarations(ctx):
    return ctx.collector.set_rules.call_args.kwargs


class TestMapperRuleDeclaration:
    @pytest.mark.asyncio
    async def test_fallback_rules_declared_under_mapper_owner(self, monkeypatch):
        route, target = _route(fallback_codes=["5xx"])
        ctx = _Ctx(
            route,
            fallback_destinations=[(1, "m1", _Registry("svc-a.static"))],
        )
        ingress_calls, _ = await _run(monkeypatch, [target], ctx)

        kwargs = _declarations(ctx)
        assert kwargs["cr_name"] == "gpustack-model-mapper"
        assert kwargs["owner"] == "mapper"
        assert kwargs["ingresses"] == ["gpustack/ai-route-route-1.internal"]
        # The expected fallback rule itself is delegated to
        # get_expected_match_list; what the plugin owns is the declaration.
        assert kwargs["rules"]  # non-empty when fallback destinations exist

    @pytest.mark.asyncio
    async def test_no_fallback_destinations_declares_strip(self, monkeypatch):
        route, target = _route()  # no fallback codes
        ctx = _Ctx(route, fallback_destinations=[])
        await _run(monkeypatch, [target], ctx)

        kwargs = _declarations(ctx)
        assert kwargs["rules"] == []

    @pytest.mark.asyncio
    async def test_degraded_mode_declares_main_path_mapping(self, monkeypatch):
        # With the LB module unavailable, the legacy per-route main-path
        # modelMapping rule is the only rewrite left — the mapper CR
        # must carry it or the upstream sees the route name.
        monkeypatch.setattr(
            "gpustack.routes.plugins.lb.gateway.lb_module_available",
            lambda cfg: False,
        )
        route, target = _route()
        ctx = _Ctx(
            route,
            fallback_destinations=[],
            destinations=[(1, "real-model", _Registry("svc-a.static"))],
        )
        await fallback_plugin._declare_mapper_rules(ctx, ctx.collector)

        kwargs = _declarations(ctx)
        rules = kwargs["rules"]
        assert len(rules) == 1
        assert rules[0].config == {"modelMapping": {"org1/route": "real-model"}}
        assert rules[0].ingress == ["gpustack/ai-route-route-1.internal"]
        assert rules[0].service == ["svc-a.static"]

    @pytest.mark.asyncio
    async def test_degraded_mode_skips_self_mapping(self, monkeypatch):
        # A route whose model already answers to the route name needs no
        # rewrite rule even in degraded mode.
        monkeypatch.setattr(
            "gpustack.routes.plugins.lb.gateway.lb_module_available",
            lambda cfg: False,
        )
        route, target = _route()
        ctx = _Ctx(
            route,
            fallback_destinations=[],
            destinations=[(1, "org1/route", _Registry("svc-a.static"))],
        )
        await fallback_plugin._declare_mapper_rules(ctx, ctx.collector)

        kwargs = _declarations(ctx)
        assert kwargs["rules"] == []

    @pytest.mark.asyncio
    async def test_delete_event_strips_mapper_rules_even_with_fallback(self):
        # The main ingress is being removed with the route; re-declaring
        # its mapper rules would leave them stale on the CR until the
        # startup cleanup pass.
        route, target = _route(fallback_codes=["5xx"])
        ctx = _Ctx(
            route,
            fallback_destinations=[(1, "m1", _Registry("svc-a.static"))],
            delete=True,
        )
        await fallback_plugin._declare_mapper_rules(ctx, ctx.collector)

        kwargs = _declarations(ctx)
        assert kwargs["cr_name"] == "gpustack-model-mapper"
        assert kwargs["owner"] == "mapper"
        assert kwargs["ingresses"] == ["gpustack/ai-route-route-1.internal"]
        assert kwargs["rules"] == []


class TestFallbackIngressAndFilter:
    @pytest.mark.asyncio
    async def test_fallback_target_creates_ingress_and_filter(self, monkeypatch):
        route, target = _route(fallback_codes=["5xx"])
        ctx = _Ctx(route, fallback_destinations=[(1, "m1", _Registry("s.static"))])
        ingress_calls, filter_calls = await _run(monkeypatch, [target], ctx)

        (fallback_ingress,) = ingress_calls
        assert fallback_ingress["ingress_name"] == (
            "ai-route-route-1.fallback.internal"
        )
        assert fallback_ingress["route_name"] == "org1/route"
        assert fallback_ingress["extra_annotations"]
        (filter_call,) = filter_calls
        assert filter_call["ingress_name"] == "ai-route-route-1.internal"

    @pytest.mark.asyncio
    async def test_no_fallback_target_tears_down(self, monkeypatch):
        from gpustack.server.bus import EventType

        route, target = _route()  # no fallback codes
        ctx = _Ctx(route)
        ingress_calls, filter_calls = await _run(monkeypatch, [target], ctx)

        assert ingress_calls[0]["event_type"] == EventType.DELETED
        assert filter_calls[0]["event_type"] == EventType.DELETED

    @pytest.mark.asyncio
    async def test_delete_event_tears_down_even_with_targets(self, monkeypatch):
        from gpustack.server.bus import EventType

        route, target = _route(fallback_codes=["5xx"])
        ctx = _Ctx(route, fallback_destinations=[], delete=True)
        ingress_calls, filter_calls = await _run(monkeypatch, [target], ctx)

        assert ingress_calls[0]["event_type"] == EventType.DELETED
        assert filter_calls[0]["event_type"] == EventType.DELETED
