from gpustack.gateway.client.extensions_higress_io_v1_api import (
    WasmPluginMatchRule,
    WasmPluginSpec,
)
import pytest

from gpustack.routes.plugins.artifacts import apply_rule_updates
from gpustack.routes.plugins.lb.config import LBPolicyConfig
from gpustack.routes.plugins.lb.reconciler import (
    CONTEXT_CR_NAME,
    _build_candidate,
    cluster_header_envoy_filter,
    envoy_filter_name,
    sync_model_route_lb,
)


class TestLBPolicyConfig:
    def test_defaults(self):
        config = LBPolicyConfig.model_validate({})
        assert config.enabled is True

    def test_gateway_rule_config(self):
        config = LBPolicyConfig.model_validate(
            {
                "health": {"fail_open": False, "unhealthy_threshold": 3},
                "reject": {"status": 503, "message": "busy"},
                "max_body_bytes": 4096,
            }
        )
        rule = config.to_gateway_rule()
        # mode is role-level and never rides a per-route rule
        assert "mode" not in rule
        # unset health knobs are omitted, not sent as explicit nulls:
        # an absent key keeps the plugin's compiled-in default
        assert rule["health"] == {"failOpen": False, "unhealthyThreshold": 3}
        assert rule["reject"] == {"status": 503, "message": "busy"}
        assert rule["maxBodyBytes"] == 4096
        # an all-defaults policy contributes only the reject response
        assert LBPolicyConfig.model_validate({}).to_gateway_rule() == {
            "reject": {"status": 503, "message": "no healthy model instance available"}
        }


class TestBuildCandidate:
    def test_weight_omitted_when_unset_or_zero(self):
        # weight presence is the weighted-dice vs capability-scoring
        # switch; 0 means "unset" here, matching the column default.
        for weight in (None, 0):
            candidate = _build_candidate(
                "outbound|80||model-1-1.static", 1, "instance", "m", weight, None
            )
            assert "weight" not in candidate
        assert (
            _build_candidate(
                "outbound|80||model-1-1.static", 1, "instance", "m", 70, None
            )["weight"]
            == 70
        )

    def test_cluster_is_carried_verbatim(self):
        candidate = _build_candidate(
            "outbound|80||model-1-1.static", 2, "instance", "m", None, 8
        )
        assert candidate["cluster"] == "outbound|80||model-1-1.static"
        assert candidate["targetId"] == "2"
        assert candidate["maxRunningRequests"] == 8

    def test_cluster_name_uses_the_registrys_real_port(self):
        from types import SimpleNamespace

        from gpustack.routes.plugins.lb.reconciler import candidate_cluster_name

        # A DNS-typed instance registry (worker hostname, real port) or
        # an https provider registry (443) must carry its port into the
        # cluster name; only a portless registry falls back to 80.
        dns = SimpleNamespace(port=8080, get_service_name=lambda: "worker-a.dns")
        assert candidate_cluster_name(dns) == "outbound|8080||worker-a.dns"
        https = SimpleNamespace(port=443, get_service_name=lambda: "p.dns")
        assert candidate_cluster_name(https) == "outbound|443||p.dns"
        portless = SimpleNamespace(port=None, get_service_name=lambda: "s.static")
        assert candidate_cluster_name(portless) == "outbound|80||s.static"


class TestSyncModelRouteLb:
    """The LB matchRule is declared on the collector, not written
    directly: one declaration for the shared CR (upsert when candidates
    render, strip otherwise), the EnvoyFilter written directly."""

    def _run(self, monkeypatch, rendered, meta=None, delete=False):
        import asyncio

        from gpustack.routes.plugins.artifacts import RouteArtifactCollector
        from gpustack.routes.plugins.lb import reconciler as rec
        from gpustack.schemas.model_routes import ModelRoute

        class _Cfg:
            gateway_namespace = "ns"
            gateway_plugin = {}
            gateway_plugin_server_url = None
            # non-embedded keeps use_local_plugin_modules() off the
            # filesystem, so plugin_spec_overrides resolves purely from
            # the (empty) operator overrides.
            gateway_mode = "external"

            def get_namespace(self):
                return "ns"

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0, meta=meta)
        envoy_filter_present = []

        async def fake_one_by_id(cls, session, id, **kw):
            return route

        async def fake_render(session, route):
            return ([{"cluster": "c"}], {"10": {"*": "m"}}) if rendered else None

        async def fake_filter(namespace, ingress_name, istio_networking_api, present):
            envoy_filter_present.append(present)

        monkeypatch.setattr(rec.ModelRoute, "one_by_id", classmethod(fake_one_by_id))
        monkeypatch.setattr(rec, "render_route", fake_render)
        monkeypatch.setattr(rec, "_ensure_envoy_filter", fake_filter)

        collector = RouteArtifactCollector()
        asyncio.run(
            sync_model_route_lb(
                cfg=_Cfg(),
                session=None,
                collector=collector,
                istio_networking_api=object(),
                model_route=route,
                ingress_name="ai-route-route-1.internal",
                event_is_delete=delete,
            )
        )
        return collector, envoy_filter_present

    def _updates(self, collector):
        # the declarations, exactly as flush would receive them
        return collector._updates[CONTEXT_CR_NAME]

    def test_rendered_route_declares_upsert(self, monkeypatch):
        collector, filters = self._run(monkeypatch, rendered=True)
        updates = self._updates(collector)
        assert len(updates) == 1
        assert updates[0].owner == "lb"
        assert updates[0].ingresses == ["ai-route-route-1.internal"]
        assert updates[0].rules[0].config["candidates"] == [{"cluster": "c"}]
        assert updates[0].create_base is None  # missing CR is not recreated
        assert filters == [True]

    def test_policy_knobs_ride_the_rendered_rule(self, monkeypatch):
        # health/reject/maxBodyBytes reach the gateway through the same
        # matchRule as the candidates — a Higress rule config overrides
        # the CR defaultConfig, so the knobs take effect per route.
        collector, _ = self._run(
            monkeypatch,
            rendered=True,
            meta={
                "lb": {
                    "health": {"unhealthy_threshold": 3},
                    "reject": {"status": 503, "message": "busy"},
                }
            },
        )
        config = self._updates(collector)[0].rules[0].config
        assert config["candidates"] == [{"cluster": "c"}]
        assert config["health"] == {"unhealthyThreshold": 3}
        assert config["reject"] == {"status": 503, "message": "busy"}

    def test_unusable_route_declares_strip(self, monkeypatch):
        collector, filters = self._run(monkeypatch, rendered=False)
        assert self._updates(collector)[0].rules == []
        assert filters == [False]

    def test_delete_event_declares_strip(self, monkeypatch):
        collector, filters = self._run(monkeypatch, rendered=True, delete=True)
        assert self._updates(collector)[0].rules == []
        assert filters == [False]

    def test_disabled_policy_declares_strip(self, monkeypatch):
        collector, _ = self._run(
            monkeypatch, rendered=True, meta={"lb": {"enabled": False}}
        )
        assert self._updates(collector)[0].rules == []

    def test_declaration_preserves_mapper_rules(self, monkeypatch):
        # The mapper sync writes dual-attached fallback modelMapping
        # rules onto the same CR — they list the main ingress as well.
        # Folding the LB declaration must never recycle them, or the
        # two writers keep deleting each other's rules.
        collector, _ = self._run(monkeypatch, rendered=True)
        live = WasmPluginSpec(
            matchRules=[
                WasmPluginMatchRule(
                    config={"candidates": ["stale"]},
                    ingress=["ai-route-route-1.internal"],
                ),
                WasmPluginMatchRule(
                    config={"modelMapping": {"a": "c"}},
                    ingress=["ai-route-route-1.internal", "ns/x.fallback"],
                    service=["svc-c"],
                ),
            ]
        )
        merged = apply_rule_updates(live, self._updates(collector))
        configs = [r.config for r in merged.matchRules]
        assert {"modelMapping": {"a": "c"}} in configs
        assert any("candidates" in c for c in configs)
        assert {"candidates": ["stale"]} not in configs


class TestEnvoyFilter:
    def test_names_and_shape(self):
        assert envoy_filter_name("ai-route-route-2.internal") == (
            "gpustack-lb-ai-route-route-2.internal"
        )
        body = cluster_header_envoy_filter("n", "ns", "ai-route-route-2.internal")
        patch = body.spec.configPatches[0]
        route_match = patch.match.routeConfiguration.vhost.route
        assert route_match.name == "ai-route-route-2.internal"
        assert patch.patch.value == {
            "route": {"cluster_header": "x-higress-target-cluster"}
        }


class TestGatewayEntriesDegrade:
    def test_unknown_module_degrades_to_plain_mapper(self, monkeypatch):
        # A manifest that does not know gpustack-lb (dependency older
        # than the version packaging it) must not crash server startup,
        # and the plain model-mapper CR must still be published so
        # model-name rewrite keeps working on non-LB routes.
        from gpustack.routes.plugins.lb.gateway import (
            LB_CONTEXT_CR_NAME,
            lb_gateway_entries,
        )
        import gpustack.gateway.plugins as plugins_module

        monkeypatch.setattr(plugins_module, "supported_plugins", [])
        assert lb_gateway_entries(None) == []

        mapper_only = [
            plugins_module.HigressPlugin(name="gpustack-model-mapper", version="1.0.0")
        ]
        monkeypatch.setattr(plugins_module, "supported_plugins", mapper_only)
        entries = lb_gateway_entries(None)
        assert [e.name for e in entries] == [LB_CONTEXT_CR_NAME]
        assert entries[0].create_only is True
        assert entries[0].spec.defaultConfig == {"modelMapping": {}}


class TestRedisFromUrl:
    def test_ip_host_builds_static_registry_and_config(self):
        from gpustack.routes.plugins.lb.gateway import (
            _redis_config_from_url,
            redis_registry_from_url,
        )

        # An IP host becomes a static registry (address carried in the
        # domain as host:port) and the plugin config points at the
        # registry's service name, never the raw host.
        registry = redis_registry_from_url("redis://192.168.32.199:30379/2")
        assert registry.name == "gpustack-redis"
        assert registry.type == "static"
        assert registry.domain == "192.168.32.199:30379"
        # The static cluster listens on 80 whatever the backend port is;
        # 6379 here would name a cluster Envoy never creates.
        assert registry.port == 80
        assert _redis_config_from_url("redis://192.168.32.199:30379/2") == {
            "service_name": "gpustack-redis.static",
            "service_port": 80,
            "database": 2,
        }

    def test_ip_host_without_port_defaults_domain_to_6379(self):
        from gpustack.routes.plugins.lb.gateway import redis_registry_from_url

        # A static registry encodes the real backend port in the
        # domain; an omitted port is Redis's own default, never 80.
        registry = redis_registry_from_url("redis://10.0.0.1")
        assert registry.domain == "10.0.0.1:6379"
        assert registry.port == 80

    def test_credentialed_url_is_refused(self):
        from gpustack.routes.plugins.lb.gateway import _redis_config_from_url

        # Credentials would be materialized verbatim in the WasmPlugin
        # CR spec; until a secret-backed mechanism exists they are
        # refused rather than leaked, and the deployment stays on
        # per-process shared data.
        assert _redis_config_from_url("redis://u:p@192.168.32.199:30379/2") is None
        assert _redis_config_from_url("redis://:p@192.168.32.199:30379/2") is None
        assert _redis_config_from_url("redis://u@192.168.32.199:30379/2") is None

    def test_malformed_port_is_refused_not_degraded(self):
        from gpustack.routes.plugins.lb.gateway import redis_registry_from_url

        # urlparse raises ValueError on a non-numeric or out-of-range
        # port; the refusal must read as an unusable redis_url, not
        # escape as the misleading "plugin module missing" degrade.
        assert redis_registry_from_url("redis://192.168.32.199:notaport/") is None
        assert redis_registry_from_url("redis://192.168.32.199:99999/") is None

    def test_ipv6_host_is_bracketed_in_the_static_domain(self):
        from gpustack.routes.plugins.lb.gateway import redis_registry_from_url

        # urlparse strips the brackets off an IPv6 literal; the
        # host:port domain must carry them again or Envoy reads the
        # colons as the port separator.
        registry = redis_registry_from_url("redis://[2001:db8::1]:6379")
        assert registry.type == "static"
        assert registry.domain == "[2001:db8::1]:6379"
        assert redis_registry_from_url("redis://[2001:db8::1]").domain == (
            "[2001:db8::1]:6379"
        )

    def test_bare_service_name_is_qualified_for_dns(self):
        from gpustack.routes.plugins.lb.gateway import (
            _redis_config_from_url,
            redis_registry_from_url,
        )

        # A single-label host cannot resolve under Envoy STRICT_DNS, so
        # it is qualified to <host>.<namespace>.svc.
        registry = redis_registry_from_url(
            "redis://gpustack-redis/0", namespace="higress-system"
        )
        assert registry.type == "dns"
        assert registry.domain == "gpustack-redis.higress-system.svc"
        assert registry.port == 6379
        assert (
            _redis_config_from_url(
                "redis://gpustack-redis/0", namespace="higress-system"
            )["service_name"]
            == "gpustack-redis.dns"
        )

    def test_rediss_and_garbage_are_refused(self):
        # The plugin's redis block has no TLS knob and no way to report
        # a bad host at config time; both stay on shared data rather
        # than half-configuring the CRs.
        from gpustack.routes.plugins.lb.gateway import (
            _redis_config_from_url,
            redis_registry_from_url,
        )

        assert _redis_config_from_url("rediss://redis.example") is None
        assert redis_registry_from_url("not-a-url") is None


class TestRedisOnBothCRs:
    def _entries(self, monkeypatch, redis_url):
        import gpustack.routes.plugins.lb.gateway as lb_gateway
        from gpustack.routes.plugins.lb.gateway import lb_gateway_entries

        # Bypass module resolution: these tests pin the defaultConfig
        # shape, not where the wasm image comes from.
        monkeypatch.setattr(
            lb_gateway,
            "plugin_spec_overrides",
            lambda name, version=None, cfg=None: {"url": "file://wasm"},
        )
        cfg = type(
            "Cfg", (), {"redis_url": redis_url, "gateway_namespace": "higress-system"}
        )
        return {e.name: e.spec.defaultConfig for e in lb_gateway_entries(cfg)}

    def test_redis_url_lands_on_both_default_configs(self, monkeypatch):
        by_name = self._entries(monkeypatch, "redis://192.168.32.199:30379/0")
        # The one knob that must never appear on a single role only:
        # one-sided redis fails silently in the data plane. The service
        # name is the McpBridge registry's, never the raw host.
        assert by_name["gpustack-model-mapper"]["redis"] == {
            "service_name": "gpustack-redis.static",
            "service_port": 80,
            "database": 0,
        }
        assert (
            by_name["gpustack-lb"]["redis"] == by_name["gpustack-model-mapper"]["redis"]
        )

    def test_no_redis_url_means_no_block(self, monkeypatch):
        by_name = self._entries(monkeypatch, None)
        for default_config in by_name.values():
            assert "redis" not in (default_config or {})


class TestHooks:
    def test_route_section_lands_in_meta(self):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import lb_plugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0, meta={"x": 1})
        asyncio.run(
            lb_plugin.on_route_write("update", route, {"enabled": True}, session=None)
        )
        assert route.meta["x"] == 1  # other meta keys are preserved
        assert route.meta["lb"]["enabled"] is True


class TestLbMode:
    """The derived lb mode is the explicit answer to "what selection
    behaviour do this route's targets add up to right now" — a client
    never infers it from weight conventions. The derivation runs on
    the reconcile path and is persisted to ``meta["lb_mode"]``; read
    paths serve the stored copy."""

    def _mode(self, weights, capability_on=False, active=True, fallback_idx=()):
        import asyncio

        import gpustack.routes.plugins as plugins_pkg
        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import (
            ModelRoute,
            ModelRouteTarget,
            TargetStateEnum,
        )

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0)
        targets = [
            ModelRouteTarget(
                id=10 + i,
                name=f"t{i}",
                route_name="r",
                route_id=1,
                weight=w,
                state=(
                    TargetStateEnum.ACTIVE if active else TargetStateEnum.UNAVAILABLE
                ),
                fallback_status_codes=(["4xx"] if i in fallback_idx else None),
            )
            for i, w in enumerate(weights)
        ]

        original_all = ModelRouteTarget.all_by_fields
        original_plugins = plugins_pkg.route_plugins

        async def fake_all(cls, session, fields, extra_conditions=None, **kw):
            return targets

        def fake_plugins():
            class Cap:
                name = "cap"
                RouteExtension = dict

                async def is_effective_on(self, route, session):
                    return capability_on

            class Lb:
                name = "lb"
                RouteExtension = dict

            return [Lb(), Cap()]

        ModelRouteTarget.all_by_fields = classmethod(fake_all)
        plugins_pkg.route_plugins = fake_plugins
        try:
            mode = asyncio.run(LBPlugin()._derive_lb_mode(route, session=None))
        finally:
            ModelRouteTarget.all_by_fields = original_all
            plugins_pkg.route_plugins = original_plugins
        return mode

    def test_no_section_when_plain_rr(self):
        # no weights and no capability policies: nothing to describe,
        # even though LB still routes the request round-robin
        assert self._mode([0, 0]) is None

    def test_scoring_when_capability_configured(self):
        assert self._mode([0, 0], capability_on=True) == "scoring"

    def test_weighted(self):
        assert self._mode([70, 30]) == "weighted"

    def test_invalid_mixed(self):
        assert self._mode([70, 0]) == "invalid"

    def test_probe_failure_propagates(self):
        # A mode derived from "the capability plugin happened to fail"
        # would misdescribe live gateway behaviour (the wasm rule stays
        # on the CR regardless) — the derivation aborts and the reconcile
        # retries instead of recording a wrong mode.
        import asyncio

        import gpustack.routes.plugins as plugins_pkg
        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0)

        class Failing:
            name = "cap"
            RouteExtension = dict

            async def is_effective_on(self, route, session):
                raise RuntimeError("storage hiccup")

        async def fake_all(cls, session, fields, extra_conditions=None, **kw):
            # one active zero-weight target: the derivation must reach the
            # capability probe instead of answering early
            return [
                ModelRouteTarget(
                    id=10,
                    name="t0",
                    route_name="r",
                    route_id=1,
                    weight=0,
                    state="active",
                )
            ]

        original_plugins = plugins_pkg.route_plugins
        original_all = ModelRouteTarget.all_by_fields
        ModelRouteTarget.all_by_fields = classmethod(fake_all)
        plugins_pkg.route_plugins = lambda: [Failing()]
        try:
            with pytest.raises(RuntimeError):
                asyncio.run(LBPlugin()._derive_lb_mode(route, session=None))
        finally:
            plugins_pkg.route_plugins = original_plugins
            ModelRouteTarget.all_by_fields = original_all

    def test_fallback_target_excluded_from_mode(self):
        # a weighted route plus a weight-0 fallback target is still
        # weighted: fallback targets are not candidates, so their weight
        # column never enters the mixed-weights verdict
        assert self._mode([70, 30, 0], fallback_idx=(2,)) == "weighted"
        # only-fallback routes have no candidates to describe
        assert self._mode([0], fallback_idx=(0,)) is None

    def test_unavailable_targets_keep_the_configured_mode(self):
        # The mode describes the configuration's shape, not the live
        # render: an all-weighted route stays weighted while its
        # instances are down (it renders nothing, but it is still
        # configured as weighted), and instance-health flaps never
        # rewrite the classification.
        assert self._mode([70], active=False) == "weighted"
        assert self._mode([0], capability_on=True, active=False) == "scoring"

    def test_state_transitions_never_flip_the_verdict(self):
        # [70, 0] is mixed whether or not the zero-weight target is
        # serving — the invalid verdict is a property of the weights,
        # not of which instances happen to be up.
        assert self._mode([70, 0]) == "invalid"
        assert self._mode([70, 0], active=False) == "invalid"


class TestRefreshLbMode:
    """refresh_lb_mode is the only writer of ``meta["lb_mode"]``: it
    persists the derived value through ``update()`` (so the UPDATED
    event reaches the watch stream), removes the key when there is
    nothing to derive, and skips the write entirely when nothing
    moved — which is also what terminates the reconcile round the
    update event costs."""

    def _patch_reload(self, monkeypatch, route):
        # refresh_lb_mode re-loads the row: the event-carried instance is
        # detached and mutating it raises ObjectDereferencedError, so the
        # write must go through the session-loaded copy.
        from gpustack.schemas.model_routes import ModelRoute

        async def fake_one_by_id(cls, session, id):
            return route

        monkeypatch.setattr(ModelRoute, "one_by_id", classmethod(fake_one_by_id))

    def _patch_update(self, monkeypatch, updates):
        from gpustack.schemas.model_routes import ModelRoute

        async def fake_update(self, session, source, auto_commit=True):
            updates.append(source)

        monkeypatch.setattr(ModelRoute, "update", fake_update)

    def test_persists_derived_mode(self, monkeypatch):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0)
        updates = []

        async def derive(self, route, session):
            return "weighted"

        monkeypatch.setattr(LBPlugin, "_derive_lb_mode", derive)
        self._patch_update(monkeypatch, updates)
        self._patch_reload(monkeypatch, route)
        asyncio.run(LBPlugin().refresh_lb_mode(route, session=None))
        assert updates == [{"meta": {"lb_mode": "weighted"}}]

    def test_stale_key_is_removed_when_no_mode(self, monkeypatch):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(
            id=1, name="r", targets=0, ready_targets=0, meta={"lb_mode": "weighted"}
        )
        updates = []

        async def derive(self, route, session):
            return None

        monkeypatch.setattr(LBPlugin, "_derive_lb_mode", derive)
        self._patch_update(monkeypatch, updates)
        self._patch_reload(monkeypatch, route)
        asyncio.run(LBPlugin().refresh_lb_mode(route, session=None))
        # a mode-less route carries no key at all, and other keys ride along
        assert updates == [{"meta": {}}]

    def test_no_write_when_unchanged(self, monkeypatch):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(
            id=1,
            name="r",
            targets=0,
            ready_targets=0,
            meta={"lb_mode": "scoring", "x": 1},
        )
        updates = []

        async def derive(self, route, session):
            return "scoring"

        monkeypatch.setattr(LBPlugin, "_derive_lb_mode", derive)
        self._patch_update(monkeypatch, updates)
        self._patch_reload(monkeypatch, route)
        asyncio.run(LBPlugin().refresh_lb_mode(route, session=None))
        assert updates == []


class TestEnrichFromMeta:
    def test_section_reads_stored_mode(self):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(
            id=1,
            name="r",
            targets=0,
            ready_targets=0,
            meta={"lb_mode": "invalid", "lb": {"enabled": False}},
        )
        sections = asyncio.run(LBPlugin().enrich_routes([route], session=None))
        assert sections == {1: {"mode": "invalid", "config": {"enabled": False}}}

    def test_no_section_without_stored_mode(self):
        import asyncio

        from gpustack.routes.plugins.lb.plugin import LBPlugin
        from gpustack.schemas.model_routes import ModelRoute

        route = ModelRoute(id=1, name="r", targets=0, ready_targets=0)
        assert asyncio.run(LBPlugin().enrich_routes([route], session=None)) == {}
