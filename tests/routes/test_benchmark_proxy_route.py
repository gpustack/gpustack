"""The benchmark proxy: the deployment's entrance, opened for a load generator.

`route` mode has to enter where client traffic enters — same route resolution,
same load balancer — but the caller is a container this server asked a worker
to start, holding a worker token rather than anyone's account. So the door is
its own prefix: worker-authenticated, and carrying the one exemption the
OpenAI handler honours.

What these pin is the shape of that exemption, because the failure mode is
silent: an exemption reachable from `/v1` would let any authenticated caller
skip the per-user model access list.
"""

from types import SimpleNamespace

import pytest

from gpustack.api.auth import get_worker_principal
from gpustack.routes import routes as routes_module


def _routes_under(prefix: str):
    return [r for r in routes_module.api_router.routes if r.path.startswith(prefix)]


class TestTheDoorExists:
    def test_the_openai_endpoints_are_served_under_the_proxy_prefix(self):
        paths = {r.path for r in _routes_under("/v2/benchmark-proxy")}
        # What a load generator calls: it appends /v1/... to the target itself.
        assert "/v2/benchmark-proxy/v1/chat/completions" in paths
        assert "/v2/benchmark-proxy/v1/completions" in paths

    def test_the_same_handler_serves_both_doors(self):
        # Not a copy: route resolution, weighting and load balancing must not
        # be able to differ between what a benchmark measures and what a
        # client gets.
        proxy = next(
            r
            for r in _routes_under("/v2/benchmark-proxy")
            if r.path.endswith("/v1/chat/completions")
        )
        public = next(
            r
            for r in routes_module.api_router.routes
            if r.path == "/v1/chat/completions"
        )
        assert proxy.endpoint is public.endpoint


class TestTheProbeThatComesFirst:
    """guidellm validates a backend by GETting `{target}/health` before it
    sends anything, and a non-200 kills the run at startup -- measured against
    a live server as `404 Not Found for /v2/benchmark-proxy/health`, with no
    request ever made."""

    def test_the_prefix_answers_health(self):
        assert "/v2/benchmark-proxy/health" in {
            r.path for r in _routes_under("/v2/benchmark-proxy")
        }

    def test_the_probe_is_authenticated_like_everything_else(self):
        # Which is what makes it worth probing: reaching it proves the token
        # works, instead of the run discovering that as 401s mid-ramp.
        health = next(
            r
            for r in _routes_under("/v2/benchmark-proxy")
            if r.path.endswith("/health")
        )
        assert get_worker_principal in {d.call for d in health.dependant.dependencies}

    @pytest.mark.asyncio
    async def test_it_answers_ok(self):
        assert await routes_module.benchmark_proxy_health() == {"status": "ok"}


class TestTheDoorIsGuarded:
    def _dependency_calls(self, route):
        return {d.call for d in route.dependant.dependencies}

    def test_it_takes_a_worker_token(self):
        proxy = next(
            r
            for r in _routes_under("/v2/benchmark-proxy")
            if r.path.endswith("/v1/chat/completions")
        )
        assert get_worker_principal in self._dependency_calls(proxy)

    def test_the_exemption_is_carried_by_this_prefix_only(self):
        proxy = next(
            r
            for r in _routes_under("/v2/benchmark-proxy")
            if r.path.endswith("/v1/chat/completions")
        )
        public = next(
            r
            for r in routes_module.api_router.routes
            if r.path == "/v1/chat/completions"
        )
        assert routes_module.mark_internal_inference in self._dependency_calls(proxy)
        # The load-bearing half: a request that did not come through the proxy
        # never has the flag, so `/v1` keeps its per-user access list.
        assert routes_module.mark_internal_inference not in self._dependency_calls(
            public
        )


class TestTheFlagItself:
    @pytest.mark.asyncio
    async def test_it_marks_the_request(self):
        request = SimpleNamespace(state=SimpleNamespace())
        await routes_module.mark_internal_inference(request)
        assert request.state.internal_inference is True
