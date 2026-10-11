"""The egress gate on caller-supplied provider base URLs.

/get-models, /test-model and /test-decision-model all dial a base URL the
request body names. ``GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST`` opts a
deployment into refusing loopback / link-local / private targets that the
allowlist does not cover, before any connection is opened, so an org owner
cannot probe the server's own network through them.
"""

import socket
from typing import Callable, Dict, List

import httpx
import pytest

from gpustack import envs
from gpustack.api.exceptions import InvalidException
from gpustack.routes import model_provider as route_module
from gpustack.utils import network as network_module
from gpustack.schemas.model_provider import (
    ModelProviderTypeEnum,
    OpenAIConfig,
    TestProviderModelInput,
    TypesafeConfig,
    TestDecisionModelInput,
)


def _stub_upstream(monkeypatch, respond: Callable[[httpx.Request], httpx.Response]):
    """Answer whichever provider endpoint dialed, recording the base URLs."""
    asked: List[Dict] = []
    real_client = httpx.AsyncClient

    def factory(*, base_url, **_kwargs):
        def record(request: httpx.Request) -> httpx.Response:
            asked.append({"base": str(base_url), "path": request.url.path})
            return respond(request)

        return real_client(transport=httpx.MockTransport(record), base_url=base_url)

    monkeypatch.setattr(route_module.httpx, "AsyncClient", factory)
    return asked


def _gate(monkeypatch, *, allowlist: List[str]):
    monkeypatch.setattr(envs, "PROVIDER_TEST_EGRESS_ALLOWLIST", allowlist, raising=True)


def _openai(endpoint) -> TestProviderModelInput:
    return TestProviderModelInput(
        api_token="sk-test",
        config=OpenAIConfig(
            type=ModelProviderTypeEnum.OPENAI,
            openaiCustomUrl=endpoint,
        ),
        model_name="gpt-test",
    )


def _decision(endpoint) -> TestDecisionModelInput:
    return TestDecisionModelInput(
        api_token="sk-test",
        config=TypesafeConfig.model_validate(
            {
                "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
                "endpoint": endpoint,
            }
        ),
    )


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": []})


class MockGetaddrinfo:
    """Record queried hosts and answer with fixed literal addresses.

    An empty address list raises gaierror, standing in for a name this
    server cannot resolve.
    """

    def __init__(self, addresses):
        self.addresses = addresses
        self.hosts = []

    def __call__(self, host, _port):
        self.hosts.append(host)
        if not self.addresses:
            raise socket.gaierror("unresolvable")
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, None, None, "", (a, 0))
            for a in self.addresses
        ]


def _verdict_ok() -> httpx.Response:
    # The decision route accepts only a well-shaped verdict, unlike the
    # chat-model route which takes any 2xx JSON.
    return httpx.Response(
        200,
        json={
            "answers": {
                "model_selection": {
                    "type": "choice",
                    "choice": "candidate-a",
                    "confidence": 0.9,
                    "probabilities": {"candidate-a": 0.9, "candidate-b": 0.1},
                }
            }
        },
    )


class TestProviderEgressGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "make_input",
        [
            lambda endpoint: _openai(endpoint),
            lambda endpoint: _decision(endpoint),
        ],
        ids=["test-model", "test-decision-model"],
    )
    async def test_an_uncovered_private_target_is_refused_without_a_request(
        self, monkeypatch, make_input
    ):
        # A literal loopback IP is never covered by a name entry, and no CIDR
        # here covers it either.
        _gate(monkeypatch, allowlist=["vllm.internal.corp"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                make_input("http://127.0.0.1:8080")
            )

        assert asked == []  # the refusal happens before any connection
        assert "EGRESS_ALLOWLIST" in raised.value.message

    @pytest.mark.asyncio
    async def test_a_listed_name_goes_through_when_gated(self, monkeypatch):
        _gate(monkeypatch, allowlist=["public.example.com"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        result = await route_module.try_model_with_provider(
            _openai("http://public.example.com:8080")
        )

        assert result.accessible is True
        assert asked[0]["base"] == "http://public.example.com:8080"

    @pytest.mark.asyncio
    async def test_an_unlisted_public_target_is_refused_when_gated(self, monkeypatch):
        _gate(monkeypatch, allowlist=["10.0.0.0/8"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(_openai("http://8.8.8.8:8080"))

        assert asked == []
        assert "EGRESS_ALLOWLIST" in raised.value.message

    @pytest.mark.asyncio
    async def test_any_target_is_allowed_when_the_allowlist_is_empty(self, monkeypatch):
        _gate(monkeypatch, allowlist=[])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        result = await route_module.try_model_with_provider(
            _openai("http://127.0.0.1:8080")
        )

        assert result.accessible is True
        assert asked[0]["base"] == "http://127.0.0.1:8080"

    @pytest.mark.asyncio
    async def test_a_non_http_scheme_is_refused_when_gated(self, monkeypatch):
        _gate(monkeypatch, allowlist=["10.0.0.0/8"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(_openai("file:///etc/passwd"))

        assert asked == []
        assert "not a URL of an allowed scheme" in raised.value.message

    @pytest.mark.asyncio
    async def test_an_unlisted_proxy_is_refused_even_for_a_listed_target(
        self, monkeypatch
    ):
        _gate(monkeypatch, allowlist=["api.example.com"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                TestProviderModelInput(
                    api_token="sk-test",
                    config=OpenAIConfig(
                        type=ModelProviderTypeEnum.OPENAI,
                        openaiCustomUrl="http://api.example.com:8080",
                    ),
                    model_name="gpt-test",
                    proxy_url="http://127.0.0.1:7890",
                )
            )

        assert asked == []
        assert "proxy URL" in raised.value.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "runner",
        [
            # an OpenAI config without a custom url falls back to the hosted
            # default api.openai.com
            lambda: route_module.try_model_with_provider(_openai(None)),
            lambda: route_module.try_decision_model_with_provider(_decision(None)),
        ],
        ids=["test-model", "test-decision-model"],
    )
    async def test_a_hosted_default_endpoint_is_exempt_from_the_gate(
        self, monkeypatch, runner
    ):
        # The hosted defaults are constants this server ships, not values the
        # caller chose, so they must keep working out of the box once the
        # allowlist is on and does not name them.
        _gate(monkeypatch, allowlist=["10.0.0.0/8"])
        asked = _stub_upstream(monkeypatch, lambda _request: _verdict_ok())

        result = await runner()

        assert result.accessible is True
        assert len(asked) == 1

    @pytest.mark.asyncio
    async def test_the_error_does_not_echo_url_userinfo(self, monkeypatch):
        # A URL may embed credentials in its userinfo; the rejection message
        # names the hostname, never the URL.
        _gate(monkeypatch, allowlist=["api.example.com"])
        _stub_upstream(monkeypatch, lambda _request: _ok())

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                _openai("http://user:secret@127.0.0.1:8080")
            )

        assert "secret" not in raised.value.message
        assert "user:" not in raised.value.message
        assert "127.0.0.1" in raised.value.message

    @pytest.mark.asyncio
    async def test_an_unresolvable_host_is_refused(self, monkeypatch):
        # With a proxy the request may still be forwardable (the proxy
        # resolves the target), so a host this server cannot resolve must
        # not be authorized.
        _gate(monkeypatch, allowlist=["api.example.com"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())
        monkeypatch.setattr(network_module.socket, "getaddrinfo", MockGetaddrinfo([]))

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                _openai("http://unresolvable.example:8080")
            )

        assert asked == []
        assert "could not be resolved" in raised.value.message

    @pytest.mark.asyncio
    async def test_a_unicode_host_is_validated_under_its_idna_form(self, monkeypatch):
        # httpx dials faß.example as xn--fa-hia.example, while getaddrinfo
        # alone would look up fass.example; the gate must verify the name
        # the client actually dials.
        _gate(monkeypatch, allowlist=["api.example.com"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())
        resolver = MockGetaddrinfo(["9.9.9.9"])
        monkeypatch.setattr(network_module.socket, "getaddrinfo", resolver)

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                _openai("http://faß.example:8080")
            )

        assert asked == []
        assert resolver.hosts == ["xn--fa-hia.example"]
        assert "9.9.9.9" in raised.value.message


class TestEgressAllowlistMatching:
    """The matching rules of egress_disallowed_address on literal inputs.

    Only cases that decide before or without a DNS lookup are covered: the
    name-matching paths return before any lookup, and literal IPs resolve
    without hitting the network.
    """

    @pytest.mark.parametrize(
        "host,allowlist,disallowed",
        [
            # CIDR coverage: a private address inside the range is allowed
            ("10.0.1.5", ["10.0.0.0/8"], None),
            ("10.0.1.5", ["10.1.0.0/16"], "10.0.1.5"),
            ("192.168.1.4", ["10.0.0.0/8", "192.168.0.0/16"], None),
            # a bare IP entry covers exactly that address
            ("127.0.0.1", ["127.0.0.1"], None),
            ("127.0.0.1", ["10.0.0.0/8"], "127.0.0.1"),
            # an unlisted address is refused, public or not -- the
            # allowlist is the whole rule
            ("8.8.8.8", ["10.0.0.0/8"], "8.8.8.8"),
            ("8.8.8.8", ["8.8.8.8"], None),
            # names match the exact host and any dot-boundary subdomain
            ("vllm.internal.corp", ["vllm.internal.corp"], None),
            ("a.vllm.internal.corp", ["internal.corp"], None),
            # a suffix match must not cross a dot boundary: the host matches
            # neither name, so it is reported as disallowed
            ("evil.internal.corp", ["vil.internal.corp"], "evil.internal.corp"),
            ("evil.internal.corp", ["internal.com"], "evil.internal.corp"),
            # an empty allowlist leaves everything unrestricted
            ("127.0.0.1", [], None),
        ],
    )
    def test_matching(self, monkeypatch, host, allowlist, disallowed):
        # Hostnames must not hit DNS: a name that matches no list entry
        # resolves to itself, so a non-matching name is reported as itself.
        monkeypatch.setattr(
            network_module.socket,
            "getaddrinfo",
            lambda h, _p: [(socket.AF_INET, None, None, "", (h, 0))],
        )
        assert network_module.egress_disallowed_address(host, allowlist) == disallowed


def _resolve(monkeypatch, *addresses):
    """Pin getaddrinfo to literal addresses, so no DNS is hit."""
    monkeypatch.setattr(
        network_module.socket,
        "getaddrinfo",
        lambda _host, _port: [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, None, None, "", (a, 0))
            for a in addresses
        ],
    )


class TestEgressResolutionHandling:
    """The resolution loop must survive dual-stack answers and scoped IPv6."""

    def test_a_dual_stack_name_does_not_raise_across_ip_versions(self, monkeypatch):
        # Membership across IP versions raises TypeError, which used to turn
        # the gate into a 500 on any dual-stack name.
        _resolve(monkeypatch, "127.0.0.1", "::1")

        assert (
            network_module.egress_disallowed_address("dual.stack", ["10.0.0.0/8"])
            == "127.0.0.1"
        )

    def test_a_scoped_ipv6_address_is_checked_after_stripping_the_scope(
        self, monkeypatch
    ):
        _resolve(monkeypatch, "fe80::1%eth0")

        assert (
            network_module.egress_disallowed_address("scoped.host", ["10.0.0.0/8"])
            == "fe80::1"
        )

    def test_an_ipv6_name_covered_by_an_ipv6_cidr_is_allowed(self, monkeypatch):
        _resolve(monkeypatch, "2001:db8:1::5")

        assert (
            network_module.egress_disallowed_address("v6.host", ["2001:db8::/32"])
            is None
        )

    def test_a_resolver_error_fails_closed(self, monkeypatch):
        # getaddrinfo raises more than gaierror: an IDN hostname raises
        # UnicodeError before any query, and resolver failures raise OSError.
        # Either way the host cannot be verified, and with a proxy in play the
        # request may still be forwardable, so it fails closed rather than
        # authorizing an unverifiable target.
        for raised in (UnicodeError("idn"), OSError("resolver")):
            monkeypatch.setattr(
                network_module.socket,
                "getaddrinfo",
                lambda _h, _p, _r=raised: (_ for _ in ()).throw(_r),
            )
            assert (
                network_module.egress_disallowed_address("some.host", ["10.0.0.0/8"])
                == network_module.EGRESS_HOST_UNRESOLVED
            )


class TestHostedDefaultExemption:
    """Only the exact default origin is exempt; anything else stays gated."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://api.openai.com", True),
            ("https://api.openai.com:443", True),
            ("https://api.openai.com/", True),
            # same host, different port or scheme: caller-chosen, not default
            ("https://api.openai.com:8080", False),
            ("http://api.openai.com", False),
            ("https://evil.com", False),
        ],
    )
    def test_match_is_on_the_full_origin(self, url, expected):
        config = OpenAIConfig(type=ModelProviderTypeEnum.OPENAI, openaiCustomUrl=url)

        assert route_module._is_provider_default_endpoint(config, url) is expected


class TestIpv6LiteralGate:
    """IPv6 literal hosts must pass through the gate, not crash it."""

    @pytest.mark.asyncio
    async def test_a_listed_ipv6_target_goes_through(self, monkeypatch):
        _gate(monkeypatch, allowlist=["2001:db8::/32"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())
        resolver = MockGetaddrinfo(["2001:db8::1"])
        monkeypatch.setattr(network_module.socket, "getaddrinfo", resolver)

        result = await route_module.try_model_with_provider(
            _openai("http://[2001:db8::1]:8080")
        )

        assert result.accessible is True
        assert resolver.hosts == ["2001:db8::1"]
        assert asked[0]["base"] == "http://[2001:db8::1]:8080"

    @pytest.mark.asyncio
    async def test_an_unlisted_ipv6_target_is_refused_not_500(self, monkeypatch):
        _gate(monkeypatch, allowlist=["10.0.0.0/8"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())
        resolver = MockGetaddrinfo(["2001:db8::1"])
        monkeypatch.setattr(network_module.socket, "getaddrinfo", resolver)

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                _openai("http://[2001:db8::1]:8080")
            )

        assert asked == []
        assert "2001:db8::1" in raised.value.message

    @pytest.mark.asyncio
    async def test_an_ipv6_proxy_is_checked_like_a_target(self, monkeypatch):
        _gate(monkeypatch, allowlist=["api.example.com"])
        asked = _stub_upstream(monkeypatch, lambda _request: _ok())
        resolver = MockGetaddrinfo(["fe80::1"])
        monkeypatch.setattr(network_module.socket, "getaddrinfo", resolver)

        with pytest.raises(InvalidException) as raised:
            await route_module.try_model_with_provider(
                TestProviderModelInput(
                    api_token="sk-test",
                    config=OpenAIConfig(
                        type=ModelProviderTypeEnum.OPENAI,
                        openaiCustomUrl="http://api.example.com:8080",
                    ),
                    model_name="gpt-test",
                    proxy_url="http://[fe80::1]:7890",
                )
            )

        assert asked == []
        assert "proxy URL" in raised.value.message
        assert "fe80::1" in raised.value.message
