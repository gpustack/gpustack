import typing

import pytest
from pydantic import TypeAdapter

from gpustack.gateway.ai_proxy_types import AIProxyDefaultConfig
from gpustack.schemas.model_provider import (
    ANTHROPIC_API_VERSION,
    ClaudeConfig,
    ModelProviderTypeEnum,
    ProviderConfigType,
)


def _config_of(type_value: str):
    return TypeAdapter(ProviderConfigType).validate_python({"type": type_value})


class TestProviderTypeCoverage:
    """``ModelProviderTypeEnum`` mirrors the provider types ai-proxy accepts, so
    it is edited whenever the plugin gains one -- and adding the member is only
    half of it. A member with no class in ``ProviderConfigType`` fails the same
    way for every provider of that type, and fails silently at the schema
    rather than anywhere near the plugin: the config cannot be validated, so it
    cannot be saved.

    ``generic`` is the standing exception. Its config
    yields no base URL, so ``provider_registry`` returns None and the provider
    is dropped from the ai-proxy config -- it needs an endpoint field before the
    union should accept it. Listed rather than skipped so that wiring it up
    trips this test instead of leaving a stale exemption behind.
    """

    UNCONSTRUCTIBLE = {ModelProviderTypeEnum.GENERIC}

    def _declared_types(self):
        return {
            typing.get_args(c.model_fields["type"].annotation)[0]
            for c in typing.get_args(ProviderConfigType)
        }

    def test_every_type_has_a_config_class(self):
        expected = set(ModelProviderTypeEnum) - self.UNCONSTRUCTIBLE
        assert self._declared_types() == expected

    @pytest.mark.parametrize("provider_type", sorted(UNCONSTRUCTIBLE))
    def test_the_exempt_types_really_are_unconstructible(self, provider_type):
        with pytest.raises(ValueError):
            _config_of(provider_type.value)

    @pytest.mark.parametrize("provider_type", list(ModelProviderTypeEnum))
    def test_the_type_reaches_the_ai_proxy_config(self, provider_type):
        # ai-proxy keys its provider registry off this value; a type the enum
        # carries but AIProxyDefaultConfig rejects would fail at reconcile time.
        spec = AIProxyDefaultConfig(id="provider-1", type=provider_type)
        assert spec.model_dump(exclude_none=True)["type"] == provider_type


class TestGaladriel:
    """Added in ai-proxy 2.0.1-patched. A thin OpenAI-compatible shell: no
    provider-specific config fields, so what it needs from us is the endpoint
    the plugin hardcodes plus the two OpenAI paths it declares as its default
    capabilities."""

    def test_endpoints_match_the_plugin(self):
        config = _config_of("galadriel")

        assert config.get_base_url() == "https://api.galadriel.com"
        assert config.get_chat_url() == (
            "https://api.galadriel.com",
            "/v1/chat/completions",
        )
        # The plugin declares ApiNameModels, so model discovery works and must
        # not fall through the "not supported for fetching models" branch.
        assert config.get_model_url() == ("https://api.galadriel.com", "/v1/models")

    def test_it_carries_no_provider_specific_fields(self):
        assert _config_of("galadriel").model_dump(exclude_unset=True) == {
            "type": ModelProviderTypeEnum.GALADRIEL
        }


def _claude(**kwargs) -> ClaudeConfig:
    return ClaudeConfig(type=ModelProviderTypeEnum.CLAUDE, **kwargs)


class TestClaudeCustomUrl:
    """Pointing a Claude provider at an Anthropic-compatible endpoint.

    ai-proxy's claude provider hardcodes ``api.anthropic.com`` and has no URL
    field of its own, so one URL of ours becomes two of the plugin's. The
    endpoint has to land in two independent places -- the McpBridge registry
    (which decides where Envoy connects) and providerDomain (which decides the
    authority it sends) -- and a value that reaches only one of them is a
    request sent to the right host with the wrong Host header, or the reverse.
    """

    def test_unset_leaves_anthropic_untouched(self):
        config = _claude()

        assert config.get_base_url() == "https://api.anthropic.com"
        assert config.get_chat_url() == ("https://api.anthropic.com", "/v1/messages")
        # Nothing derived at all, so an existing provider's plugin config is
        # byte-for-byte what is already deployed.
        assert config.ai_proxy_derived_fields() == {}

    def test_the_custom_host_becomes_the_base_url(self):
        # This is what provider_registry parses, so it decides the registry's
        # domain, port, protocol, and static-vs-dns type.
        config = _claude(claudeCustomUrl="http://192.168.50.14:8080")

        assert config.get_base_url() == "http://192.168.50.14:8080"

    def test_the_plugin_knobs(self):
        config = _claude(claudeCustomUrl="http://192.168.50.14")

        # Two keys, and only two. No protocol, because a custom endpoint is
        # still exposed as OpenAI unless the operator asks for passthrough, and
        # no providerBasePath, because an endpoint at the root has no prefix to
        # prepend -- ``applyProviderBasePath`` skips a path that already starts
        # with the value, so "/" would be a no-op key in every deployed config.
        assert config.ai_proxy_derived_fields() == {
            "providerDomain": "192.168.50.14",
        }

    def test_a_port_stays_in_the_authority(self):
        # providerDomain becomes :authority verbatim; dropping the port would
        # send Host: 192.168.50.14 to a server listening on 8080.
        config = _claude(claudeCustomUrl="http://192.168.50.14:8080")

        assert config.ai_proxy_derived_fields()["providerDomain"] == (
            "192.168.50.14:8080"
        )

    def test_a_base_path_prefixes_both_the_plugin_and_our_own_calls(self):
        config = _claude(claudeCustomUrl="https://gw.example.com/anthropic/")

        assert config.ai_proxy_derived_fields()["providerBasePath"] == "/anthropic"
        # get_chat_url / get_model_url are what the server itself calls for
        # test-model and get-models, bypassing the gateway -- so the prefix has
        # to be applied here too, by us.
        assert config.get_chat_url() == (
            "https://gw.example.com",
            "/anthropic/v1/messages",
        )
        assert config.get_model_url() == (
            "https://gw.example.com",
            "/anthropic/v1/models",
        )

    def test_the_derived_fields_reach_the_ai_proxy_config(self):
        # AIProxyDefaultConfig ignores fields it does not declare, and dropping
        # these is silent: the config validates, the plugin loads, and traffic
        # goes to api.anthropic.com.
        config = _claude(claudeCustomUrl="http://192.168.50.14:8080/anthropic")

        spec = AIProxyDefaultConfig.model_validate(
            {
                "id": "provider-1",
                "type": config.type,
                **config.model_dump_with_default_override(),
            }
        ).model_dump(exclude_none=True)

        assert spec["providerDomain"] == "192.168.50.14:8080"
        assert spec["providerBasePath"] == "/anthropic"
        assert "protocol" not in spec

    def test_passthrough_is_reached_by_setting_the_plugin_field(self):
        # extra="allow" is the escape hatch for a knob we have not modelled, and
        # protocol is the one an Anthropic-compatible endpoint most often wants:
        # "original" forwards unconverted instead of round-tripping via OpenAI.
        config = _claude(claudeCustomUrl="http://192.168.50.14", protocol="original")

        spec = AIProxyDefaultConfig.model_validate(
            {
                "id": "provider-1",
                "type": config.type,
                **config.model_dump_with_default_override(),
            }
        )

        assert spec.protocol == "original"

    @pytest.mark.parametrize(
        "url",
        [
            "192.168.50.14",  # parses as a path, leaving no host at all
            "192.168.50.14:8080",  # and this one as scheme "192.168.50.14"
            "ftp://192.168.50.14",
            "/anthropic",
        ],
    )
    def test_a_url_with_no_usable_host_is_refused(self, url):
        # Accepting it would build the base URL "https://" and an McpBridge
        # registry with an empty domain, which fails at reconcile time -- far
        # from the field that caused it.
        with pytest.raises(ValueError, match="claudeCustomUrl"):
            _claude(claudeCustomUrl=url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:pw@10.0.0.1:8080",
            "http://user@10.0.0.1:8080",
            "https://:pw@gw.example.com",
        ],
    )
    def test_credentials_in_the_url_are_refused(self, url):
        # The netloc travels on as providerDomain, which ai-proxy writes to
        # :authority verbatim -- so accepting this would send the credentials in
        # a header on every proxied request, and log them all the way along.
        with pytest.raises(ValueError, match="must not carry credentials"):
            _claude(claudeCustomUrl=url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.1:8080?key=abc",
            "http://10.0.0.1:8080/anthropic#frag",
        ],
    )
    def test_a_query_or_fragment_is_refused(self, url):
        # Only the origin and the path are read. Accepting the rest would imply
        # it is sent, and an endpoint that needs a query parameter cannot be
        # reached this way at all.
        with pytest.raises(ValueError, match="origin and an optional path"):
            _claude(claudeCustomUrl=url)

    def test_credentials_never_reach_the_derived_fields(self):
        # The assertion the validator exists for, stated where the value is
        # consumed: nothing that could carry a secret gets this far.
        with pytest.raises(ValueError):
            _claude(claudeCustomUrl="http://user:pw@10.0.0.1:8080")

        clean = _claude(claudeCustomUrl="http://10.0.0.1:8080")

        assert clean.ai_proxy_derived_fields()["providerDomain"] == "10.0.0.1:8080"
        assert clean.get_base_url() == "http://10.0.0.1:8080"


class TestClaudeVersion:
    """``anthropic-version`` is sent explicitly rather than left to ai-proxy.

    The plugin defaults it to the same value today, but that default is the
    plugin's: a re-sync can move it, and the two paths into a Claude provider
    would then disagree -- the gateway on the plugin's version, get-models and
    test-model on ours. The failure that produces is a provider that tests fine
    from the UI and fails in inference, or the reverse.
    """

    def test_the_field_defaults_to_the_shared_constant(self):
        assert _claude().claudeVersion == ANTHROPIC_API_VERSION

    def test_the_route_layer_uses_the_same_constant(self):
        # The route layer must hold the same object, not its own copy of the
        # string: two definitions is what lets the two paths drift apart.
        from gpustack.routes import model_provider as route_module

        assert route_module.ANTHROPIC_API_VERSION is ANTHROPIC_API_VERSION

    def test_the_version_reaches_the_plugin_even_when_never_set(self):
        # exclude_unset drops the field default, so a provider stored before
        # this default existed dumps no version at all -- the override is what
        # keeps those on our value instead of the plugin's.
        stored_before_the_default = ClaudeConfig.model_validate(
            {"type": ModelProviderTypeEnum.CLAUDE}
        )

        values = stored_before_the_default.model_dump_with_default_override()

        assert values["claudeVersion"] == ANTHROPIC_API_VERSION

    def test_an_explicit_version_wins(self):
        config = _claude(claudeVersion="2025-01-01")

        assert config.model_dump_with_default_override()["claudeVersion"] == (
            "2025-01-01"
        )

    def test_it_survives_the_ai_proxy_config(self):
        config = _claude()

        spec = AIProxyDefaultConfig.model_validate(
            {
                "id": "provider-1",
                "type": config.type,
                **config.model_dump_with_default_override(),
            }
        )

        assert spec.claudeVersion == ANTHROPIC_API_VERSION
