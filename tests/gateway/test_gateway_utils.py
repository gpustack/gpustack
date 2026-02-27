from gpustack.gateway.utils import RoutePrefix, provider_registry
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
    OpenAIConfig,
    OllamaConfig,
)
from gpustack.gateway.client.networking_higress_io_v1_api import McpBridgeRegistry


def test_flattened_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"]
    ).flattened_prefixes() == [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1-openai/chat/completions",
        "/v1-openai/completions",
        "/v1-openai/responses",
    ]


def test_regex_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"]
    ).regex_prefixes() == [
        r"/(v1)(-openai)?(/chat/completions)",
        r"/(v1)(-openai)?(/completions)",
        r"/(v1)(-openai)?(/responses)",
    ]
    assert RoutePrefix(
        ["/chat/completions", "/completions"], support_legacy=False
    ).regex_prefixes() == [
        r"/(v1)()(/chat/completions)",
        r"/(v1)()(/completions)",
    ]


def test_provider_registry_static_ip():
    provider = ModelProvider(
        id=1,
        name="provider-1",
        config=OpenAIConfig(
            type=ModelProviderTypeEnum.OPENAI, openaiCustomUrl="http://1.2.3.4/v1"
        ),
        proxy_url="http://proxy.example.com:8080",
    )
    reg = provider_registry(provider)
    assert isinstance(reg, McpBridgeRegistry)
    assert reg.domain == "1.2.3.4:80"
    assert reg.port == 80
    assert reg.protocol == "http"
    assert reg.type == "static"
    assert reg.name == "provider-1"
    assert reg.proxyName is not None
    assert reg.proxyName == "provider-1-proxy"


def test_provider_registry_dns():
    provider = ModelProvider(
        id=2,
        name="provider-2",
        config=OpenAIConfig(
            type=ModelProviderTypeEnum.OPENAI,
            openaiCustomUrl="https://provider.example.com:8443/v1",
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "provider.example.com"
    assert reg.port == 8443
    assert reg.protocol == "https"
    assert reg.type == "dns"
    assert reg.name == "provider-2"
    assert reg.proxyName is None


def test_ollama_registry():
    provider = ModelProvider(
        id=3,
        name="provider-3",
        config=OllamaConfig(
            type=ModelProviderTypeEnum.OLLAMA,
            ollamaServerHost="localhost",
            ollamaServerPort=8080,
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "localhost"
    assert reg.port == 8080
    assert reg.protocol == "http"
    assert reg.type == "dns"
    assert reg.name == "provider-3"

    provider = ModelProvider(
        id=3,
        name="provider-3",
        config=OllamaConfig(
            type=ModelProviderTypeEnum.OLLAMA,
            ollamaServerHost="1.2.3.4",
            ollamaServerPort=8080,
        ),
    )
    reg = provider_registry(provider)
    assert reg.domain == "1.2.3.4:8080"
    assert reg.port == 80
    assert reg.type == "static"
    assert reg.protocol == "http"
