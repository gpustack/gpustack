import ssl

from gpustack.config.config import get_openid_configuration


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"issuer": "https://issuer.example.com"}


def test_get_openid_configuration_uses_system_trust_store(monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get(self, url):
            captured["url"] = url
            return _FakeResponse()

    monkeypatch.setattr("gpustack.config.config.httpx.Client", _FakeClient)
    monkeypatch.setattr(
        "gpustack.config.config.use_proxy_env_for_url", lambda url: False
    )

    result = get_openid_configuration("https://issuer.example.com")

    assert result == {"issuer": "https://issuer.example.com"}
    assert (
        captured["url"] == "https://issuer.example.com/.well-known/openid-configuration"
    )
    assert captured["timeout"] == 10
    assert captured["trust_env"] is False
    assert isinstance(captured["verify"], ssl.SSLContext)
