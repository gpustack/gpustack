from types import SimpleNamespace
from unittest.mock import patch

from gpustack.config.config import Config
from gpustack.routes.clusters import get_server_url
from gpustack.schemas.clusters import ClusterUpdate


def _request(url):
    return SimpleNamespace(url=url)


def _url(scheme="http", hostname="127.0.0.1", port=None):
    return SimpleNamespace(scheme=scheme, hostname=hostname, port=port)


def test_get_server_url_strips_trailing_slash_from_cluster_override():
    server_url = get_server_url(
        _request(_url()),
        "http://example.com:30080/",
    )

    assert server_url == "http://example.com:30080"
    assert f"{server_url}/v2/clusters/1/manifests" == (
        "http://example.com:30080/v2/clusters/1/manifests"
    )


def test_get_server_url_strips_trailing_slash_from_external_url():
    config = SimpleNamespace(server_external_url="http://example.com:30080/")

    with patch("gpustack.routes.clusters.get_global_config", return_value=config):
        server_url = get_server_url(_request(_url()), None)

    assert server_url == "http://example.com:30080"
    assert f"{server_url}/v2/clusters/1/manifests" == (
        "http://example.com:30080/v2/clusters/1/manifests"
    )


def _fallback_config(**overrides):
    config = SimpleNamespace(
        server_external_url=None,
        ssl_certfile=None,
        ssl_keyfile=None,
        api_port=30080,
        get_advertise_address=lambda: "10.0.0.5",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_get_server_url_ignores_forged_request_host():
    # Security: a forged X-Forwarded-Host (reflected in request.url) must never
    # leak into the worker-facing URL. Fall back to the advertise address.
    with patch(
        "gpustack.routes.clusters.get_global_config",
        return_value=_fallback_config(),
    ):
        server_url = get_server_url(
            _request(_url(scheme="https", hostname="evil.example", port=8443)),
            None,
        )

    assert server_url == "http://10.0.0.5:30080"
    assert "evil.example" not in server_url


def test_get_server_url_fallback_uses_https_when_tls_configured():
    with patch(
        "gpustack.routes.clusters.get_global_config",
        return_value=_fallback_config(ssl_certfile="cert.pem", ssl_keyfile="key.pem"),
    ):
        server_url = get_server_url(_request(_url()), None)

    assert server_url == "https://10.0.0.5:30080"


def test_cluster_update_normalizes_server_url():
    cluster = ClusterUpdate(
        name="k8s",
        server_url="http://example.com:30080/",
    )

    assert cluster.server_url == "http://example.com:30080"


def test_get_trusted_hosts_resolution(tmp_path):
    data_dir = str(tmp_path / "data")

    # Explicit trusted_hosts wins verbatim (including the "*" opt-out).
    explicit = Config(data_dir=data_dir, trusted_hosts=["a.example", "*"])
    assert explicit.get_trusted_hosts() == ["a.example", "*"]

    # Derived from server_external_url when trusted_hosts is unset.
    derived = Config(
        data_dir=data_dir, server_external_url="https://gpustack.example:30080"
    )
    assert derived.get_trusted_hosts() == ["gpustack.example"]

    # An explicit empty list behaves like unset: derive from server_external_url.
    empty = Config(
        data_dir=data_dir,
        trusted_hosts=[],
        server_external_url="https://gpustack.example:30080",
    )
    assert empty.get_trusted_hosts() == ["gpustack.example"]

    # Neither set -> empty (middleware ignores X-Forwarded-Host).
    assert Config(data_dir=data_dir).get_trusted_hosts() == []


def test_config_normalizes_server_external_url(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = Config(
        data_dir=str(tmp_path / "data"),
        server_external_url="http://example.com:30080/",
    )

    assert config.server_external_url == "http://example.com:30080"
