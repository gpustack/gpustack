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


def test_get_server_url_falls_back_to_request_origin():
    server_url = get_server_url(
        _request(_url(scheme="https", hostname="gpustack.example", port=8443)),
        None,
    )

    assert server_url == "https://gpustack.example:8443"


def test_cluster_update_normalizes_server_url():
    cluster = ClusterUpdate(
        name="k8s",
        server_url="http://example.com:30080/",
    )

    assert cluster.server_url == "http://example.com:30080"


def test_config_normalizes_server_external_url(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = Config(
        data_dir=str(tmp_path / "data"),
        server_external_url="http://example.com:30080/",
    )

    assert config.server_external_url == "http://example.com:30080"
