from starlette.requests import Request
import pytest

from gpustack.cmd.prerun import prepare_observability_config
from gpustack.config.config import Config
from gpustack.routes import prometheus as prometheus_routes


def test_config_uses_non_default_builtin_observability_ports(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))

    assert cfg.builtin_prometheus_port == 19090
    assert cfg.builtin_grafana_port == 13000
    assert cfg.get_builtin_prometheus_url() == "http://127.0.0.1:19090"
    assert cfg.get_grafana_url() == "http://127.0.0.1:13000"


def test_external_grafana_disables_builtin_prometheus_proxy_target(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        grafana_url="https://grafana.example.com",
    )

    assert cfg.get_grafana_url() == "https://grafana.example.com"
    assert cfg.get_builtin_prometheus_url() is None


def test_prepare_prometheus_config_writes_observability_env(tmp_path, monkeypatch):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        builtin_prometheus_port=19100,
        builtin_grafana_port=13100,
    )
    tmp_run = tmp_path / "run"
    observability_config_path = tmp_run / "observability" / ".env"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_run))

    prometheus_config = tmp_path / "prometheus" / "prometheus.yml"
    grafana_provisioning_dir = tmp_path / "grafana" / "provisioning"
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(prometheus_config))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(grafana_provisioning_dir))

    prepare_observability_config(cfg)

    env_text = observability_config_path.read_text()
    prom_text = prometheus_config.read_text()
    datasource_text = (
        grafana_provisioning_dir / "datasources" / "datasource.yaml"
    ).read_text()

    assert "PROMETHEUS_PORT=19100" in env_text
    assert "GF_SERVER_HTTP_PORT=13100" in env_text
    assert f"PROMETHEUS_DATA_DIR={tmp_path / 'data' / 'prometheus'}" in env_text
    assert "127.0.0.1:10161/metrics/targets" in prom_text
    assert "url: http://127.0.0.1:19100/prometheus" in datasource_text


@pytest.mark.asyncio
async def test_prometheus_proxy_uses_configured_builtin_port(monkeypatch, tmp_path):
    custom_port = 19999
    cfg = Config(data_dir=str(tmp_path / "data"), builtin_prometheus_port=custom_port)
    monkeypatch.setattr(prometheus_routes, "get_global_config", lambda: cfg)

    captured = {}

    async def fake_proxy_to(request, url):
        captured["url"] = url
        return {"url": url}

    monkeypatch.setattr(prometheus_routes, "proxy_to", fake_proxy_to)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/prometheus/api/v1/query",
            "query_string": b"query=up",
            "headers": [],
        }
    )

    response = await prometheus_routes.prometheus_proxy("api/v1/query", request)

    expected_url = f"http://127.0.0.1:{custom_port}/prometheus/api/v1/query?query=up"
    assert response == {"url": expected_url}
    assert captured["url"] == expected_url
