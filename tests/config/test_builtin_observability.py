import json
import os
import subprocess
from pathlib import Path

import yaml

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
        builtin_prometheus_scrape_configs_dir=str(tmp_path / "missing"),
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
    assert "PROMETHEUS_ENABLE_REMOTE_WRITE_RECEIVER=false" in env_text
    assert "GF_SERVER_HTTP_PORT=13100" in env_text
    assert f"PROMETHEUS_DATA_DIR={tmp_path / 'data' / 'prometheus'}" in env_text
    assert "127.0.0.1:10161/metrics/targets" in prom_text
    assert "scrape_config_files:" in prom_text
    assert "url: http://127.0.0.1:19100/prometheus" in datasource_text


def test_prepare_prometheus_config_can_disable_drop_ins(tmp_path, monkeypatch):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        gateway_mode="disabled",
        builtin_prometheus_scrape_configs_dir=None,
    )
    prometheus_config = tmp_path / "prometheus.yml"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(prometheus_config))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(tmp_path / "grafana"))

    prepare_observability_config(cfg)

    assert (
        prometheus_config.read_text()
        == """# Managed by GPUStack
global:
  scrape_interval: 15s
  scrape_timeout: 10s
  evaluation_interval: 15s
scrape_configs:
  - job_name: gpustack-worker-discovery
    scrape_interval: 5s
    http_sd_configs:
      - url: "http://127.0.0.1:10161/metrics/targets"
        refresh_interval: 1m
  - job_name: gpustack-proxy-worker-discovery
    scrape_interval: 5s
    proxy_url: "http://127.0.0.1:30079"
    http_sd_configs:
      - url: "http://127.0.0.1:10161/metrics/proxy-targets"
        refresh_interval: 1m
  - job_name: gpustack-server
    scrape_interval: 5s
    static_configs:
      - targets:
          - 127.0.0.1:10161
"""
    )


@pytest.mark.parametrize("empty", [False, True])
def test_prepare_prometheus_config_includes_globs_for_missing_or_empty_directory(
    tmp_path, monkeypatch, empty
):
    drop_in = tmp_path / "scrape_configs.d"
    if empty:
        drop_in.mkdir()
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        builtin_prometheus_scrape_configs_dir=str(drop_in),
    )
    prometheus_config = tmp_path / "prometheus.yml"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(prometheus_config))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(tmp_path / "grafana"))

    prepare_observability_config(cfg)

    assert yaml.safe_load(prometheus_config.read_text())["scrape_config_files"] == [
        f"{drop_in}/*.yml",
        f"{drop_in}/*.yaml",
    ]


def test_prepare_prometheus_config_includes_scrape_config_globs(tmp_path, monkeypatch):
    drop_in = tmp_path / r"scrape\?[configs] #.d"
    drop_in.mkdir()
    (drop_in / "extra.yml").write_text("scrape_configs: []\n")
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        builtin_prometheus_scrape_configs_dir=str(drop_in),
    )
    prometheus_config = tmp_path / "prometheus.yml"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(prometheus_config))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(tmp_path / "grafana"))

    prepare_observability_config(cfg)

    prom_text = prometheus_config.read_text()
    scrape_config_files = yaml.safe_load(prom_text)["scrape_config_files"]
    expected_drop_in = str(drop_in).replace("\\", "\\\\")
    expected_drop_in = expected_drop_in.replace("?", "\\?").replace("[", "\\[")
    assert "job_name: gpustack-worker-discovery" in prom_text
    assert scrape_config_files == [
        f"{expected_drop_in}/*.yml",
        f"{expected_drop_in}/*.yaml",
    ]


def test_prepare_prometheus_config_rejects_literal_star_in_directory(
    tmp_path, monkeypatch
):
    drop_in = tmp_path / "scrape*configs.d"
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        builtin_prometheus_scrape_configs_dir=str(drop_in),
    )
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(tmp_path / "prometheus.yml"))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(tmp_path / "grafana"))

    with pytest.raises(ValueError, match="must not contain '\\*'"):
        prepare_observability_config(cfg)


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_prometheus_config_sets_remote_write_receiver(
    tmp_path, monkeypatch, enabled
):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        builtin_prometheus_remote_write_receiver=enabled,
    )
    tmp_run = tmp_path / "run"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_run))
    monkeypatch.setenv("PROMETHEUS_CONFIG_FILE", str(tmp_path / "prometheus.yml"))
    monkeypatch.setenv("GF_PATHS_PROVISIONING", str(tmp_path / "grafana"))

    prepare_observability_config(cfg)

    env_text = (tmp_run / "observability" / ".env").read_text()
    assert f"PROMETHEUS_ENABLE_REMOTE_WRITE_RECEIVER={str(enabled).lower()}" in env_text


@pytest.mark.parametrize("enabled", [False, True])
def test_prometheus_run_sets_remote_write_receiver_flag(tmp_path, enabled):
    run_script = (
        Path(__file__).resolve().parents[2]
        / "pack/rootfs/etc/s6-overlay/s6-rc.d/prometheus/run"
    ).read_text()
    fake_prometheus = tmp_path / "prometheus"
    captured_args = tmp_path / "args"
    fake_prometheus.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" > {captured_args!s}\n'
    )
    fake_prometheus.chmod(0o755)
    run_script = run_script.replace("/usr/local/bin/prometheus", str(fake_prometheus))
    run_script = run_script.replace(
        'SCRIPT_ROOT=/etc/s6-overlay/scripts\nsource "$SCRIPT_ROOT/base.sh"',
        f'SCRIPT_ROOT={tmp_path!s}\nGPUSTACK_OBSERVABILITY_CONFIG={tmp_path / "observability.env"!s}\ncreateDir() {{ mkdir -p "$1"; }}',
    )
    (tmp_path / "default-variables.sh").write_text("")
    (tmp_path / "observability.env").write_text("")
    prometheus_config = tmp_path / "prometheus.yml"
    prometheus_config.write_text("")
    env = os.environ | {
        "LOG_DIR": str(tmp_path / "log"),
        "PROMETHEUS_CONFIG_FILE": str(prometheus_config),
        "PROMETHEUS_DATA_DIR": str(tmp_path / "data"),
        "PROMETHEUS_PORT": "19090",
        "PROMETHEUS_ENABLE_REMOTE_WRITE_RECEIVER": str(enabled).lower(),
    }

    subprocess.run(["bash", "-c", run_script], check=True, env=env)

    args = captured_args.read_text().splitlines()
    assert ("--web.enable-remote-write-receiver" in args) is enabled
    assert args[:2] == [
        f"--config.file={prometheus_config}",
        f"--storage.tsdb.path={tmp_path / 'data'}",
    ]


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


def test_cache_service_dashboard_json_matches_route_contract(tmp_path):
    """The bundled dashboard must carry the uid the redirect endpoint uses
    by default and the template variables the redirect query string sets."""
    dashboard_path = (
        Path(__file__).resolve().parents[2]
        / "docker-compose"
        / "grafana"
        / "grafana_dashboards"
        / "gpustack-cache-service.json"
    )
    with dashboard_path.open() as f:
        dashboard = json.load(f)

    cfg = Config(data_dir=str(tmp_path / "data"))
    assert dashboard["uid"] == cfg.grafana_cache_service_dashboard_uid
    assert dashboard["title"] == "GPUStack Cache Service"

    # The redirect query string sets these two; further variables (worker
    # filter, attached-models chain) are dashboard-internal.
    var_names = [var["name"] for var in dashboard["templating"]["list"]]
    assert var_names[:2] == ["cluster_name", "cache_service_name"]


def test_provider_dashboard_json_matches_declaration():
    """Every provider declaring a dashboard_uid must ship a provisioned
    dashboard JSON with that uid and the template variables the redirect
    query string sets."""
    from gpustack.server.cache_provider_catalog import load_cache_providers

    dashboards_dir = (
        Path(__file__).resolve().parents[2]
        / "docker-compose"
        / "grafana"
        / "grafana_dashboards"
    )
    dashboards_by_uid = {}
    for path in dashboards_dir.glob("*.json"):
        with path.open() as f:
            dashboard = json.load(f)
        dashboards_by_uid[dashboard["uid"]] = dashboard

    declared = [p for p in load_cache_providers(reload=True) if p.dashboard_uid]
    assert declared, "XSKY MeshFusion declares a provider dashboard"
    for provider in declared:
        dashboard = dashboards_by_uid.get(provider.dashboard_uid)
        assert dashboard is not None, (
            f"provider '{provider.name}' declares dashboard_uid "
            f"'{provider.dashboard_uid}' but no dashboard JSON ships with it"
        )
        var_names = [var["name"] for var in dashboard["templating"]["list"]]
        assert var_names[:2] == ["cluster_name", "cache_service_name"]
