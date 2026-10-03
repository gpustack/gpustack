import json
import shlex
from pathlib import Path

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


def test_external_prometheus_wins_over_the_builtin_one(tmp_path):
    """The query target and the proxy target are not the same question: the
    proxy only ever fronts the embedded Prometheus, while the server's own
    queries follow the external one when it is configured."""
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        prometheus_url="https://prometheus.example.com/",
    )

    assert cfg.get_prometheus_url() == "https://prometheus.example.com"
    assert cfg.get_builtin_prometheus_url() == "http://127.0.0.1:19090"


def test_external_prometheus_survives_disabled_builtin_observability(tmp_path):
    """The case the setting exists for: observability delegated to somebody
    else's stack, where without it every server-side metric query goes dark."""
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        disable_builtin_observability=True,
        prometheus_url="https://prometheus.example.com",
    )

    assert cfg.get_builtin_prometheus_url() is None
    assert cfg.get_prometheus_url() == "https://prometheus.example.com"


def test_no_prometheus_at_all_is_a_first_class_answer(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"), disable_builtin_observability=True)

    assert cfg.get_prometheus_url() is None


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
    prometheus_data_dir = shlex.quote(str(tmp_path / 'data' / 'prometheus'))
    assert f"PROMETHEUS_DATA_DIR={prometheus_data_dir}" in env_text
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


def _dashboards_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "docker-compose"
        / "grafana"
        / "grafana_dashboards"
    )


def _variables(dashboard_file: str) -> dict:
    with (_dashboards_dir() / dashboard_file).open() as f:
        dashboard = json.load(f)
    return {var["name"]: var for var in dashboard["templating"]["list"]}


@pytest.mark.parametrize(
    "dashboard_file", [p.name for p in _dashboards_dir().glob("*.json")]
)
def test_custom_variable_options_are_comma_separated(dashboard_file):
    """A custom variable's query is a comma-separated list, and Grafana
    re-derives the options from it on load — so a hand-written `options`
    array agreeing with a query that does not parse hides the fault. A `|`
    between values yields one option whose text is the whole string, and a
    variable read as `x=~"$var"` then matches everything no matter which of
    its two entries the reader picks: a filter that silently does nothing.
    """
    for name, var in _variables(dashboard_file).items():
        if var.get("type") != "custom":
            continue
        query = var["query"]
        assert isinstance(query, str), f"{dashboard_file}:{name} query must be a string"
        values = [value.strip() for value in query.split(",")]
        for value in values:
            assert value, f"{dashboard_file}:{name} has an empty option"
            assert "|" not in value, (
                f"{dashboard_file}:{name} option {value!r} carries a '|'; "
                "custom variables split on ',' and this becomes one option"
            )


def test_pd_dashboard_derives_the_counted_role_from_the_model():
    """Which side of a pair counts KV transfers follows from the connector,
    so the dashboard reads it off the exporter rather than asking. A custom
    variable here would leave it wherever the last reader left it, and the
    transfer panels would report a flat zero on a healthy group whose
    connector moves bytes the other way.
    """
    var = _variables("gpustack-pd.json")["counted_role"]

    assert var["type"] == "query"
    assert "gpustack:pd_mode_info" in var["definition"]
    assert "$model_name" in var["definition"]
    assert var["regex"] == '/kv_counted_role="([^"]+)"/'


def test_provider_dashboard_json_matches_declaration():
    """A provider dashboard and the declaration pointing at it ship
    together. A declared uid with no dashboard is a dead link from the
    service page; a dashboard no provider claims is dead weight — what a
    provider moving to a package of its own leaves behind. Neither side
    is checked by the other's absence, so both are."""
    from gpustack.config.config import Config
    from gpustack.server.cache_provider_catalog import asset_providers

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

    declared = {
        provider.dashboard_uid: provider.name
        for provider in asset_providers()
        if provider.dashboard_uid
    }
    for uid, name in declared.items():
        dashboard = dashboards_by_uid.get(uid)
        assert dashboard is not None, (
            f"provider '{name}' declares dashboard_uid '{uid}' but no "
            "dashboard JSON ships with it"
        )
        var_names = [var["name"] for var in dashboard["templating"]["list"]]
        assert var_names[:2] == ["cluster_name", "cache_service_name"]

    # The platform's own dashboards answer to the config defaults rather
    # than to a provider; a gpustack-prefixed one that neither claims is
    # a leftover.
    fields = Config.model_fields
    platform = {
        fields[name].default
        for name in (
            "grafana_worker_dashboard_uid",
            "grafana_model_dashboard_uid",
            "grafana_pd_dashboard_uid",
            "grafana_cache_service_dashboard_uid",
        )
    }
    # Held to the same two-way rule as a provider's: naming a uid in the
    # config is the declaration, so a default pointing at nothing is the
    # same dead link a provider's would be.
    missing = sorted(uid for uid in platform if uid not in dashboards_by_uid)
    assert not missing, f"config defaults with no dashboard JSON: {missing}"

    orphans = sorted(
        uid
        for uid in dashboards_by_uid
        if uid.startswith("gpustack-") and uid not in platform and uid not in declared
    )
    assert not orphans, f"dashboards nothing declares: {orphans}"
