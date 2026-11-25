import argparse
from pathlib import Path
import json
import re

import pytest
import yaml

from gpustack.cmd import prerun
from gpustack.cmd.prerun import check_ports_availability, prepare_s6_overlay
from gpustack.config.config import Config

HIGRESS_CONFIGMAP_PATH = Path("configmaps/higress-config.yaml")
HIGRESS_FILTER_PATH = Path("envoyfilters/gpustack-otel-tracing.yaml")


def test_prepare_s6_overlay_enables_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)

    stale_migration = user_contents / "gpustack-migration"
    stale_migration.write_text("")

    # prepare_s6_overlay should cleanup the base dir and generate base on the input services
    prepare_s6_overlay(["postgres"], ["gpustack-migration"], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert (user_contents / "gpustack-migration").exists()


def test_prepare_s6_overlay_cleans_dependency_only_services(tmp_path):
    s6_base_path = tmp_path / "s6-rc.d"
    user_contents = s6_base_path / "user" / "contents.d"
    user_contents.mkdir(parents=True)
    (user_contents / "gpustack-migration").write_text("")

    prepare_s6_overlay(["postgres"], [], s6_base_path)

    assert (user_contents / "postgres").exists()
    assert not (user_contents / "gpustack-migration").exists()


def test_builtin_tempo_config_and_service(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run_dir))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES", raising=False)
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        gateway_mode="disabled",
        disable_builtin_observability=True,
        builtin_tempo_otlp_port=13417,
        builtin_tempo_http_port=13200,
    )
    s6_base = tmp_path / "s6-rc.d"
    service = s6_base / "user" / "contents.d" / "tempo"
    active = prerun.prepare_builtin_tempo(cfg)
    services = prerun.determine_enabled_services(cfg, tempo_active=active)
    prepare_s6_overlay(services, [], s6_base)

    config = yaml.safe_load((run_dir / "tempo" / "config.yaml").read_text())
    assert service.exists()
    assert "tempo" not in prerun.determine_dependency_services(cfg)
    assert "grafana" not in services
    assert config["server"] == {
        "http_listen_address": "127.0.0.1",
        "http_listen_port": 13200,
        "grpc_listen_address": "127.0.0.1",
        "grpc_listen_port": 9095,
    }
    receiver = config["distributor"]["receivers"]["otlp"]["protocols"]
    assert receiver == {"grpc": {"endpoint": "0.0.0.0:13417"}}
    assert config["storage"]["trace"] == {
        "backend": "local",
        "blocklist_poll": "1m",
        "wal": {"path": str(tmp_path / "data" / "tempo" / "wal")},
        "local": {"path": str(tmp_path / "data" / "tempo" / "blocks")},
    }
    assert config["compactor"]["compaction"] == {
        "block_retention": "168h",
        "max_retained_bytes": 30_000_000_000,
    }
    assert config["ingester"]["flush_all_on_shutdown"] is True
    assert {
        port
        for port, service_name in prerun.ports_for_services(cfg).items()
        if service_name == "tempo"
    } == {13417, 13200, 9095}
    with pytest.raises(Exception, match="Port conflict: 9095"):
        prerun.ports_for_services(cfg.model_copy(update={"port": 9095}))

    monkeypatch.setattr(prerun, "is_port_available", lambda port: port != 13417)
    monkeypatch.setattr(prerun, "is_port_reusable", lambda port: False)
    check_ports_availability(cfg, *services)
    assert service.exists()
    assert cfg.effective_trace_otlp_endpoint.endswith(":13417")

    external = cfg.model_copy(update={"trace_otlp_endpoint": "collector:4317"})
    active = prerun.prepare_builtin_tempo(external)
    services = prerun.determine_enabled_services(external, tempo_active=active)
    prepare_s6_overlay(services, [], s6_base)
    assert not service.exists()
    assert "tempo" not in prerun.ports_for_services(external).values()


@pytest.mark.parametrize("run_dir", [None, ""])
def test_helm_tempo_uses_default_run_dir(tmp_path, monkeypatch, run_dir):
    if run_dir is None:
        monkeypatch.delenv("GPUSTACK_RUN_DIR", raising=False)
    else:
        monkeypatch.setenv("GPUSTACK_RUN_DIR", run_dir)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes")
    monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_POD", "gpustack-server-0")
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        gateway_mode="disabled",
        server_id="gpustack-server-0",
        trace_otlp_endpoint="gpustack-tempo:4317",
    )
    paths = []
    monkeypatch.setattr(
        prerun, "_write_yaml_atomically", lambda path, _content: paths.append(path)
    )

    assert prerun.prepare_builtin_tempo(cfg)
    assert paths == [Path("/run/gpustack/tempo/config.yaml")]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12000000000", 12_000_000_000),
        ("30 GB", 30_000_000_000),
        ("0", 30_000_000_000),
        ("-1", 30_000_000_000),
        (str(2**64), 30_000_000_000),
    ],
)
def test_builtin_tempo_max_retained_bytes(
    tmp_path, monkeypatch, caplog, value, expected
):
    run_dir = tmp_path / "run"
    env_name = "GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run_dir))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv(env_name, value)
    cfg = Config(data_dir=str(tmp_path / "data"), gateway_mode="disabled")

    assert prerun.prepare_builtin_tempo(cfg)

    config = yaml.safe_load((run_dir / "tempo" / "config.yaml").read_text())
    assert config["compactor"]["compaction"]["max_retained_bytes"] == expected
    assert (env_name in caplog.text) is (value != str(expected))


@pytest.mark.parametrize("helm", [False, True])
@pytest.mark.parametrize(
    "value,expected",
    [
        ("", "168h"),
        ("+1h30.5m", "+1h30.5m"),
        ("1.5h", "1.5h"),
        ("7days", "168h"),
        ("1h unknown", "168h"),
        ("0s", "168h"),
        ("-1h", "168h"),
        ("9223372036854775807ns", "9223372036854775807ns"),
        ("9223372036854775808ns", "168h"),
    ],
)
def test_builtin_tempo_retention(tmp_path, monkeypatch, caplog, helm, value, expected):
    run_dir = tmp_path / "run"
    env_name = "GPUSTACK_BUILTIN_TEMPO_RETENTION"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run_dir))
    monkeypatch.setenv(env_name, value)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("GPUSTACK_BUILTIN_TEMPO_POD", raising=False)
    if helm:
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes")
        monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_POD", "gpustack-server-0")
        monkeypatch.setenv("GPUSTACK_SERVER_ID", "gpustack-server-0")
    cfg = Config(data_dir=str(tmp_path / "data"), gateway_mode="disabled")

    assert prerun.prepare_builtin_tempo(cfg)

    config = yaml.safe_load((run_dir / "tempo" / "config.yaml").read_text())
    assert config["compactor"]["compaction"]["block_retention"] == expected
    assert (env_name in caplog.text) is bool(value and value != expected)


def test_disable_tracing_environment_skips_tempo_service_and_ports(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run_dir))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("GPUSTACK_DISABLE_TRACING", "true")
    cfg = Config(data_dir=str(tmp_path / "data"), gateway_mode="disabled")

    assert not prerun.prepare_builtin_tempo(cfg)
    assert not (run_dir / "tempo").exists()
    assert "tempo" not in prerun.ports_for_services(cfg).values()
    assert cfg.effective_trace_otlp_endpoint is None


def test_tempo_config_write_failure_skips_service_without_aborting_prerun(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        database_url="postgresql://user:pass@database.example/db",
        gateway_mode="disabled",
        disable_builtin_observability=True,
    )
    s6_base = tmp_path / "s6-rc.d"
    prepare_s6_overlay(["tempo"], [], s6_base)
    monkeypatch.setattr(prerun, "parse_args", lambda args: cfg)
    monkeypatch.setattr(prerun, "is_port_available", lambda port: True)

    def fail_to_write(path, content):
        raise PermissionError("Tempo config is not writable")

    monkeypatch.setattr(prerun, "_write_yaml_atomically", fail_to_write)

    prerun.run(argparse.Namespace(s6_base_path=str(s6_base)))

    assert "Unable to configure embedded Tempo" in caplog.text
    assert not (s6_base / "user" / "contents.d" / "tempo").exists()


@pytest.mark.parametrize(
    "server_id,disable_tracing,expected",
    [
        ("gpustack-server-0", False, True),
        ("gpustack-server-1", False, False),
        ("gpustack-server-0", True, False),
    ],
)
@pytest.mark.parametrize("tls", [False, True])
def test_helm_tempo_uses_the_selected_servers_data_dir(
    tmp_path, monkeypatch, server_id, disable_tracing, expected, tls
):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "server-data"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run_dir))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes")
    monkeypatch.setenv("GPUSTACK_SERVER_ID", server_id)
    monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_POD", "gpustack-server-0")
    monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_RETENTION", "12h")
    monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES", "12000000000")
    tls_dir = tmp_path / "tempo-tls"
    if tls:
        monkeypatch.setenv("GPUSTACK_BUILTIN_TEMPO_TLS_DIR", str(tls_dir))
    endpoint = "gpustack-tempo.inference.svc.cluster.local:4317"
    cfg = Config(
        data_dir=str(data_dir),
        gateway_mode="disabled",
        trace_otlp_endpoint=endpoint,
        disable_tracing=disable_tracing,
    )

    active = prerun.prepare_builtin_tempo(cfg)
    services = prerun.determine_enabled_services(cfg, tempo_active=active)
    assert active is expected
    assert ("tempo" in services) is expected
    assert (prerun.ports_for_services(cfg).get(4317) == "tempo") is expected
    assert "tempo" not in prerun.determine_dependency_services(cfg)
    assert cfg.effective_trace_otlp_endpoint == (None if disable_tracing else endpoint)
    config_path = run_dir / "tempo" / "config.yaml"
    assert config_path.exists() is expected
    if expected:
        config = yaml.safe_load(config_path.read_text())
        if tls:
            assert config["server"]["http_listen_address"] == "0.0.0.0"
            assert config["server"]["http_tls_config"] == {
                "cert_file": str(tls_dir / "server.crt"),
                "key_file": str(tls_dir / "server.key"),
                "client_ca_file": str(tls_dir / "ca.crt"),
                "client_auth_type": "RequireAndVerifyClientCert",
            }
        else:
            assert config["server"]["http_listen_address"] == "127.0.0.1"
            assert "http_tls_config" not in config["server"]
        assert config["storage"]["trace"] == {
            "backend": "local",
            "blocklist_poll": "1m",
            "wal": {"path": str(data_dir / "tempo" / "wal")},
            "local": {"path": str(data_dir / "tempo" / "blocks")},
        }
        assert config["compactor"]["compaction"] == {
            "block_retention": "12h",
            "max_retained_bytes": 12_000_000_000,
        }


def _one_gpustack_port(monkeypatch):
    monkeypatch.setattr(prerun, "ports_for_services", lambda cfg: {50000: "gpustack"})


def test_check_ports_availability_accepts_a_timewait_only_port(monkeypatch):
    """A recreated container leaves TIME_WAIT sockets on the fixed service
    ports; the check must not fail a port the services can rebind."""
    _one_gpustack_port(monkeypatch)
    monkeypatch.setattr(prerun, "is_port_available", lambda port, host=None: False)
    monkeypatch.setattr(prerun, "is_port_reusable", lambda port: True)

    check_ports_availability(None)


def test_check_ports_availability_fails_a_port_held_by_a_live_listener(monkeypatch):
    _one_gpustack_port(monkeypatch)
    monkeypatch.setattr(prerun, "is_port_available", lambda port, host=None: False)
    monkeypatch.setattr(prerun, "is_port_reusable", lambda port: False)

    with pytest.raises(Exception, match="not available"):
        check_ports_availability(None)


@pytest.fixture
def collector_path(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(run))
    template = (
        Path(__file__).parents[2] / "pack/rootfs/etc/gpustack/logging/docker.alloy.tmpl"
    )
    monkeypatch.setattr(prerun, "ALLOY_LOG_TEMPLATE", template)
    return run / "logging" / "config.alloy"


@pytest.mark.parametrize("worker", [False, True])
def test_log_collector_is_independent_of_builtin_observability(
    tmp_path, monkeypatch, collector_path, worker
):
    container_id = "a" * 64
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"17 16 8:1 /var/lib/docker/containers/{container_id}/hosts /etc/hosts "
        "rw,relatime - ext4 /dev/root rw\n"
        f"18 16 8:1 /var/lib/docker/containers/{container_id}/hostname /etc/hostname "
        "rw,relatime - ext4 /dev/root rw\n"
    )
    monkeypatch.setattr(prerun, "_DOCKER_MOUNTINFO", mountinfo)
    monkeypatch.setenv("HOSTNAME", "custom-hostname")
    monkeypatch.setattr(
        prerun.runtime_envs,
        "GPUSTACK_RUNTIME_DOCKER_HOST",
        "http+unix:///run/docker-alt.sock",
    )
    monkeypatch.setenv(
        "GPUSTACK_LOG_OTLP_ENDPOINT", "https://collector.example/v1/logs"
    )
    monkeypatch.setenv("GPUSTACK_LOG_OTLP_AUTHORIZATION", "Bearer literal$token")
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_dir=str(tmp_path / "logs"),
        disable_builtin_observability=True,
        gateway_mode="disabled",
        server_url="http://server" if worker else None,
        token="test" if worker else None,
    )

    assert prerun.prepare_log_collection(cfg)
    assert "alloy" in prerun.determine_enabled_services(cfg, logs_active=True)
    assert "prometheus" not in prerun.determine_enabled_services(cfg, logs_active=True)
    assert collector_path.stat().st_mode & 0o777 == 0o600
    rendered = collector_path.read_text()
    assert 'loki.source.docker "application"' in rendered
    assert 'host = "unix:///run/docker-alt.sock"' in rendered
    assert (
        f'targets = [{{"__meta_docker_container_id" = "{container_id}"}}]' in rendered
    )
    assert "otelcol.receiver.loki.application.receiver" in rendered
    assert "gpustack.logs.export.application" in rendered
    assert ".jsonl" not in rendered
    assert collector_path.with_name("storage-path").read_text() == str(
        tmp_path / "data/log-collector/alloy"
    )
    assert 'otelcol.receiver.filelog "higress"' not in rendered
    for name in ("container.log", "container.workload-*.log"):
        assert (
            str(tmp_path / f"logs/engine-export/archives/*/{name}.[0-9]*") in rendered
        )
    assert 'endpoint = "https://collector.example/v1/logs"' in rendered
    assert 'headers = {"Authorization" = "Bearer literal$token"}' in rendered


def test_missing_docker_identity_keeps_file_sources(
    tmp_path, monkeypatch, collector_path, caplog
):
    monkeypatch.setattr(prerun, "_DOCKER_MOUNTINFO", tmp_path / "missing-mountinfo")
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_dir=str(tmp_path / "logs"),
        gateway_mode="embedded",
        log_otlp_endpoint="http://collector/v1/logs",
    )
    assert prerun.prepare_log_collection(cfg)
    rendered = collector_path.read_text()
    assert (
        "Unable to identify this Docker container for application logs" in caplog.text
    )
    assert 'loki.source.docker "application"' not in rendered
    assert 'otelcol.receiver.filelog "higress"' in rendered
    assert 'otelcol.receiver.filelog "engine"' in rendered
    assert str(tmp_path / "logs/higress/access.log") in rendered
    assert str(tmp_path / "logs/higress/access.log.[1-5]") in rendered


@pytest.fixture
def embedded_gateway(tmp_path, monkeypatch):
    templates = (
        Path(__file__).parents[2] / "pack/rootfs/etc/gpustack/data/defaultConfig"
    )
    monkeypatch.setattr(prerun, "HIGRESS_DEFAULT_CONFIG", templates)
    monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = Config(data_dir=str(tmp_path / "data"), gateway_mode="embedded")
    return cfg, Path(cfg.higress_base_dir()) / "apiserver", templates


@pytest.mark.parametrize(
    ("endpoint", "host", "discovery"),
    [
        ("collector.example:4317", "collector.example", "STRICT_DNS"),
        ("192.0.2.1:4317", "192.0.2.1", "STATIC"),
        ("[2001:db8::1]:4317", "2001:db8::1", "STATIC"),
    ],
)
def test_embedded_otel_filter_address_and_request_id(
    embedded_gateway, endpoint, host, discovery
):
    cfg, apiserver, _ = embedded_gateway
    prerun.prepare_gateway_config(
        cfg.model_copy(update={"trace_otlp_endpoint": endpoint})
    )
    config = yaml.safe_load((apiserver / HIGRESS_FILTER_PATH).read_text())
    cluster_patch, hcm_patch, router_patch, routes_patch = config["spec"][
        "configPatches"
    ]
    cluster = cluster_patch["patch"]["value"]
    address = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]
    assert address["endpoint"]["address"]["socket_address"] == {
        "address": host,
        "port_value": 4317,
    }
    assert cluster["type"] == discovery
    (protocol,) = cluster["typed_extension_protocol_options"].values()
    assert protocol["explicit_http_config"] == {"http2_protocol_options": {}}
    hcm = hcm_patch["patch"]["value"]["typed_config"]
    tracing = hcm["tracing"]
    assert tracing["provider"]["typed_config"]["grpc_service"]["envoy_grpc"] == {
        "cluster_name": cluster["name"]
    }
    assert tracing["custom_tags"] == [
        {"tag": "gpustack.request_id", "request_header": {"name": "x-request-id"}}
    ]
    assert tracing["random_sampling"] == {"value": 100}
    assert tracing["max_path_tag_length"] == 0
    mapper = hcm["local_reply_config"]["mappers"][0]
    assert mapper["headers_to_add"] == [
        {
            "header": {"key": "x-gpustack-request-id", "value": "%REQ(X-REQUEST-ID)%"},
            "append_action": "ADD_IF_ABSENT",
        }
    ]
    assert (
        routes_patch["patch"]["value"]["response_headers_to_add"]
        == mapper["headers_to_add"]
    )
    assert router_patch["patch"]["value"]["typed_config"]["start_child_span"] is True


def test_disabled_tracing_leaves_fresh_higress_data_alone(embedded_gateway):
    cfg, apiserver, _ = embedded_gateway
    prerun.prepare_gateway_config(cfg.model_copy(update={"disable_tracing": True}))
    assert not apiserver.exists()


def test_embedded_logs_only_keeps_request_id_after_disabling_tracing(embedded_gateway):
    cfg, apiserver, _ = embedded_gateway
    logs = cfg.model_copy(update={"log_otlp_endpoint": "http://collector:4318/v1/logs"})
    prerun.prepare_gateway_config(
        logs.model_copy(update={"trace_otlp_endpoint": "collector:4317"})
    )

    prerun.prepare_gateway_config(logs.model_copy(update={"disable_tracing": True}))
    config = yaml.safe_load((apiserver / HIGRESS_FILTER_PATH).read_text())
    manager_patch, route_patch = config["spec"]["configPatches"]
    manager = manager_patch["patch"]["value"]["typed_config"]
    assert "tracing" not in manager
    expected = [
        {
            "header": {"key": "x-gpustack-request-id", "value": "%REQ(X-REQUEST-ID)%"},
            "append_action": "ADD_IF_ABSENT",
        }
    ]
    assert manager["local_reply_config"]["mappers"][0]["headers_to_add"] == expected
    assert route_patch["patch"]["value"]["response_headers_to_add"] == expected
    configmap = yaml.safe_load((apiserver / HIGRESS_CONFIGMAP_PATH).read_text())
    mesh = yaml.safe_load(configmap["data"]["mesh"])
    assert mesh["accessLogEncoding"] == "JSON"
    assert "traceparent" not in json.loads(mesh["accessLogFormat"])

    prerun.prepare_gateway_config(cfg.model_copy(update={"disable_tracing": True}))
    configmap = yaml.safe_load((apiserver / HIGRESS_CONFIGMAP_PATH).read_text())
    assert yaml.safe_load(configmap["data"]["mesh"])["accessLogEncoding"] == "TEXT"


def test_disabling_tracing_removes_filter_even_without_configmap(embedded_gateway):
    cfg, apiserver, _ = embedded_gateway
    filter_path = apiserver / HIGRESS_FILTER_PATH
    filter_path.parent.mkdir(parents=True)
    filter_path.write_text("stale generated resource")

    prerun.prepare_gateway_config(cfg.model_copy(update={"disable_tracing": True}))

    assert not filter_path.exists()
    assert not (apiserver / HIGRESS_CONFIGMAP_PATH).exists()


def test_embedded_tracing_preserves_operator_access_log_format(embedded_gateway):
    cfg, apiserver, templates = embedded_gateway
    configmap_path = apiserver / HIGRESS_CONFIGMAP_PATH
    configmap_path.parent.mkdir(parents=True)
    configmap = yaml.safe_load((templates / HIGRESS_CONFIGMAP_PATH).read_text())
    mesh = yaml.safe_load(configmap["data"]["mesh"])
    mesh["accessLogFormat"] = "%START_TIME% %REQ(X-REQUEST-ID)%"
    configmap["data"]["mesh"] = yaml.safe_dump(mesh)
    current_config = yaml.safe_dump(configmap)
    configmap_path.write_text(current_config)

    prerun.prepare_gateway_config(
        cfg.model_copy(update={"trace_otlp_endpoint": "collector.example:4317"})
    )
    assert (apiserver / HIGRESS_FILTER_PATH).exists()
    assert configmap_path.read_text() == current_config

    prerun.prepare_gateway_config(cfg.model_copy(update={"disable_tracing": True}))
    assert not (apiserver / HIGRESS_FILTER_PATH).exists()
    assert configmap_path.read_text() == current_config


@pytest.mark.parametrize(
    ("encoding", "access_log_format", "valid"),
    [
        ("JSON", '{"path":"%REQ(:PATH)%","request_id":"%REQ(X-REQUEST-ID)%"}', True),
        ("TEXT", '{"path":"%REQ(:PATH)%","request_id":"%REQ(X-REQUEST-ID)%"}', True),
        ("JSON", '{"path":"%REQ(:PATH)%"}', False),
        ("TEXT", "", False),
    ],
)
def test_embedded_log_collection_reports_incompatible_format(
    embedded_gateway, caplog, encoding, access_log_format, valid
):
    cfg, apiserver, templates = embedded_gateway
    configmap_path = apiserver / HIGRESS_CONFIGMAP_PATH
    configmap_path.parent.mkdir(parents=True)
    configmap = yaml.safe_load((templates / HIGRESS_CONFIGMAP_PATH).read_text())
    mesh = yaml.safe_load(configmap["data"]["mesh"])
    mesh["accessLogEncoding"] = encoding
    mesh["accessLogFormat"] = access_log_format
    configmap["data"]["mesh"] = yaml.safe_dump(mesh)
    contents = yaml.safe_dump(configmap)
    configmap_path.write_text(contents)

    prerun.prepare_gateway_config(
        cfg.model_copy(update={"log_otlp_endpoint": "http://collector/v1/logs"})
    )

    assert configmap_path.read_text() == contents
    assert (
        "Could not verify the Higress access-log template" in caplog.text
    ) is not valid


def test_embedded_gateway_reconfigures_tracing_without_replacing_existing_settings(
    embedded_gateway,
):
    cfg, root, templates = embedded_gateway
    cm_path = root / HIGRESS_CONFIGMAP_PATH
    cm_path.parent.mkdir(parents=True)
    original = yaml.safe_load((templates / HIGRESS_CONFIGMAP_PATH).read_text())
    mesh = yaml.safe_load(original["data"]["mesh"])
    mesh["dnsRefreshRate"] = "30s"
    original["data"]["mesh"] = yaml.safe_dump(mesh)
    cm_path.write_text(yaml.safe_dump(original))

    prerun.prepare_gateway_config(
        cfg.model_copy(update={"trace_otlp_endpoint": "collector.internal:4317"})
    )
    envoy_path = root / HIGRESS_FILTER_PATH
    assert envoy_path.exists()
    updated = yaml.safe_load(cm_path.read_text())
    updated_mesh = yaml.safe_load(updated["data"]["mesh"])
    assert updated_mesh["dnsRefreshRate"] == "30s"
    assert updated["data"]["higress"] == original["data"]["higress"]
    assert updated_mesh["accessLogEncoding"] == "JSON"
    assert "%REQ(TRACEPARENT):55%" in updated_mesh["accessLogFormat"]

    prerun.prepare_gateway_config(cfg.model_copy(update={"disable_tracing": True}))
    assert not envoy_path.exists()
    restored = yaml.safe_load(cm_path.read_text())
    assert yaml.safe_load(restored["data"]["mesh"]) == mesh


def test_embedded_gateway_defaults_to_builtin_tempo(embedded_gateway):
    cfg, root, _ = embedded_gateway
    cfg = cfg.model_copy(update={"advertise_address": "192.0.2.10"})

    prerun.prepare_gateway_config(cfg)

    config = yaml.safe_load((root / HIGRESS_FILTER_PATH).read_text())
    cluster = config["spec"]["configPatches"][0]["patch"]["value"]
    endpoints = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"]
    assert endpoints[0]["endpoint"]["address"]["socket_address"] == {
        "address": "192.0.2.10",
        "port_value": 4317,
    }


def test_disabled_collector_removes_stale_config_and_service(tmp_path, collector_path):
    collector_path.parent.mkdir(parents=True)
    collector_path.write_text("stale")
    cfg = Config(data_dir=str(tmp_path / "data"), log_otlp_endpoint=None)

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()
    assert "alloy" not in prerun.determine_enabled_services(cfg)

    s6_base_path = tmp_path / "s6-rc.d"
    prepare_s6_overlay(["alloy"], [], s6_base_path)
    service = s6_base_path / "user/contents.d/alloy"
    assert service.exists()
    prepare_s6_overlay([], [], s6_base_path)
    assert not service.exists()


@pytest.mark.parametrize(
    ("endpoint", "authorization"),
    [
        ("http://user:private@collector/v1/logs", None),
        ("http://collector/v1/logs", "Bearer private"),
        ("https://collector/v1/logs", "Bearer private\nX-Other: invalid"),
        ("grpc://collector/v1/logs", None),
    ],
)
def test_invalid_otlp_settings_skip_collector_without_logging_secrets(
    tmp_path, collector_path, caplog, endpoint, authorization
):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_otlp_endpoint=endpoint,
        log_otlp_authorization=authorization,
    )

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    "patterns, message",
    [
        ('["valid", ""]', "GPUSTACK_LOG_REDACT_PATTERNS[1]"),
        ('["valid", 1]', "GPUSTACK_LOG_REDACT_PATTERNS[1]"),
        ('"secret"', "must be a JSON array"),
        ('["secret"', "must be a JSON array"),
    ],
)
def test_docker_rejects_invalid_redaction_configuration(
    tmp_path, collector_path, caplog, monkeypatch, patterns, message
):
    monkeypatch.setenv("GPUSTACK_LOG_REDACT_PATTERNS", patterns)
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_otlp_endpoint="http://collector/v1/logs",
    )

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()
    assert message in caplog.text
    assert "secret" not in caplog.text


def test_helm_logging_does_not_start_a_container_collector(tmp_path, collector_path):
    cfg = Config(
        data_dir=str(tmp_path / "data"),
        log_otlp_endpoint="http://collector/v1/logs",
        log_pod_release="inference",
    )

    assert not prerun.prepare_log_collection(cfg)
    assert not collector_path.exists()


def test_vllm_pd_logs_keep_native_engine_and_coordinator_ids(collector_path):
    pipeline = prerun.ALLOY_LOG_TEMPLATE.with_name("common.alloy").read_text()
    (coordinator_pattern,) = re.findall(
        r"value = `(\(\?P<pd_coordinator_id>[^`]+)`", pipeline
    )
    patterns = [
        re.compile(json.loads(literal.replace("%s", coordinator_pattern)))
        for literal in re.findall(
            r'ExtractPatterns\(log\.body, ("(?:[^"\\]|\\.)*")\)', pipeline
        )
    ]
    coordinator = (
        "___prefill_addr_127.0.0.1:8000___decode_addr_[::1]:9000_"
        "78651a8e19a445a3a3ce157e1578f1d7"
    )
    for engine_id in (
        f"chatcmpl-{coordinator}-b7a64399",
        f"0_chatcmpl-{coordinator}_0-92309ad8",
    ):
        line = f"(APIServer pid=12) INFO 11-16 08:01:00 Added request {engine_id}."
        assert any(
            (match := pattern.fullmatch(line))
            and match.groupdict()
            == {
                "engine_request_id": engine_id,
                "pd_coordinator_id": coordinator,
            }
            for pattern in patterns
        )

    router = (
        "2026-11-16 08:01:00 DEBUG vllm_router_rs::routers::http::vllm_pd_router: "
        f"src/routers/http/vllm_pd_router.rs:1323: Generated vLLM request ID: {coordinator}"
    )
    assert any(
        (match := pattern.fullmatch(router))
        and match.groupdict() == {"pd_coordinator_id": coordinator}
        for pattern in patterns
    )


@pytest.mark.parametrize("bootstrap_room", [0, 6_721_395_813_317_762_539, None])
def test_sglang_pd_attempt_id_preserves_bootstrap_room(collector_path, bootstrap_room):
    pipeline = prerun.ALLOY_LOG_TEMPLATE.with_name("common.alloy").read_text()
    pattern, replacement = re.search(
        r'replace_pattern\(log\.cache\["engine_json"\], '
        r'("(?:[^"\\]|\\.)*"), ("(?:[^"\\]|\\.)*")\)',
        pipeline,
    ).groups()
    replacement = re.sub(r"\$(\d+)", r"\\g<\1>", json.loads(replacement))
    record = {"obj": {"bootstrap_room": bootstrap_room, "other_field": 1}}
    parsed = json.loads(re.sub(json.loads(pattern), replacement, json.dumps(record)))

    expected = str(bootstrap_room) if bootstrap_room is not None else None
    assert parsed["obj"] == {"bootstrap_room": expected, "other_field": 1}
