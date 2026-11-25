from pathlib import Path
import json
import re

import pytest
import yaml

from gpustack.cmd import prerun
from gpustack.cmd.prerun import check_ports_availability, prepare_s6_overlay
from gpustack.config.config import Config

HIGRESS_CONFIGMAP_PATH = Path("configmaps/higress-config.yaml")


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
