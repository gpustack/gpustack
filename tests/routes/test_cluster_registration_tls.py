import hashlib
from types import SimpleNamespace

import pytest

from gpustack import envs
from gpustack.config.config import Config
from gpustack.routes import clusters
from gpustack.schemas.clusters import SensitiveRegistrationConfig
from gpustack.schemas.config import (
    PredefinedConfigNoDefaults,
    parse_base_model_to_env_vars,
)


def _registration(server_url, ssl_ca_certfile=None, ssl_certfile=None):
    cluster = SimpleNamespace(
        worker_config=None,
        registration_token="registration-token",
        server_url=None,
        system_default_container_registry=None,
    )
    request = SimpleNamespace(
        url=SimpleNamespace(scheme=server_url.split(":", 1)[0], netloc="server")
    )
    config = SimpleNamespace(
        server_external_url=server_url,
        ssl_ca_certfile=ssl_ca_certfile,
        ssl_certfile=ssl_certfile,
        effective_trace_otlp_endpoint=None,
    )
    return request, cluster, config


def test_registration_includes_ca_checksum_for_https(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"test ca\n")
    request, cluster, config = _registration("https://server", ssl_ca_certfile=str(ca))
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(clusters, "read_server_ca_bundle", lambda *_: b"test ca\n")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert (
        registration.env["GPUSTACK_SERVER_CA_CERT_SHA256"]
        == hashlib.sha256(b"test ca\n").hexdigest()
    )


def test_registration_omits_ca_checksum_for_http(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"test ca\n")
    request, cluster, config = _registration("http://server", ssl_ca_certfile=str(ca))
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert "GPUSTACK_SERVER_CA_CERT_SHA256" not in registration.env


def test_registration_omits_ca_checksum_for_cluster_server_override(monkeypatch):
    request, cluster, config = _registration("https://server")
    cluster.server_url = "https://proxy.example"
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(
        clusters,
        "read_server_ca_bundle",
        lambda *_: pytest.fail("must not read the server CA for an override"),
    )

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert "GPUSTACK_SERVER_CA_CERT_SHA256" not in registration.env


def test_registration_includes_ca_checksum_for_matching_cluster_server_override(
    monkeypatch,
):
    request, cluster, config = _registration("https://server")
    cluster.server_url = "https://server"
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(clusters, "read_server_ca_bundle", lambda *_: b"test ca\n")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert (
        registration.env["GPUSTACK_SERVER_CA_CERT_SHA256"]
        == hashlib.sha256(b"test ca\n").hexdigest()
    )


def test_registration_rejects_unreadable_ca_bundle(monkeypatch, caplog):
    request, cluster, config = _registration("https://server", ssl_ca_certfile="ca.pem")
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(
        clusters,
        "read_server_ca_bundle",
        lambda *_: (_ for _ in ()).throw(OSError("permission denied")),
    )

    with pytest.raises(clusters.InternalServerErrorException):
        clusters.get_registration_from_cluster(request, cluster)

    assert "Failed to read the server CA bundle" in caplog.text


def test_registration_sets_insecure_tls_env_from_worker_config(monkeypatch, tmp_path):
    """The cluster's ``insecure_tls`` must reach the worker as
    ``GPUSTACK_INSECURE_TLS`` in the registration command -- the only
    channel read before the worker's first TLS handshake to the server."""
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"test ca\n")
    request, cluster, config = _registration("https://server", ssl_ca_certfile=str(ca))
    cluster.worker_config = PredefinedConfigNoDefaults(insecure_tls=True)
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(clusters, "read_server_ca_bundle", lambda *_: b"test ca\n")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert registration.env[envs.INSECURE_TLS_ENV] == "true"
    # Verification is skipped, so the CA bootstrap checksum is not shipped.
    assert "GPUSTACK_SERVER_CA_CERT_SHA256" not in registration.env


def test_registration_omits_insecure_tls_env_when_unset(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"test ca\n")
    request, cluster, config = _registration("https://server", ssl_ca_certfile=str(ca))
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")
    monkeypatch.setattr(clusters, "read_server_ca_bundle", lambda *_: b"test ca\n")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert envs.INSECURE_TLS_ENV not in registration.env


def test_registration_env_var_name_matches_the_runtime_reader():
    """The env var is derived from the field name; ``gpustack.envs`` reads it
    by its own constant. A rename on either side would silently stop
    skipping verification, so pin the linkage."""
    env = parse_base_model_to_env_vars(
        SensitiveRegistrationConfig(token="token", insecure_tls=True)
    )
    assert env[envs.INSECURE_TLS_ENV] == "true"


@pytest.mark.parametrize(
    "server_url,docker,options,expected",
    [
        ("http://server:30080", True, {}, "server:4317"),
        ("http://[2001:db8::1]:30080", True, {}, "[2001:db8::1]:4317"),
        (
            "http://server:30080",
            True,
            {"trace_advertise_endpoint": "remote-tempo:14431"},
            "remote-tempo:14431",
        ),
        (
            "http://server:30080",
            True,
            {"trace_otlp_endpoint": "collector:4317"},
            "collector:4317",
        ),
        (
            "http://server:30080",
            False,
            {"trace_otlp_endpoint": "gpustack-tempo.test.svc.cluster.local:4317"},
            "gpustack-tempo.test.svc.cluster.local:4317",
        ),
        (
            "http://server:30080",
            True,
            {
                "disable_tracing": True,
                "trace_otlp_endpoint": "collector:4317",
                "trace_advertise_endpoint": "remote-tempo:14431",
            },
            None,
        ),
        ("http://server$(example):30080", True, {}, None),
        ("http://server:30080", False, {}, None),
    ],
)
def test_registration_trace_endpoint(
    monkeypatch, tmp_path, server_url, docker, options, expected
):
    monkeypatch.delenv("GPUSTACK_RUN_DIR", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    if docker:
        monkeypatch.setenv("GPUSTACK_RUN_DIR", str(tmp_path / "run"))
    request, cluster, _ = _registration(server_url)
    cluster.server_url = server_url
    config = Config(data_dir=str(tmp_path), advertise_address="172.19.0.2", **options)
    monkeypatch.setattr(clusters, "get_global_config", lambda: config)
    monkeypatch.setattr(clusters, "get_cluster_image_name", lambda *_: "image")

    registration = clusters.get_registration_from_cluster(request, cluster)

    assert registration.env.get("GPUSTACK_TRACE_OTLP_ENDPOINT") == expected
