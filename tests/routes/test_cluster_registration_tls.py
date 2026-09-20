import hashlib
from types import SimpleNamespace

import pytest

from gpustack.routes import clusters


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
