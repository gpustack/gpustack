from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gpustack.routes import cacerts


def test_ca_certificates_route_is_public_but_hidden_from_schema():
    from gpustack.routes.routes import api_router

    route = next(route for route in api_router.routes if route.path == "/v2/cacerts")
    assert route.include_in_schema is False
    assert not route.dependencies


def test_get_ca_certificates_returns_explicit_ca_bundle(monkeypatch):
    monkeypatch.setattr(
        cacerts,
        "get_global_config",
        lambda: SimpleNamespace(ssl_ca_certfile="ca.pem", ssl_certfile=None),
    )
    monkeypatch.setattr(cacerts, "read_server_ca_bundle", lambda *_: b"test ca\n")

    response = cacerts.get_ca_certificates()

    assert response.body == b"test ca\n"
    assert response.media_type == "application/x-pem-file"
    assert response.headers["cache-control"] == "no-store"


def test_get_ca_certificates_falls_back_to_server_certificate(monkeypatch):
    monkeypatch.setattr(
        cacerts,
        "get_global_config",
        lambda: SimpleNamespace(ssl_ca_certfile=None, ssl_certfile="cert.pem"),
    )
    monkeypatch.setattr(cacerts, "read_server_ca_bundle", lambda *_: b"cert\n")

    response = cacerts.get_ca_certificates()

    assert response.body == b"cert\n"


def test_get_ca_certificates_is_unavailable_without_global_config(monkeypatch):
    monkeypatch.setattr(cacerts, "get_global_config", lambda: None)

    with pytest.raises(HTTPException) as error:
        cacerts.get_ca_certificates()

    assert error.value.status_code == 404


def test_get_ca_certificates_is_unavailable_without_tls_files(monkeypatch):
    monkeypatch.setattr(
        cacerts,
        "get_global_config",
        lambda: SimpleNamespace(ssl_ca_certfile=None, ssl_certfile=None),
    )

    with pytest.raises(HTTPException) as error:
        cacerts.get_ca_certificates()

    assert error.value.status_code == 404


def test_get_ca_certificates_is_unavailable_when_certificate_file_is_unreadable(
    monkeypatch,
):
    monkeypatch.setattr(
        cacerts,
        "get_global_config",
        lambda: SimpleNamespace(ssl_ca_certfile="ca.pem", ssl_certfile=None),
    )
    monkeypatch.setattr(
        cacerts,
        "read_server_ca_bundle",
        lambda *_: (_ for _ in ()).throw(OSError("permission denied")),
    )

    with pytest.raises(HTTPException) as error:
        cacerts.get_ca_certificates()

    assert error.value.status_code == 500
    assert isinstance(error.value.__cause__, OSError)


def test_get_ca_certificates_is_unavailable_when_certificate_file_is_empty(
    monkeypatch, tmp_path
):
    certificate = tmp_path / "empty.pem"
    certificate.write_bytes(b"")
    monkeypatch.setattr(
        cacerts,
        "get_global_config",
        lambda: SimpleNamespace(ssl_ca_certfile=str(certificate), ssl_certfile=None),
    )

    with pytest.raises(HTTPException) as error:
        cacerts.get_ca_certificates()

    assert error.value.status_code == 500
    assert isinstance(error.value.__cause__, ValueError)
