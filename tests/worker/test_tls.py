import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpustack.worker import tls


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def aiter_bytes(self):
        yield self.content


class _InsecureClient:
    def __init__(self, response, **kwargs):
        self.response = response
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        self.url = url
        return self.response

    def stream(self, method, url):
        self.method = method
        self.url = url
        return self.response


def test_ca_certificate_uses_system_trust_directory(monkeypatch, tmp_path):
    system_trust_directory = tmp_path / "system-trust"
    custom_directory = tmp_path / "custom-trust"
    monkeypatch.setattr(tls, "DEFAULT_CUSTOM_CA_DIR", str(system_trust_directory))
    monkeypatch.setenv("GPUSTACK_CUSTOM_CA_DIR", str(custom_directory))

    certificate_path = tls._write_ca_certificate(b"test ca\n")

    assert certificate_path == str(system_trust_directory / tls.SERVER_CA_PATH)
    assert (system_trust_directory / tls.SERVER_CA_PATH).read_bytes() == b"test ca\n"
    assert not custom_directory.exists()


def test_merge_server_ca_preserves_custom_ssl_cert_file(monkeypatch, tmp_path):
    custom_bundle = tmp_path / "custom-ca.pem"
    custom_bundle.write_bytes(b"custom ca")
    monkeypatch.setenv("SSL_CERT_FILE", str(custom_bundle))

    tls._merge_server_ca_into_ssl_cert_file(b"server ca\n")

    merged_bundle = tls.os.environ["SSL_CERT_FILE"]
    assert merged_bundle != str(custom_bundle)
    assert Path(merged_bundle).read_bytes() == b"custom ca\nserver ca\n"

    tls._merge_server_ca_into_ssl_cert_file(b"server ca\n")

    assert tls.os.environ["SSL_CERT_FILE"] == merged_bundle
    assert Path(merged_bundle).exists()

    tls._merge_server_ca_into_ssl_cert_file(b"rotated server ca\n")

    rotated_bundle = tls.os.environ["SSL_CERT_FILE"]
    assert rotated_bundle != merged_bundle
    assert not Path(merged_bundle).exists()
    assert Path(rotated_bundle).read_bytes() == (
        b"custom ca\nserver ca\nrotated server ca\n"
    )


@pytest.mark.asyncio
async def test_bootstrap_imports_verified_ca(monkeypatch):
    bundle = b"test ca\n"
    monkeypatch.setenv(
        tls.SERVER_CA_CHECKSUM_ENV, hashlib.sha256(bundle).hexdigest().upper()
    )
    verification = iter([False, True])

    async def can_verify(*args):
        return next(verification)

    client = _InsecureClient(_Response(bundle))
    monkeypatch.setattr(tls, "_can_verify_server", can_verify)
    ssl_context = Mock(return_value=object())
    ssl_context.cache_clear = Mock()
    monkeypatch.setattr(tls, "make_ssl_context", ssl_context)

    def client_factory(**kwargs):
        client.kwargs = kwargs
        return client

    monkeypatch.setattr(tls.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(tls, "_write_ca_certificate", lambda _: "/ca/server.crt")
    monkeypatch.setattr(tls, "use_proxy_env_for_url", lambda _: False)
    run = Mock()
    monkeypatch.setattr(tls.subprocess, "run", run)

    await tls.ensure_server_tls_trust("https://server.example")

    assert client.method == "GET"
    assert client.url == "https://server.example/v2/cacerts"
    assert client.kwargs["trust_env"] is False
    run.assert_called_once_with(
        ["update-ca-certificates"], check=True, capture_output=True
    )


@pytest.mark.asyncio
async def test_bootstrap_skips_ca_download_when_server_is_already_trusted(monkeypatch):
    monkeypatch.setenv(tls.SERVER_CA_CHECKSUM_ENV, "a" * 64)

    async def can_verify(*args):
        return True

    monkeypatch.setattr(tls, "_can_verify_server", can_verify)
    monkeypatch.setattr(tls, "make_ssl_context", lambda: object())

    await tls.ensure_server_tls_trust("https://server.example")


@pytest.mark.asyncio
async def test_bootstrap_preserves_server_connection_error(monkeypatch):
    monkeypatch.setenv(tls.SERVER_CA_CHECKSUM_ENV, "a" * 64)

    async def can_verify(*args):
        raise tls.httpx.ConnectError("connection refused")

    monkeypatch.setattr(tls, "_can_verify_server", can_verify)
    monkeypatch.setattr(tls, "make_ssl_context", lambda: object())

    with pytest.raises(tls.httpx.ConnectError, match="connection refused"):
        await tls.ensure_server_tls_trust("https://server.example")


@pytest.mark.asyncio
async def test_bootstrap_rejects_ca_checksum_mismatch(monkeypatch):
    monkeypatch.setenv(tls.SERVER_CA_CHECKSUM_ENV, "a" * 64)

    async def can_verify(*args):
        return False

    monkeypatch.setattr(tls, "_can_verify_server", can_verify)
    monkeypatch.setattr(tls, "make_ssl_context", lambda: object())
    monkeypatch.setattr(
        tls.httpx, "AsyncClient", lambda **kwargs: _InsecureClient(_Response(b"ca"))
    )

    with pytest.raises(RuntimeError, match="does not match the checksum"):
        await tls.ensure_server_tls_trust("https://server.example")


@pytest.mark.asyncio
async def test_bootstrap_rejects_oversized_ca_bundle(monkeypatch):
    monkeypatch.setattr(tls, "MAX_SERVER_CA_BUNDLE_BYTES", 3)
    monkeypatch.setattr(tls, "use_proxy_env_for_url", lambda _: False)
    monkeypatch.setattr(
        tls.httpx, "AsyncClient", lambda **kwargs: _InsecureClient(_Response(b"test"))
    )

    with pytest.raises(RuntimeError, match="exceeds the maximum size"):
        await tls._download_server_ca_bundle("https://server.example/v2/cacerts")


def test_install_ca_certificate_removes_file_when_trust_update_fails(
    monkeypatch, tmp_path
):
    certificate_path = tmp_path / "server.crt"
    certificate_path.write_bytes(b"test ca\n")
    monkeypatch.setattr(tls, "_write_ca_certificate", lambda _: str(certificate_path))
    monkeypatch.setattr(
        tls.subprocess,
        "run",
        Mock(side_effect=FileNotFoundError("update-ca-certificates")),
    )

    with pytest.raises(RuntimeError, match="Failed to import"):
        tls._install_ca_certificate(b"test ca\n")

    assert not certificate_path.exists()


def test_install_ca_certificate_includes_trust_update_stderr(monkeypatch, tmp_path):
    certificate_path = tmp_path / "server.crt"
    certificate_path.write_bytes(b"test ca\n")
    monkeypatch.setattr(tls, "_write_ca_certificate", lambda _: str(certificate_path))
    monkeypatch.setattr(
        tls.subprocess,
        "run",
        Mock(
            side_effect=tls.subprocess.CalledProcessError(
                1, "update-ca-certificates", stderr=b"invalid certificate"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="invalid certificate"):
        tls._install_ca_certificate(b"test ca\n")

    assert not certificate_path.exists()


@pytest.mark.asyncio
async def test_bootstrap_skips_http_server(monkeypatch):
    monkeypatch.setenv(tls.SERVER_CA_CHECKSUM_ENV, "a" * 64)

    await tls.ensure_server_tls_trust("http://server.example")


@pytest.mark.parametrize(
    "message, expected",
    [
        ("certificate verify failed: self-signed certificate", True),
        ("certificate verify failed: unable to get local issuer certificate", True),
        ("certificate verify failed: certificate has expired", False),
        ("certificate verify failed: hostname mismatch", False),
    ],
)
def test_recognizes_only_untrusted_issuer_errors(message, expected):
    assert tls._is_certificate_verification_error(RuntimeError(message)) is expected
