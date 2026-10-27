import datetime
import hashlib
import ssl
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gpustack.worker import tls


@pytest.fixture
def tls_material(tmp_path):
    """A private root and its server leaf for in-memory TLS handshakes."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "server.example")])

    def certificate(subject, public_key, is_ca):
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), True)
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("server.example")]), False
            )
            .sign(ca_key, hashes.SHA256())
            .public_bytes(serialization.Encoding.PEM)
        )

    ca = certificate(ca_name, ca_key.public_key(), True)
    leaf = certificate(server_name, key.public_key(), False)
    cert_path = tmp_path / "leaf.pem"
    cert_path.write_bytes(leaf)
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return ca, leaf, context


def _handshake(client_context, server_context):
    client_in, client_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_context.wrap_bio(
        client_in, client_out, server_hostname="server.example"
    )
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    for _ in range(10):
        try:
            client.do_handshake()
            return
        except ssl.SSLWantReadError:
            pass
        server_in.write(client_out.read())
        try:
            server.do_handshake()
        except ssl.SSLWantReadError:
            pass
        client_in.write(server_out.read())
    pytest.fail("TLS handshake did not complete")


@pytest.fixture(autouse=True)
def isolate_tls_state(monkeypatch):
    monkeypatch.setattr(tls, "_merged_ssl_cert_files", {})
    # Record the environment even when the variable was initially absent, since
    # activation sets it directly and the fixture must restore it afterward.
    monkeypatch.setenv("SSL_CERT_FILE", "")
    monkeypatch.delenv("SSL_CERT_FILE")
    factory = tls.make_ssl_context
    factory.cache_clear()
    yield
    for path in list(tls._merged_ssl_cert_files):
        tls._cleanup_merged_ssl_cert_file(path)
    factory.cache_clear()


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
    monkeypatch.setattr(tls, "DEFAULT_CUSTOM_CA_DIR", str(system_trust_directory))

    certificate_path = tls._write_ca_certificate(b"test ca\n")

    assert certificate_path == str(system_trust_directory / tls.SERVER_CA_PATH)
    assert (system_trust_directory / tls.SERVER_CA_PATH).read_bytes() == b"test ca\n"


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
    assert rotated_bundle == merged_bundle
    assert Path(rotated_bundle).read_bytes() == (b"custom ca\nrotated server ca\n")


@pytest.mark.asyncio
async def test_bootstrap_imports_verified_ca(monkeypatch, tls_material):
    bundle = tls_material[0]
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv(
        tls.SERVER_CA_CHECKSUM_ENV, hashlib.sha256(bundle).hexdigest().upper()
    )
    verification = iter([False, True, True])

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
        ("certificate verify failed: self signed certificate", True),
        ("certificate verify failed: unable to get local issuer certificate", True),
        ("certificate verify failed: certificate has expired", False),
        ("certificate verify failed: hostname mismatch", False),
    ],
)
def test_recognizes_only_untrusted_issuer_errors(message, expected):
    assert tls._is_certificate_verification_error(RuntimeError(message)) is expected


@pytest.mark.parametrize("code", [18, 19, 20, 21, 10, 62])
@pytest.mark.asyncio
async def test_can_verify_uses_nested_verification_code(monkeypatch, code):
    cause = ssl.SSLCertVerificationError(1, "localized error")
    cause.verify_code = code
    error = httpx.ConnectError("self signed certificate")
    error.__cause__ = cause
    client = _InsecureClient(None)
    client.get = AsyncMock(side_effect=error)
    monkeypatch.setattr(tls.httpx, "AsyncClient", lambda **_: client)
    context = ssl.create_default_context()

    if code in {18, 19, 20, 21}:
        assert not await tls._can_verify_server("https://server.example", context)
    else:
        with pytest.raises(httpx.ConnectError) as raised:
            await tls._can_verify_server("https://server.example", context)
        assert raised.value is error


@pytest.mark.asyncio
async def test_can_verify_passes_ssl_context_without_deprecation(monkeypatch):
    context = ssl.create_default_context()

    async def get(client, url):
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert await tls._can_verify_server("https://server.example", context)


@pytest.mark.asyncio
async def test_bootstrap_rejects_leaf_as_trust_anchor(monkeypatch, tls_material):
    ca, leaf, server_context = tls_material
    monkeypatch.setenv(tls.SERVER_CA_CHECKSUM_ENV, hashlib.sha256(leaf).hexdigest())
    monkeypatch.setattr(tls, "_download_server_ca_bundle", AsyncMock(return_value=leaf))
    install = Mock()
    monkeypatch.setattr(tls, "_install_ca_certificate", install)

    # Mock the HTTP boundary while preserving the real OpenSSL handshake and
    # exception chain used by httpx to report certificate verification errors.
    class HandshakeClient(_InsecureClient):
        async def get(self, url):
            try:
                _handshake(self.kwargs["verify"], server_context)
            except ssl.SSLCertVerificationError as error:
                raise httpx.ConnectError(str(error)) from error

    monkeypatch.setattr(
        tls.httpx, "AsyncClient", lambda **kwargs: HandshakeClient(None, **kwargs)
    )
    with pytest.raises(RuntimeError, match="not a usable trust anchor"):
        await tls.ensure_server_tls_trust("https://server.example")
    install.assert_not_called()
    # Ensure that the same server is verifiable with its issuing root.
    _handshake(ssl.create_default_context(cadata=ca.decode()), server_context)


def test_install_wraps_directory_permission_error(monkeypatch):
    monkeypatch.setattr(tls.os, "makedirs", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(RuntimeError, match="Failed to import") as raised:
        tls._install_ca_certificate(b"ca")
    assert isinstance(raised.value.__cause__, PermissionError)


@pytest.mark.parametrize("system_available", [True, False])
@pytest.mark.asyncio
async def test_bootstrap_uses_process_bundle_when_system_trust_is_unavailable(
    monkeypatch, tls_material, system_available
):
    ca, _, server_context = tls_material
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    install = Mock(return_value="/ca/server.crt")
    if not system_available:
        install.side_effect = RuntimeError("system store is not writable")
    monkeypatch.setattr(tls, "_install_ca_certificate", install)

    async def can_verify(url, context):
        try:
            _handshake(context, server_context)
            return True
        except ssl.SSLCertVerificationError:
            return False

    monkeypatch.setattr(tls, "_can_verify_server", can_verify)
    await tls._activate_server_ca("https://server.example", ca)

    assert ca in Path(tls.os.environ["SSL_CERT_FILE"]).read_bytes()
    _handshake(tls.make_ssl_context(), server_context)


@pytest.mark.asyncio
async def test_activation_does_not_report_success_when_final_verification_fails(
    monkeypatch, tls_material, caplog
):
    ca, _, _ = tls_material
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        tls, "_install_ca_certificate", Mock(return_value="/ca/server.crt")
    )
    monkeypatch.setattr(tls, "_can_verify_server", AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="TLS verification still failed"):
        await tls._activate_server_ca("https://server.example", ca)
    assert "Imported the server CA" not in caplog.text
