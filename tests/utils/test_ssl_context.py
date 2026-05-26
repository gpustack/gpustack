import datetime
import http.server
import socket
import ssl
import threading
from types import SimpleNamespace
from unittest import mock

import certifi
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gpustack import ssl_context
from gpustack.ssl_context import make_ssl_context, resolve_ca_bundle


def _clear_caches():
    make_ssl_context.cache_clear()
    ssl_context._warn_unsupported_ssl_env_vars.cache_clear()


def test_resolve_ca_bundle_prefers_ssl_cert_file_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom-bundle.pem"
    custom.write_text("dummy")
    monkeypatch.setenv("SSL_CERT_FILE", str(custom))

    assert resolve_ca_bundle() == str(custom)


def test_resolve_ca_bundle_falls_back_when_env_file_missing(
    monkeypatch, tmp_path, caplog
):
    """``SSL_CERT_FILE`` pointing at a non-existent path must not poison
    verification -- fall through to the next candidate, and emit a warning
    so operators notice the misconfiguration instead of silently running
    against an unintended trust anchor.
    """
    ssl_context._warn_unsupported_ssl_env_vars.cache_clear()
    missing = tmp_path / "does-not-exist.pem"
    monkeypatch.setenv("SSL_CERT_FILE", str(missing))

    distro_bundle = tmp_path / "distro-bundle.pem"
    distro_bundle.write_text("dummy")
    try:
        with mock.patch.object(
            ssl,
            "get_default_verify_paths",
            return_value=SimpleNamespace(cafile=str(distro_bundle)),
        ):
            with caplog.at_level("WARNING", logger="gpustack.ssl_context"):
                assert resolve_ca_bundle() == str(distro_bundle)
    finally:
        ssl_context._warn_unsupported_ssl_env_vars.cache_clear()

    assert any(
        "SSL_CERT_FILE" in record.message and str(missing) in record.message
        for record in caplog.records
    ), "expected a warning naming the missing SSL_CERT_FILE path"


def test_resolve_ca_bundle_warns_when_ssl_cert_dir_set(monkeypatch, tmp_path, caplog):
    """``SSL_CERT_DIR`` is intentionally ignored (would route through the
    hash_dir accumulation path). Operators who set it should see a clear
    warning instead of silently running with an unintended trust anchor.
    """
    ssl_context._warn_unsupported_ssl_env_vars.cache_clear()
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))

    with caplog.at_level("WARNING", logger="gpustack.ssl_context"):
        try:
            resolve_ca_bundle()
        finally:
            ssl_context._warn_unsupported_ssl_env_vars.cache_clear()

    assert any(
        "SSL_CERT_DIR" in record.message and "ignored" in record.message
        for record in caplog.records
    ), "expected a warning explaining SSL_CERT_DIR is ignored"


def test_resolve_ca_bundle_falls_back_to_certifi(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    with mock.patch.object(
        ssl,
        "get_default_verify_paths",
        return_value=SimpleNamespace(cafile=None),
    ):
        assert resolve_ca_bundle() == certifi.where()


def test_make_ssl_context_is_singleton(monkeypatch):
    """Two calls return the *same* context object so that one bundle load
    is shared across every httpx client / auth flow in the process.
    """
    _clear_caches()
    monkeypatch.setattr(ssl_context, "resolve_ca_bundle", lambda: certifi.where())
    try:
        first = make_ssl_context()
        second = make_ssl_context()
        assert first is second
        assert isinstance(first, ssl.SSLContext)
    finally:
        _clear_caches()


# ---------------------------------------------------------------------------
# Functional tests: a real TLS handshake against a self-signed cert. These
# are the load-bearing ones -- they confirm the new context actually verifies
# certificates correctly, not just that it points at the right file.
# ---------------------------------------------------------------------------


def _make_self_signed_cert(tmp_path, hostname="localhost"):
    """Generate a self-signed cert + key on disk; return (cert_path, key_path)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture
def tls_server(tmp_path):
    """Start a tiny TLS server in a background thread; yield (port, ca_path).

    The CA bundle is just the server's self-signed cert -- pointing
    SSL_CERT_FILE at it should let a client verify the server, and not
    pointing at it should make verification fail.
    """
    cert_path, key_path = _make_self_signed_cert(tmp_path)

    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(8)

    httpd = http.server.HTTPServer.__new__(http.server.HTTPServer)
    # Manually configure without re-binding (we already have the socket).
    http.server.HTTPServer.__init__(
        httpd,
        ("127.0.0.1", port),
        http.server.SimpleHTTPRequestHandler,
        bind_and_activate=False,
    )
    httpd.socket = server_ctx.wrap_socket(sock, server_side=True)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, cert_path
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_make_ssl_context_verifies_against_custom_bundle(monkeypatch, tls_server):
    """SSL_CERT_FILE pointing at our self-signed cert → handshake succeeds."""
    port, cert_path = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    make_ssl_context.cache_clear()

    ctx = make_ssl_context()
    try:
        # Use httpx to drive a real request through the new context.
        with httpx.Client(verify=ctx, timeout=5) as client:
            # Server cert SAN=localhost; resolve via localhost directly and
            # pin sni_hostname so SNI/SAN verification lines up regardless of
            # the runner's IPv4/IPv6 ordering for "localhost".
            resp = client.get(
                f"https://localhost:{port}/",
                extensions={"sni_hostname": "localhost"},
            )
        assert resp.status_code in (200, 404)  # any valid HTTP response is fine
    finally:
        make_ssl_context.cache_clear()


def test_make_ssl_context_rejects_untrusted_cert(monkeypatch, tls_server, tmp_path):
    """Bundle that does NOT contain the server's cert → verification fails."""
    port, _cert_path = tls_server
    # Point SSL_CERT_FILE at certifi (which obviously doesn't contain our
    # ephemeral self-signed cert).
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    make_ssl_context.cache_clear()

    ctx = make_ssl_context()
    try:
        with httpx.Client(verify=ctx, timeout=5) as client:
            with pytest.raises(httpx.ConnectError) as exc_info:
                client.get(f"https://localhost:{port}/")
        # The error chain should contain a TLS cert verification failure.
        assert (
            "CERTIFICATE_VERIFY_FAILED" in str(exc_info.value)
            or "certificate verify failed" in str(exc_info.value).lower()
        )
    finally:
        make_ssl_context.cache_clear()


def test_make_ssl_context_loads_nonempty_ca_bundle(monkeypatch):
    """Sanity check: the resolved context comes with some CA roots loaded.

    Catches the regression where the resolved bundle file exists but is
    empty / malformed, which would silently accept everything or reject
    everything depending on OpenSSL's mood.

    Pin the resolution to ``certifi.where()`` so the assertion does not
    depend on the runner's distro CA bundle (some minimal CI images ship
    an empty ``/etc/ssl/certs/ca-certificates.crt``).
    """
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    make_ssl_context.cache_clear()
    try:
        ctx = make_ssl_context()
        # binary_form=False returns parsed dicts; we just want >0.
        assert len(ctx.get_ca_certs()) > 0
    finally:
        make_ssl_context.cache_clear()
