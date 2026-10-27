import datetime
import hashlib
import os
from unittest.mock import Mock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gpustack.utils.certificates import read_server_ca_bundle, server_ca_checksum
from gpustack.utils import certificates


@pytest.fixture
def certificate():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(minutes=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


def test_read_server_ca_bundle_prefers_explicit_ca_file(tmp_path, certificate):
    cert = tmp_path / "server.pem"
    cert.write_bytes(certificate)
    ca = tmp_path / "ca.pem"
    ca.write_bytes(certificate)

    bundle = read_server_ca_bundle(str(ca), str(cert))

    assert bundle == certificate
    assert server_ca_checksum(bundle) == hashlib.sha256(bundle).hexdigest()


def test_read_server_ca_bundle_uses_server_certificate_as_fallback(
    tmp_path, certificate
):
    cert = tmp_path / "server.pem"
    cert.write_bytes(certificate)

    assert read_server_ca_bundle(None, str(cert)) == certificate


def test_read_server_ca_bundle_returns_only_certificate_blocks(tmp_path, certificate):
    cert = tmp_path / "combined.pem"
    cert.write_bytes(
        certificate
        + b"-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n"
        + b"operator note\n"
    )

    assert read_server_ca_bundle(str(cert), None) == certificate


@pytest.mark.parametrize("contents", [b"", b"not a certificate"])
def test_read_server_ca_bundle_rejects_missing_certificates(tmp_path, contents):
    cert = tmp_path / "invalid.pem"
    cert.write_bytes(contents)

    with pytest.raises(ValueError, match="contains no certificates"):
        read_server_ca_bundle(str(cert), None)


def test_read_server_ca_bundle_rejects_invalid_certificate(tmp_path):
    cert = tmp_path / "invalid.pem"
    cert.write_bytes(
        b"-----BEGIN CERTIFICATE-----\n" b"dGVzdA==\n" b"-----END CERTIFICATE-----\n"
    )

    with pytest.raises(ValueError, match="contains an invalid certificate"):
        read_server_ca_bundle(str(cert), None)


def test_read_server_ca_bundle_returns_none_without_tls_files():
    assert read_server_ca_bundle(None, None) is None
    assert server_ca_checksum(None) is None


@pytest.mark.parametrize("replace_file", [True, False])
def test_ca_bundle_cache_invalidates_when_file_changes(
    monkeypatch, tmp_path, certificate, replace_file
):
    path = tmp_path / "ca.pem"
    path.write_bytes(certificate)
    extract = Mock(wraps=certificates._extract_certificates)
    monkeypatch.setattr(certificates, "_extract_certificates", extract)
    assert read_server_ca_bundle(str(path), None) == certificate
    assert read_server_ca_bundle(str(path), None) == certificate
    assert extract.call_count == 1

    if replace_file:
        replacement = tmp_path / "replacement.pem"
        replacement.write_bytes(certificate * 2)
        os.replace(replacement, path)
    else:
        path.write_bytes(certificate * 2)
    assert read_server_ca_bundle(str(path), None) == certificate * 2
    assert extract.call_count == 2

    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_server_ca_bundle(str(path), None)
