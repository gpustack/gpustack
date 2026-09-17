import hashlib
import re
from typing import Optional

from cryptography import x509


_CERTIFICATE_PATTERN = re.compile(
    rb"-----BEGIN CERTIFICATE-----\r?\n"
    rb"(?:[A-Za-z0-9+/=\r\n])+"
    rb"-----END CERTIFICATE-----"
)


def _extract_certificates(data: bytes, path: str) -> bytes:
    certificates = []
    for match in _CERTIFICATE_PATTERN.finditer(data):
        certificate = match.group(0) + b"\n"
        try:
            x509.load_pem_x509_certificate(certificate)
        except ValueError as error:
            raise ValueError(
                f"Server CA certificate file {path} contains an invalid certificate."
            ) from error
        certificates.append(certificate)

    if not certificates:
        raise ValueError(f"Server CA certificate file {path} contains no certificates.")
    return b"".join(certificates)


def read_server_ca_bundle(
    ssl_ca_certfile: Optional[str], ssl_certfile: Optional[str]
) -> Optional[bytes]:
    """Read the CA bundle workers use to verify the server.

    An explicit CA bundle is needed for certificates signed by a private CA.
    A server certificate is a valid trust anchor only when it is self-signed,
    but remains a useful compatibility fallback: the worker verifies it before
    installing it and reports a clear error when it cannot establish a chain.
    """
    path = ssl_ca_certfile or ssl_certfile
    if not path:
        return None

    with open(path, "rb") as certfile:
        bundle = certfile.read()

    return _extract_certificates(bundle, path)


def server_ca_checksum(bundle: Optional[bytes]) -> Optional[str]:
    """Return the SHA-256 checksum for a canonical PEM bundle."""
    if bundle is None:
        return None
    return hashlib.sha256(bundle).hexdigest()
