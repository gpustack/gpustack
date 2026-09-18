"""Bootstrap TLS trust for workers joining a private-CA server."""

import asyncio
import atexit
import hashlib
import logging
import os
import subprocess
import tempfile
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from gpustack.ssl_context import make_ssl_context
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)

SERVER_CA_CHECKSUM_ENV = "GPUSTACK_SERVER_CA_CERT_SHA256"
SERVER_CA_PATH = "gpustack-server-ca.crt"
DEFAULT_CUSTOM_CA_DIR = "/usr/local/share/ca-certificates"
MAX_SERVER_CA_BUNDLE_BYTES = 1024 * 1024
MERGED_SSL_CERT_FILE_PREFIX = "gpustack-server-ca-bundle-"
_merged_ssl_cert_files = set()


def _server_endpoint(server_url: str, path: str) -> str:
    return urljoin(f"{server_url.rstrip('/')}/", path.lstrip("/"))


def _is_certificate_verification_error(error: BaseException) -> bool:
    """Whether a transport error means the server's issuer is untrusted."""
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if (
            "self-signed certificate" in message
            or "unknown ca" in message
            or "unable to get local issuer certificate" in message
            or "unable to verify the first certificate" in message
            or "certificate signed by unknown authority" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _can_verify_server(server_url: str, verify) -> bool:
    try:
        async with httpx.AsyncClient(
            verify=verify,
            timeout=5,
            trust_env=use_proxy_env_for_url(server_url),
        ) as client:
            await client.get(server_url)
        return True
    except httpx.TransportError as error:
        if _is_certificate_verification_error(error):
            return False
        raise


def _write_ca_certificate(bundle: bytes) -> str:
    os.makedirs(DEFAULT_CUSTOM_CA_DIR, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".gpustack-server-ca-", suffix=".crt", dir=DEFAULT_CUSTOM_CA_DIR
    )
    try:
        with os.fdopen(fd, "wb") as certfile:
            certfile.write(bundle)
        target_path = os.path.join(DEFAULT_CUSTOM_CA_DIR, SERVER_CA_PATH)
        os.replace(temporary_path, target_path)
    except Exception:
        os.unlink(temporary_path)
        raise
    return target_path


def _write_temporary_ca_certificate(bundle: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="gpustack-server-ca-", suffix=".crt")
    with os.fdopen(fd, "wb") as certfile:
        certfile.write(bundle)
    return path


def _cleanup_merged_ssl_cert_file(path: str) -> None:
    _merged_ssl_cert_files.discard(path)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _merge_server_ca_into_ssl_cert_file(bundle: bytes) -> None:
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if not ssl_cert_file or not os.path.isfile(ssl_cert_file):
        return

    with open(ssl_cert_file, "rb") as certfile:
        existing_bundle = certfile.read()
    if bundle in existing_bundle:
        return
    if not existing_bundle.endswith(b"\n"):
        existing_bundle += b"\n"

    fd, merged_path = tempfile.mkstemp(
        prefix=MERGED_SSL_CERT_FILE_PREFIX, suffix=".pem"
    )
    with os.fdopen(fd, "wb") as certfile:
        certfile.write(existing_bundle)
        certfile.write(bundle)
    os.environ["SSL_CERT_FILE"] = merged_path
    _merged_ssl_cert_files.add(merged_path)
    atexit.register(_cleanup_merged_ssl_cert_file, merged_path)
    if ssl_cert_file in _merged_ssl_cert_files:
        _cleanup_merged_ssl_cert_file(ssl_cert_file)


def _install_ca_certificate(bundle: bytes) -> str:
    target_path = _write_ca_certificate(bundle)
    try:
        subprocess.run(["update-ca-certificates"], check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        try:
            os.unlink(target_path)
        except FileNotFoundError:
            pass
        stderr = error.stderr.decode(errors="replace").strip() if error.stderr else ""
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(
            f"Failed to import the server CA certificate at {target_path}: {error}{detail}"
        ) from error
    except OSError as error:
        try:
            os.unlink(target_path)
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Failed to import the server CA certificate at {target_path}: {error}"
        ) from error
    return target_path


async def _download_server_ca_bundle(ca_url: str) -> bytes:
    chunks = []
    total = 0
    try:
        async with httpx.AsyncClient(
            verify=False,
            timeout=5,
            follow_redirects=False,
            trust_env=use_proxy_env_for_url(ca_url),
        ) as client:
            async with client.stream("GET", ca_url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_SERVER_CA_BUNDLE_BYTES:
                        raise RuntimeError(
                            "The server CA certificate exceeds the maximum "
                            f"size of {MAX_SERVER_CA_BUNDLE_BYTES} bytes."
                        )
                    chunks.append(chunk)
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Failed to download the server CA certificate from {ca_url}: {error}"
        ) from error
    return b"".join(chunks)


async def ensure_server_tls_trust(server_url: str) -> None:
    """Install a pinned CA bundle only when normal TLS cannot trust the server.

    The checksum arrives in the registration command, outside the unverified
    connection used to retrieve the public CA bundle. A public-CA server never
    reaches the bootstrap path because its regular TLS handshake succeeds.
    """
    checksum = os.environ.get(SERVER_CA_CHECKSUM_ENV)
    if not checksum:
        return
    checksum = checksum.lower()
    if urlparse(server_url).scheme != "https":
        return
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError(f"{SERVER_CA_CHECKSUM_ENV} must be a SHA-256 hex digest")

    ssl_context = await asyncio.to_thread(make_ssl_context)
    if await _can_verify_server(server_url, ssl_context):
        return

    ca_url = _server_endpoint(server_url, "/v2/cacerts")
    bundle = await _download_server_ca_bundle(ca_url)
    actual_checksum = hashlib.sha256(bundle).hexdigest()
    if actual_checksum != checksum:
        raise RuntimeError(
            "The downloaded server CA certificate does not match the checksum "
            "in the registration command."
        )

    temporary_path = await asyncio.to_thread(_write_temporary_ca_certificate, bundle)
    try:
        verified = await _can_verify_server(server_url, temporary_path)
        if not verified:
            raise RuntimeError(
                "The downloaded server certificate is not a usable trust anchor. "
                "Configure the server --ssl-ca-certfile with the issuing CA bundle."
            )
    finally:
        await asyncio.to_thread(os.unlink, temporary_path)

    await asyncio.to_thread(_install_ca_certificate, bundle)
    await asyncio.to_thread(_merge_server_ca_into_ssl_cert_file, bundle)
    make_ssl_context.cache_clear()
    logger.info("Imported the server CA certificate for worker registration.")
