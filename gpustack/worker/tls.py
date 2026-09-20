"""Bootstrap TLS trust for workers joining a private-CA server."""

import asyncio
import atexit
import hashlib
import logging
import os
import ssl
import subprocess
import tempfile
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from gpustack.ssl_context import make_ssl_context, resolve_ca_bundle
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)

SERVER_CA_CHECKSUM_ENV = "GPUSTACK_SERVER_CA_CERT_SHA256"
SERVER_CA_PATH = "gpustack-server-ca.crt"
DEFAULT_CUSTOM_CA_DIR = "/usr/local/share/ca-certificates"
MAX_SERVER_CA_BUNDLE_BYTES = 1024 * 1024
MERGED_SSL_CERT_FILE_PREFIX = "gpustack-server-ca-bundle-"
_merged_ssl_cert_files = {}


def _server_endpoint(server_url: str, path: str) -> str:
    return urljoin(f"{server_url.rstrip('/')}/", path.lstrip("/"))


def _is_certificate_verification_error(error: BaseException) -> bool:
    """Whether a transport error means the server's issuer is untrusted."""
    seen = set()
    messages = []
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            code = getattr(current, "verify_code", None)
            if code is not None:
                return code in {18, 19, 20, 21}
        messages.append(str(current).lower().replace("self-signed", "self signed"))
        current = current.__cause__ or current.__context__
    for message in messages:
        if (
            "self signed certificate" in message
            or "unknown ca" in message
            or "unable to get local issuer certificate" in message
            or "unable to verify the first certificate" in message
            or "certificate signed by unknown authority" in message
        ):
            return True
    return False


async def _can_verify_server(server_url: str, verify: ssl.SSLContext) -> bool:
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
    _merged_ssl_cert_files.pop(path, None)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _merge_server_ca_into_ssl_cert_file(bundle: bytes) -> str:
    ssl_cert_file = resolve_ca_bundle()
    existing_bundle = _merged_ssl_cert_files.get(ssl_cert_file)
    if existing_bundle is None:
        with open(ssl_cert_file, "rb") as certfile:
            existing_bundle = certfile.read()
    if not existing_bundle.endswith(b"\n"):
        existing_bundle += b"\n"
    merged_bundle = existing_bundle
    if bundle not in existing_bundle:
        merged_bundle += bundle

    # Always rebuild from the operator's bundle so rotated bootstrap CAs do not
    # accumulate. Reuse our path to keep long-lived subprocess environments valid.
    managed = ssl_cert_file in _merged_ssl_cert_files
    if managed:
        with open(ssl_cert_file, "rb") as certfile:
            if certfile.read() == merged_bundle:
                return ssl_cert_file
    fd, merged_path = tempfile.mkstemp(
        prefix=MERGED_SSL_CERT_FILE_PREFIX,
        suffix=".pem",
        dir=os.path.dirname(ssl_cert_file) if managed else None,
    )
    try:
        with os.fdopen(fd, "wb") as certfile:
            certfile.write(merged_bundle)
        if managed:
            os.replace(merged_path, ssl_cert_file)
            return ssl_cert_file
    except Exception:
        os.unlink(merged_path)
        raise
    _merged_ssl_cert_files[merged_path] = existing_bundle
    atexit.register(_cleanup_merged_ssl_cert_file, merged_path)
    os.environ["SSL_CERT_FILE"] = merged_path
    return merged_path


def _install_ca_certificate(bundle: bytes) -> str:
    target_path = None
    try:
        target_path = _write_ca_certificate(bundle)
        subprocess.run(["update-ca-certificates"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as error:
        if target_path:
            try:
                os.unlink(target_path)
            except OSError:
                logger.debug("Unable to remove server CA file %s", target_path)
        stderr = getattr(error, "stderr", None)
        detail = f": {stderr.decode(errors='replace').strip()}" if stderr else ""
        raise RuntimeError(
            f"Failed to import the server CA certificate into the system trust "
            f"store: {error}{detail}"
        ) from error
    return target_path


async def _activate_server_ca(server_url: str, bundle: bytes) -> None:
    try:
        target_path = await asyncio.to_thread(_install_ca_certificate, bundle)
        logger.debug("Installed the server CA certificate at %s", target_path)
    except RuntimeError as error:
        logger.warning("%s; using a process CA bundle instead.", error)
        await asyncio.to_thread(_merge_server_ca_into_ssl_cert_file, bundle)
    else:
        if os.environ.get("SSL_CERT_FILE"):
            await asyncio.to_thread(_merge_server_ca_into_ssl_cert_file, bundle)

    make_ssl_context.cache_clear()
    context = await asyncio.to_thread(make_ssl_context)
    if not await _can_verify_server(server_url, context):
        # The active trust bundle may be certifi rather than the OS bundle that
        # update-ca-certificates writes. Install into the bundle clients use.
        await asyncio.to_thread(_merge_server_ca_into_ssl_cert_file, bundle)
        make_ssl_context.cache_clear()
        context = await asyncio.to_thread(make_ssl_context)
        if not await _can_verify_server(server_url, context):
            raise RuntimeError(
                "The server CA was imported but TLS verification still failed. "
                "Configure SSL_CERT_FILE with a readable PEM CA bundle."
            )


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
    """Install a checksum-verified CA only when normal TLS cannot trust the server.

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
        bootstrap_context = await asyncio.to_thread(
            ssl.create_default_context, cafile=temporary_path
        )
        verified = await _can_verify_server(server_url, bootstrap_context)
        if not verified:
            raise RuntimeError(
                "The downloaded server certificate is not a usable trust anchor. "
                "Configure the server --ssl-ca-certfile with the issuing CA bundle."
            )
    finally:
        await asyncio.to_thread(os.unlink, temporary_path)

    await _activate_server_ca(server_url, bundle)
    logger.info("Imported the server CA certificate for worker registration.")
