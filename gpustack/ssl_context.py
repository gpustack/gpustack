"""SSL context factory for the worker / client side.

Replaces the previous ``truststore.SSLContext`` usage. ``truststore`` on Linux
calls ``ctx.set_default_verify_paths()`` on every ``wrap_socket()`` call, which
pushes a new ``X509_LOOKUP_hash_dir`` entry into the OpenSSL ``X509_STORE``
without deduplication. The lookup list grows linearly with the number of TLS
handshakes, and certificate verification scales as ``K(K+1)/2`` directory
scans -- a worker that has been up for a few days starts spending tens of
seconds per handshake, far exceeding the gateway's
``transport_socket_connect_timeout``.

This factory pins the trust store to a CA bundle **file** via
``X509_LOOKUP_file``, so the lookup state is fixed once at startup and never
accumulates. Self-signed / private CAs continue to work via the existing
``update-ca-certificates`` flow in ``gpustack-prerun.sh``, which merges
``/usr/local/share/ca-certificates/*.crt`` into the OS bundle before the
process starts.

See ``.cache/issues/worker-not-ready/README.md`` for the full investigation.
"""

from __future__ import annotations

import logging
import os
import ssl
from functools import lru_cache

import certifi

logger = logging.getLogger(__name__)


def resolve_ca_bundle() -> str:
    """Return the CA bundle file to verify peers against.

    Precedence (first existing file wins):

    1. ``SSL_CERT_FILE`` env var. OpenSSL's own override knob; lets operators
       point at a custom bundle without rebuilding the image. If set but the
       path does not exist, a warning is logged and resolution falls through
       to the next candidate -- the misconfiguration is loud, not silent.
    2. ``ssl.get_default_verify_paths().cafile``. Whatever OpenSSL was
       compiled to use as the default bundle on this platform -- each
       distro configures it to point at the same file that
       ``update-ca-certificates`` / ``update-ca-trust`` write back to,
       so user-added private CAs are picked up transparently.
    3. ``certifi.where()``. Last-resort fallback for environments where
       OpenSSL's compiled default points at a missing file (stripped
       container images, broken installs) or both of the above are unset.
    """
    _warn_unsupported_ssl_env_vars()
    for candidate in (
        os.environ.get("SSL_CERT_FILE"),
        ssl.get_default_verify_paths().cafile,
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    return certifi.where()


@lru_cache(maxsize=1)
def _warn_unsupported_ssl_env_vars() -> None:
    """Emit one-shot warnings for SSL env vars that gpustack handles
    differently from OpenSSL's defaults.

    - ``SSL_CERT_FILE`` set but the file does not exist: we fall through
      to the system default, but the misconfiguration would otherwise be
      silent.
    - ``SSL_CERT_DIR`` is intentionally ignored (would route through the
      ``X509_LOOKUP_hash_dir`` path this module avoids). Operators who
      set it should see how to achieve the same intent instead.

    Memoized so a process emits each warning at most once, regardless of
    how many handshakes / clients trigger resolution.
    """
    env_file = os.environ.get("SSL_CERT_FILE")
    if env_file and not os.path.isfile(env_file):
        logger.warning(
            "SSL_CERT_FILE=%r is set but the file does not exist; "
            "falling back to the system default CA bundle.",
            env_file,
        )

    env_dir = os.environ.get("SSL_CERT_DIR")
    if env_dir:
        logger.warning(
            "SSL_CERT_DIR=%r is set but ignored. To trust extra CAs, drop "
            "them into /usr/local/share/ca-certificates/ (picked up by "
            "update-ca-certificates) or point SSL_CERT_FILE at a merged "
            "bundle.",
            env_dir,
        )


@lru_cache(maxsize=1)
def make_ssl_context() -> ssl.SSLContext:
    """Return a process-wide ``ssl.SSLContext`` that trusts the OS bundle.

    The bundle file is read once into OpenSSL's in-memory hash table.
    Repeated ``wrap_socket()`` calls reuse that state -- no per-handshake
    directory scanning, no unbounded ``X509_LOOKUP`` accumulation.

    .. warning::

        The returned context is a **process-wide singleton** (memoized by
        ``lru_cache``). Callers MUST NOT mutate it -- no
        ``load_cert_chain()``, no ``check_hostname = False``, no
        ``verify_mode = CERT_NONE``, no ``set_ciphers()``. Mutations would
        leak across every caller (httpx clients, auth flows, etc.) and
        silently weaken TLS verification elsewhere in the process.

        If you need a customized context (insecure mode, client cert,
        pinned ciphers, ...), construct your own ``ssl.SSLContext`` --
        don't reach for this factory.
    """
    return ssl.create_default_context(cafile=resolve_ca_bundle())
