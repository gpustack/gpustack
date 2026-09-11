import logging

import pytest

from gpustack.utils.forwarded import ForwardedHostPortMiddleware


async def _noop(*args, **kwargs):
    pass


async def _run(trusted_hosts, forwarded_host, host=b"real.example:80"):
    """
    trusted_hosts: list of trusted hostnames in config
    forwarded_host: the value of the X-Forwarded-Host header from client
    host: the real value of the Host header in the incoming request,
     if forwarded_host is untrusted, host will be returned
    """
    headers = [(b"host", host)]
    if forwarded_host is not None:
        headers.append((b"x-forwarded-host", forwarded_host))
    scope = {"type": "http", "headers": headers, "server": ("real.example", 80)}

    async def app(scope, receive, send):
        pass

    middleware = ForwardedHostPortMiddleware(app, trusted_hosts=trusted_hosts)
    await middleware(scope, _noop, _noop)
    return scope


def _host_header(scope):
    return dict(scope["headers"]).get(b"host")


@pytest.mark.asyncio
async def test_trusted_forwarded_host_rewrites_scope():
    scope = await _run(["proxy.example"], b"proxy.example:8443")
    assert _host_header(scope) == b"proxy.example:8443"
    assert scope["server"] == ("proxy.example", 8443)


@pytest.mark.asyncio
async def test_untrusted_forwarded_host_is_ignored():
    scope = await _run(["proxy.example"], b"evil.example")
    assert _host_header(scope) == b"real.example:80"
    assert scope["server"] == ("real.example", 80)


@pytest.mark.asyncio
async def test_wildcard_trusts_any_host():
    scope = await _run(["*"], b"anything.example")
    assert _host_header(scope) == b"anything.example"


@pytest.mark.asyncio
async def test_wildcard_non_utf8_forwarded_host_does_not_crash():
    # HTTP header octets may be non-UTF-8; wildcard mode skips the trust
    # decode, so the scope rewrite must decode leniently (latin-1) not crash.
    scope = await _run(["*"], b"caf\xe9.example:8080")
    assert _host_header(scope) == b"caf\xe9.example:8080"
    assert scope["server"] == ("caf\xe9.example", 8080)


@pytest.mark.asyncio
async def test_empty_allowlist_ignores_forwarded_host():
    scope = await _run([], b"evil.example")
    assert _host_header(scope) == b"real.example:80"


@pytest.mark.asyncio
async def test_port_lenient_and_ipv6_matching():
    # A different port on a trusted hostname still matches.
    scope = await _run(["proxy.example"], b"proxy.example:9999")
    assert _host_header(scope) == b"proxy.example:9999"
    assert scope["server"] == ("proxy.example", 9999)
    # Bracketed IPv6 forwarded host matches a bare IPv6 allowlist entry;
    # scope["server"] must carry the bare host (no brackets).
    scope = await _run(["::1"], b"[::1]:8080")
    assert _host_header(scope) == b"[::1]:8080"
    assert scope["server"] == ("::1", 8080)


# --- X-Forwarded-Proto diagnostics ------------------------------------------
#
# uvicorn owns the scheme: it parses this header before any app middleware and
# rewrites scope["scheme"] only for peers in forwarded_allow_ips, whose default
# is loopback. When it declines, the request is still served — so without the
# warning below, a proxy on another host produces a server that quietly believes
# it is on HTTP and never marks the session cookie Secure.


async def _run_proto(forwarded_proto, scheme="http"):
    """Return (scope, warning_count) after one request through the middleware."""
    headers = [(b"host", b"real.example:80")]
    if forwarded_proto is not None:
        headers.append((b"x-forwarded-proto", forwarded_proto))
    scope = {
        "type": "http",
        "headers": headers,
        "server": ("real.example", 80),
        "scheme": scheme,
    }

    async def app(scope, receive, send):
        pass

    middleware = ForwardedHostPortMiddleware(app, trusted_hosts=["*"])
    await middleware(scope, _noop, _noop)
    return scope, middleware._warned_proto


@pytest.mark.asyncio
async def test_https_proto_ignored_by_uvicorn_is_reported():
    # The header says https, the scheme says http: uvicorn saw the header and
    # refused it, which is the misconfiguration worth a log line.
    _, warned = await _run_proto(b"https")
    assert warned


@pytest.mark.asyncio
async def test_honoured_proto_is_not_reported():
    # uvicorn trusted the peer and already rewrote the scheme, so there is
    # nothing wrong to report.
    _, warned = await _run_proto(b"https", scheme="https")
    assert not warned


@pytest.mark.asyncio
async def test_plain_http_proto_is_not_reported():
    # A proxy honestly reporting http is not a downgrade; only the direction
    # that loses protection is flagged.
    _, warned = await _run_proto(b"http")
    assert not warned


@pytest.mark.asyncio
async def test_absent_proto_is_not_reported():
    _, warned = await _run_proto(None)
    assert not warned


@pytest.mark.asyncio
async def test_proto_chain_reads_the_client_facing_hop():
    # "https, http" is what several proxies each appending leaves behind; the
    # first entry is the scheme the client actually spoke.
    _, warned = await _run_proto(b"https, http")
    assert warned


@pytest.mark.asyncio
async def test_proto_warning_is_logged_once_however_many_requests(caplog):
    # It describes a deployment, not a request, so repeating it per request would
    # bury it in the log it is meant to stand out in.
    headers = [(b"host", b"real.example:80"), (b"x-forwarded-proto", b"https")]

    async def app(scope, receive, send):
        pass

    middleware = ForwardedHostPortMiddleware(app, trusted_hosts=["*"])

    with caplog.at_level(logging.WARNING, logger="gpustack.utils.forwarded"):
        for _ in range(3):
            await middleware(
                {
                    "type": "http",
                    "headers": headers,
                    "server": ("real.example", 80),
                    "scheme": "http",
                },
                _noop,
                _noop,
            )

    proto_warnings = [
        r for r in caplog.records if "X-Forwarded-Proto" in r.getMessage()
    ]
    assert len(proto_warnings) == 1
    # The remedy has to be in the message: the symptom (no Secure flag) shows up
    # nowhere near the cause (an unlisted proxy address).
    assert "forwarded_allow_ips" in proto_warnings[0].getMessage()
