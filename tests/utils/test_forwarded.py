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
