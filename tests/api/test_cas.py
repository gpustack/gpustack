"""Unit tests for the CAS (Central Authentication Service) flow in
:mod:`gpustack.routes.auth`. Mock-based; no live CAS server required."""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ConflictException, UnauthorizedException
from gpustack.routes import auth as auth_route


class _Response:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        # ``validate_cas_ticket`` parses XML from the raw bytes (so the
        # ``<?xml … encoding=… ?>`` declaration is honoured). Fakes
        # mirror that by encoding the test fixture under the encoding
        # the fixture declares — UTF-8 in all our cases.
        self.content = text.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            # Mirror httpx's raise_for_status shape just enough for the
            # caller's except branch to match.
            import httpx

            request = httpx.Request("GET", "http://example/")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


def _client(xml: str, *, status_code: int = 200) -> MagicMock:
    """A fake httpx.AsyncClient whose ``get`` returns ``xml``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=_Response(xml, status_code=status_code))
    return client


def _config(
    *,
    server_url: str = "https://cas.example.com/cas",
    validate_endpoint: Optional[str] = "/serviceValidate",
) -> MagicMock:
    cfg = MagicMock()
    cfg.cas_server_url = server_url
    cfg.cas_validate_endpoint = validate_endpoint
    return cfg


_SUCCESS_WITH_NS = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:user>alice</cas:user>
    <cas:attributes>
      <cas:displayName>Alice Smith</cas:displayName>
      <cas:email>alice@example.com</cas:email>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""

_SUCCESS_WITHOUT_NS = """<?xml version="1.0" encoding="UTF-8"?>
<serviceResponse>
  <authenticationSuccess>
    <user>bob</user>
    <attributes>
      <displayName>Bob Jones</displayName>
    </attributes>
  </authenticationSuccess>
</serviceResponse>
"""

_FAILURE = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationFailure code="INVALID_TICKET">
    Ticket not recognised
  </cas:authenticationFailure>
</cas:serviceResponse>
"""

_SUCCESS_WITHOUT_USER = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:attributes>
      <cas:displayName>Nameless</cas:displayName>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""


@pytest.mark.asyncio
async def test_validate_cas_ticket_parses_namespaced_success():
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITH_NS),
        ticket="ST-123",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "alice"
    assert data["displayName"] == "Alice Smith"
    assert data["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_validate_cas_ticket_parses_unnamespaced_success():
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITHOUT_NS),
        ticket="ST-456",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "bob"
    assert data["displayName"] == "Bob Jones"


@pytest.mark.asyncio
async def test_validate_cas_ticket_surfaces_authentication_failure():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client(_FAILURE),
            ticket="ST-bad",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    msg = str(exc.value.message)
    assert "INVALID_TICKET" in msg
    assert "Ticket not recognised" in msg


@pytest.mark.asyncio
async def test_validate_cas_ticket_rejects_malformed_xml():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client("<not-xml"),
            ticket="ST-bad",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "parse" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_validate_cas_ticket_rejects_response_without_user():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client(_SUCCESS_WITHOUT_USER),
            ticket="ST-789",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "username" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_validate_cas_ticket_surfaces_http_error():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client("ignored", status_code=503),
            ticket="ST-noop",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "503" in str(exc.value.message)


_SUCCESS_WITH_COMMENT = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:user>carol</cas:user>
    <cas:attributes>
      <!-- legacy field, kept for downstream compat -->
      <cas:displayName>Carol Lin</cas:displayName>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""


@pytest.mark.asyncio
async def test_validate_cas_ticket_skips_xml_comments_in_attributes():
    """``ElementTree`` represents XML comments with a non-string ``tag``
    (the ``Comment`` factory itself). The attribute parser must skip
    them rather than crash on ``.split``."""
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITH_COMMENT),
        ticket="ST-comment",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "carol"
    assert data["displayName"] == "Carol Lin"


_SUCCESS_WITH_MULTI_VALUED_GROUPS = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:user>dave</cas:user>
    <cas:attributes>
      <cas:displayName>Dave Park</cas:displayName>
      <cas:groups>engineering</cas:groups>
      <cas:groups>admins</cas:groups>
      <cas:groups>oncall</cas:groups>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""


@pytest.mark.asyncio
async def test_validate_cas_ticket_collects_multi_valued_attributes():
    """CAS 3.0 emits repeated child elements for multi-valued attributes
    (groups / roles / …). Naively overwriting on each iteration loses
    all but the last; accumulate into a list instead so future group
    sync has the full set, while single-valued attributes stay plain
    strings for backwards-compat with the username / full-name /
    avatar consumers.
    """
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITH_MULTI_VALUED_GROUPS),
        ticket="ST-dave",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "dave"
    # Single-valued stays a plain string.
    assert data["displayName"] == "Dave Park"
    # Multi-valued lifts to a list in document order.
    assert data["groups"] == ["engineering", "admins", "oncall"]


_XXE_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE serviceResponse [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:user>&xxe;</cas:user>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""


@pytest.mark.asyncio
async def test_validate_cas_ticket_does_not_resolve_external_entities():
    """A malicious / compromised CAS response must not be able to coax
    the XML parser into reading local files or hitting the network via
    XXE. With ``resolve_entities=False`` the ``&xxe;`` reference is
    never expanded, so ``<cas:user>`` ends up empty and we reject the
    response cleanly with ``UnauthorizedException`` — crucially without
    having touched ``file:///etc/passwd``.
    """
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client(_XXE_PAYLOAD),
            ticket="ST-xxe",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    msg = str(exc.value.message)
    assert "username" in msg.lower()
    assert "root:" not in msg
    assert "/etc/passwd" not in msg


@pytest.mark.asyncio
async def test_validate_cas_ticket_normalizes_endpoint_without_leading_slash():
    """Operators occasionally configure ``cas_validate_endpoint`` as
    ``p3/serviceValidate`` (no leading slash). The URL builder must
    tolerate that rather than emit ``…/casp3/serviceValidate``.
    """
    client = _client(_SUCCESS_WITH_NS)
    await auth_route.validate_cas_ticket(
        client,
        ticket="ST-noslash",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(validate_endpoint="p3/serviceValidate"),
    )
    url = client.get.await_args.args[0]
    assert "/cas/p3/serviceValidate?" in url
    assert "/casp3/" not in url


@pytest.mark.asyncio
async def test_validate_cas_ticket_strips_trailing_slash_and_uses_endpoint():
    """The validator must rstrip the configured base URL and respect a
    custom ``cas_validate_endpoint``."""
    client = _client(_SUCCESS_WITH_NS)
    await auth_route.validate_cas_ticket(
        client,
        ticket="ST-abc",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(
            server_url="https://cas.example.com/cas/",
            validate_endpoint="/p3/serviceValidate",
        ),
    )
    url = client.get.await_args.args[0]
    assert url.startswith("https://cas.example.com/cas/p3/serviceValidate?")
    assert "ticket=ST-abc" in url
    # ``service`` is fully percent-encoded (``safe=''``) — slashes
    # included — so strict CAS servers / proxies don't misparse the
    # query value.
    assert "service=https%3A%2F%2Fgpustack.example.com" in url
    assert "%2Fauth%2Fcas%2Fcallback" in url


def test_cas_added_to_auth_provider_enum():
    from gpustack.schemas.users import AuthProviderEnum

    assert AuthProviderEnum.CAS.value == "CAS"
    assert AuthProviderEnum.CAS == "CAS"


def _request_with_config(server_config) -> MagicMock:
    """Minimal Request stub for ``get_auth_config``: only ``.app.state.server_config`` is read."""
    request = MagicMock()
    request.app.state.server_config = server_config
    return request


@pytest.mark.asyncio
async def test_cas_callback_rejects_existing_user_from_other_source(monkeypatch):
    """A CAS login carrying a username that already exists as a Local
    / OIDC / SAML user must NOT be silently bound to that account —
    otherwise an attacker who can register the username on the CAS
    server hijacks the existing GPUStack admin / OIDC user. The
    response is a 303 redirect to ``/login?error=source_conflict``
    rather than a raw JSON 409 page, so the login form can render the
    actionable "ask an admin to link / convert" message via its
    ``?error=`` handler."""
    from gpustack.schemas.users import AuthProviderEnum

    monkeypatch.setattr(
        auth_route, "validate_cas_ticket", AsyncMock(return_value={"username": "admin"})
    )
    monkeypatch.setattr(auth_route, "use_proxy_env_for_url", lambda url: False)
    monkeypatch.setattr(auth_route, "make_ssl_context", lambda: None)
    monkeypatch.setattr(
        auth_route.httpx, "AsyncClient", lambda **kw: _AsyncClientFake()
    )
    existing_local_admin = MagicMock()
    existing_local_admin.source = AuthProviderEnum.Local
    monkeypatch.setattr(
        auth_route.User,
        "first_by_fields",
        AsyncMock(return_value=existing_local_admin),
    )

    request = MagicMock()
    request.app.state.server_config.cas_server_url = "https://cas.example.com/cas"
    request.app.state.server_config.cas_callback_url = None
    request.app.state.server_config.server_external_url = "https://gpustack.example.com"
    request.app.state.server_config.cas_username_attribute = None
    request.app.state.server_config.cas_full_name_attribute = None
    request.app.state.server_config.cas_avatar_attribute = None
    request.app.state.server_config.external_auth_default_inactive = False
    request.app.url_path_for.return_value = "/auth/cas/callback"
    request.query_params = {"ticket": "ST-attacker"}

    response = await auth_route.cas_callback(request=request, session=MagicMock())
    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == auth_route.SOURCE_CONFLICT_LOGIN_URL


@pytest.mark.asyncio
async def test_resolve_external_user_raises_conflict_on_mismatch(monkeypatch):
    """The structured 409 path that the browser callbacks layer the
    redirect on top of: direct callers (and the existing tests) get a
    typed ``ConflictException`` with the message that surfaces in the
    login UI."""
    from gpustack.schemas.users import AuthProviderEnum

    existing = MagicMock()
    existing.source = AuthProviderEnum.Local
    monkeypatch.setattr(
        auth_route.User, "first_by_fields", AsyncMock(return_value=existing)
    )

    with pytest.raises(ConflictException) as exc:
        await auth_route._resolve_external_user(
            MagicMock(), "admin", AuthProviderEnum.CAS
        )
    assert "different authentication source" in str(exc.value.message)
    assert "administrator" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_cas_callback_translates_other_failures_to_auth_failed(monkeypatch):
    """Anything that isn't a source conflict — bad ticket, malformed
    response, IdP unreachable — lands on ``/login?error=auth_failed``.
    Mirrors the source-conflict redirect: the browser never sees a
    raw JSON error page, and the underlying detail goes to the server
    log via the decorator's ``logger.exception`` call."""
    from gpustack.api.exceptions import UnauthorizedException

    # validate_cas_ticket raises Unauthorized on a bad ticket — the
    # most representative non-conflict failure for this callback.
    monkeypatch.setattr(
        auth_route,
        "validate_cas_ticket",
        AsyncMock(side_effect=UnauthorizedException(message="bad ticket")),
    )
    monkeypatch.setattr(auth_route, "use_proxy_env_for_url", lambda url: False)
    monkeypatch.setattr(auth_route, "make_ssl_context", lambda: None)
    monkeypatch.setattr(
        auth_route.httpx, "AsyncClient", lambda **kw: _AsyncClientFake()
    )

    request = MagicMock()
    request.app.state.server_config.cas_server_url = "https://cas.example.com/cas"
    request.app.state.server_config.cas_callback_url = None
    request.app.state.server_config.server_external_url = "https://gpustack.example.com"
    request.app.url_path_for.return_value = "/auth/cas/callback"
    request.query_params = {"ticket": "ST-bad"}

    response = await auth_route.cas_callback(request=request, session=MagicMock())
    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == auth_route.AUTH_FAILED_LOGIN_URL


class _AsyncClientFake:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


def _auth_config(
    *,
    external_auth_type=None,
    data_dir: str = "/tmp/__gpustack_test_nonexistent__",
) -> MagicMock:
    cfg = MagicMock()
    cfg.external_auth_type = external_auth_type
    cfg.data_dir = data_dir
    return cfg


@pytest.mark.asyncio
async def test_get_auth_config_advertises_cas_external_auth():
    from gpustack.schemas.users import AuthProviderEnum

    request = _request_with_config(
        _auth_config(external_auth_type=AuthProviderEnum.CAS)
    )
    result = await auth_route.get_auth_config(request=request, session=None)
    assert result["external_auth"] == {
        "type": AuthProviderEnum.CAS,
        "login_url": "/auth/cas/login",
    }
    # CAS post-dates the deprecated boolean shape, so it must not be
    # backfilled into a fourth ``is_cas`` field.
    assert "is_cas" not in result
    assert result["is_oidc"] is False
    assert result["is_saml"] is False


@pytest.mark.asyncio
async def test_get_auth_config_advertises_oidc_external_auth():
    from gpustack.schemas.users import AuthProviderEnum

    request = _request_with_config(
        _auth_config(external_auth_type=AuthProviderEnum.OIDC)
    )
    result = await auth_route.get_auth_config(request=request, session=None)
    assert result["external_auth"] == {
        "type": AuthProviderEnum.OIDC,
        "login_url": "/auth/oidc/login",
    }
    # Deprecated boolean shape kept for older UI bundles.
    assert result["is_oidc"] is True
    assert result["is_saml"] is False


def _cas_request(
    *,
    cas_callback_url=None,
    server_external_url=None,
    base_url: str = "http://localhost/",
) -> MagicMock:
    """Stub a ``Request`` carrying ``app.state.server_config`` plus the
    minimal URL-resolution surface used by ``_cas_service_url``."""
    request = MagicMock()
    request.app.state.server_config.cas_callback_url = cas_callback_url
    request.app.state.server_config.server_external_url = server_external_url
    request.url_for.return_value = base_url.rstrip("/") + "/auth/cas/callback"
    request.app.url_path_for.return_value = "/auth/cas/callback"
    return request


def test_cas_service_url_uses_only_starlette_request_surface():
    """Earlier we mocked ``request.url_path_for`` directly on the
    Request, which silently faked an attribute that doesn't exist on
    real Starlette ``Request`` objects — the only ``url_path_for``
    lives on the router / app. Reproduce here without ``MagicMock``'s
    permissive auto-attribute behaviour to lock in: ``_cas_service_url``
    must only touch ``request.url_for`` / ``request.app.url_path_for``,
    never a phantom ``request.url_path_for``.
    """

    class _StrictRequest:
        """Mimics Starlette's ``Request`` surface narrowly — accessing
        an undefined attribute raises ``AttributeError`` instead of
        materialising a Mock."""

        def __init__(self, app, server_config):
            self.app = app
            self.app.state.server_config = server_config

        def url_for(self, name):
            return f"http://example.test/{name}"

    class _App:
        class state:  # noqa: D401 — namespace, not class
            pass

        def url_path_for(self, name):
            return f"/auth/{name.replace('cas_callback', 'cas/callback')}"

    config = MagicMock()
    config.cas_callback_url = None
    config.server_external_url = "https://gpustack.example.com"
    request = _StrictRequest(_App(), config)

    result = auth_route._cas_service_url(request, config)
    assert result == "https://gpustack.example.com/auth/cas/callback"


def test_cas_service_url_prefers_explicit_callback_override():
    request = _cas_request(
        cas_callback_url="https://gpustack.example.com/cas-cb",
        server_external_url="https://gpustack.example.com",
    )
    assert (
        auth_route._cas_service_url(request, request.app.state.server_config)
        == "https://gpustack.example.com/cas-cb"
    )


def test_cas_service_url_uses_server_external_url_when_no_override():
    """A reverse-proxy / TLS-terminated deployment relies on this path
    so the ``service`` URL has the real external scheme + host instead
    of whatever ``request.base_url`` reflects."""
    request = _cas_request(
        cas_callback_url=None,
        server_external_url="https://gpustack.example.com/",
        base_url="http://localhost:8080/",
    )
    assert (
        auth_route._cas_service_url(request, request.app.state.server_config)
        == "https://gpustack.example.com/auth/cas/callback"
    )


def test_cas_service_url_falls_back_to_request_when_neither_set():
    request = _cas_request(
        cas_callback_url=None,
        server_external_url=None,
        base_url="http://192.168.0.1:8080/",
    )
    assert (
        auth_route._cas_service_url(request, request.app.state.server_config)
        == "http://192.168.0.1:8080/auth/cas/callback"
    )


@pytest.mark.asyncio
async def test_get_auth_config_returns_null_external_auth_when_local():
    request = _request_with_config(_auth_config(external_auth_type=None))
    result = await auth_route.get_auth_config(request=request, session=None)
    assert result["external_auth"] is None
    assert result["is_oidc"] is False
    assert result["is_saml"] is False
