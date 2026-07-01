import ssl
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from gpustack.api.auth import get_current_user, worker_auth
from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes import auth as auth_route
from gpustack.routes.auth import oidc_callback


class DummyWorkerConfig:
    token = "registration-token"

    def get_server_url(self):
        return "http://example.com"


@pytest.mark.asyncio
async def test_get_current_user_accepts_x_api_key(monkeypatch):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    user = await get_current_user(
        request=request,
        session=session,
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user
    assert request.state.user is expected_user
    assert request.state.api_key is expected_key


@pytest.mark.asyncio
async def test_worker_auth_accepts_x_api_key():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert await worker_auth(request=request, x_api_key="worker-token") is None


@pytest.mark.asyncio
async def test_worker_auth_rejects_missing_credentials():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    with pytest.raises(UnauthorizedException):
        await worker_auth(request=request)


@pytest.mark.asyncio
async def test_get_current_user_falls_back_to_x_api_key_when_bearer_empty(
    monkeypatch,
):
    session = object()
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = {}
    request.client = type("Client", (), {"host": "10.0.0.1"})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()

    expected_user = type("User", (), {"is_active": True})()
    expected_key = object()

    auth_mock = AsyncMock(return_value=(expected_user, expected_key))
    monkeypatch.setattr("gpustack.api.auth.get_user_from_api_token", auth_mock)

    user = await get_current_user(
        request=request,
        session=session,
        bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
        x_api_key="sk_test_value",
    )

    auth_mock.assert_awaited_once_with(session, "sk_test_value")
    assert user is expected_user


@pytest.mark.asyncio
async def test_worker_auth_falls_back_to_x_api_key_when_bearer_empty():
    request = type("Request", (), {})()
    request.headers = {"X-Higress-Llm-Model": "claude-sonnet"}
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.token = "worker-token"
    request.app.state.config = DummyWorkerConfig()
    request.app.state.http_client_no_proxy = object()

    assert (
        await worker_auth(
            request=request,
            bearer_token=HTTPAuthorizationCredentials(scheme="Bearer", credentials=""),
            x_api_key="worker-token",
        )
        is None
    )


def _make_request(headers=None, client_host="127.0.0.1"):
    request = type("Request", (), {})()
    request.state = type("State", (), {})()
    request.headers = headers or {}
    request.client = type("Client", (), {"host": client_host})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type("Config", (), {"gateway_mode": None})()
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_host,headers",
    [
        # Genuine local request — no longer auto-trusted.
        ("127.0.0.1", {"host": "127.0.0.1:30080"}),
        # Reverse-proxy-fronted remote attacker arriving with TCP peer 127.0.0.1.
        (
            "127.0.0.1",
            {"host": "gpustack.example.com", "x-forwarded-for": "8.8.8.8"},
        ),
        # IPv6 loopback.
        ("::1", {"host": "[::1]:30080"}),
        # External IP.
        ("10.0.0.1", {"host": "gpustack.example.com"}),
    ],
)
async def test_get_current_user_requires_credentials(monkeypatch, client_host, headers):
    # The auto-admin localhost shortcut has been removed entirely.
    # Every unauthenticated request — local, proxied, or remote — must be
    # rejected.
    session = object()
    request = _make_request(headers=headers, client_host=client_host)

    first_by_field = AsyncMock()
    get_by_username = AsyncMock()
    monkeypatch.setattr("gpustack.api.auth.User.first_by_field", first_by_field)
    monkeypatch.setattr(
        "gpustack.api.auth.UserService.get_by_username", get_by_username
    )

    with pytest.raises(UnauthorizedException):
        await get_current_user(request=request, session=session)
    # No DB lookup path may fire when there are no credentials.
    first_by_field.assert_not_awaited()
    get_by_username.assert_not_awaited()


@pytest.mark.asyncio
async def test_oidc_callback_uses_system_trust_store(monkeypatch):
    captured = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, data=None):
            return type(
                "Resp",
                (),
                {
                    "status_code": 200,
                    "text": '{"access_token":"token","id_token":"id"}',
                },
            )()

    request = type("Request", (), {})()
    request.app = type("App", (), {})()
    request.app.state = type("State", (), {})()
    request.app.state.server_config = type(
        "Config",
        (),
        {
            "oidc_client_id": "client-id",
            "oidc_client_secret": "client-secret",
            "oidc_redirect_uri": "https://gpustack.example.com/auth/oidc/callback",
            "openid_configuration": {
                "token_endpoint": "https://issuer.example.com/token"
            },
            "external_auth_name": None,
            "external_auth_full_name": None,
            "external_auth_avatar_url": None,
            "external_auth_default_inactive": False,
            # Group sync defaults to False; this test exercises the
            # trust-store path, not group sync.
            "external_auth_group_sync": False,
            "external_auth_groups": None,
        },
    )()
    request.app.state.jwt_manager = type(
        "JWTManager", (), {"create_jwt_token": lambda self, username: "jwt-token"}
    )()
    request.query_params = {"code": "auth-code", "state": "test-state"}
    request.cookies = {"gpustack_oidc_state": "test-state"}
    # Read by the session-cookie hardening (``secure=request.url.scheme
    # == "https"``). Pretend the inbound request was HTTPS so the
    # ``secure`` flag would have flipped on in production.
    request.url = type("URL", (), {"scheme": "https"})()

    monkeypatch.setattr("gpustack.routes.auth.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("gpustack.routes.auth.use_proxy_env_for_url", lambda url: False)
    monkeypatch.setattr(
        "gpustack.routes.auth.get_oidc_user_data",
        AsyncMock(return_value={"email": "user@example.com", "name": "Test User"}),
    )
    # Return an existing user already tagged as the OIDC source so the
    # cross-provider-takeover guard treats this as a legitimate repeat
    # login rather than a username collision from a different IdP.
    from gpustack.schemas.users import AuthProviderEnum

    existing_oidc_user = type(
        "User", (), {"is_active": True, "source": AuthProviderEnum.OIDC}
    )()
    monkeypatch.setattr(
        "gpustack.routes.auth.User.first_by_fields",
        AsyncMock(return_value=existing_oidc_user),
    )

    response = await oidc_callback(request=request, session=object())

    assert response.status_code in (302, 307)
    assert captured["trust_env"] is False
    assert captured["timeout"] is not None
    assert isinstance(captured["verify"], ssl.SSLContext)


@pytest.mark.asyncio
async def test_legacy_server_token_principal_authenticates():
    """Pre-2.0 workers authenticate every request with Basic
    ``system/worker/<uuid>:<server-token>``. The minted in-memory
    principal must construct without tripping the schema's
    is_admin-on-non-USER guard and pass the worker / cluster gates."""
    from fastapi.security import HTTPBasicCredentials

    from gpustack.api.auth import (
        authenticate_system_principal,
        get_cluster_principal,
        get_worker_principal,
        is_server_token_principal,
    )
    from gpustack.schemas.principals import PrincipalType

    config = type("Config", (), {"token": "server-token"})()
    principal = await authenticate_system_principal(
        config,
        HTTPBasicCredentials(username="system/worker/abc", password="server-token"),
    )

    assert principal is not None
    assert principal.kind == PrincipalType.SYSTEM
    assert principal.is_admin is False
    assert principal.id is None
    assert is_server_token_principal(principal)
    assert (await get_cluster_principal(principal)) is principal
    assert (await get_worker_principal(principal)) is principal

    # Wrong password mints nothing.
    rejected = await authenticate_system_principal(
        config,
        HTTPBasicCredentials(username="system/worker/abc", password="wrong"),
    )
    assert rejected is None


@pytest.mark.asyncio
async def test_persisted_system_principal_without_links_hits_admin_gate():
    """A persisted SYSTEM principal (id set) with neither worker nor
    cluster back-reference is NOT the server-token principal and must
    fall through to the admin gate."""
    from gpustack.api.auth import get_worker_principal, is_server_token_principal
    from gpustack.api.exceptions import ForbiddenException
    from gpustack.schemas.principals import Principal, PrincipalType

    principal = Principal(name="system/worker-orphan", kind=PrincipalType.SYSTEM)
    principal.id = 123

    assert not is_server_token_principal(principal)
    with pytest.raises(ForbiddenException):
        await get_worker_principal(principal)


def test_saml_settings_signature_floor():
    """Signature floor: at least one of ``wantAssertionsSigned`` /
    ``wantMessagesSigned`` must be True after the helper resolves the
    operator's ``--saml-security``. Both defaulting to False (the
    OneLogin ship state) would let ``process_response`` admit forged
    (unsigned) assertions — the vulnerability this fix exists to
    close. Operators who already opted in to either — some IdPs sign
    only the Response, others only the Assertion — keep their choice.
    """
    config = MagicMock()

    # Operator passed no security config → toolkit ships with both
    # off; helper defaults ``wantAssertionsSigned`` on to enforce the
    # floor.
    config.saml_security = "{}"
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is True

    # Operator signs only the outer ``<Response>`` (some IdPs do this,
    # not the Assertion). Respect that — don't force both.
    config.saml_security = '{"wantAssertionsSigned": false, "wantMessagesSigned": true}'
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is False
    assert security.get("wantMessagesSigned") is True

    # Operator signs only the ``<Assertion>`` (typical Keycloak
    # setup). Respect that too.
    config.saml_security = '{"wantAssertionsSigned": true, "wantMessagesSigned": false}'
    security = auth_route._saml_settings(config)["security"]
    assert security.get("wantAssertionsSigned") is True
    assert security.get("wantMessagesSigned") is False


def test_saml_unsigned_escape_hatch_detects_both_false():
    """The unsigned escape hatch flags only when *both*
    ``wantAssertionsSigned`` and ``wantMessagesSigned`` are the
    literal ``False`` from operator input — a missing key, or a
    half-opt-out, must not turn the hatch on. The callback branches
    on this to decide whether to skip signature verification, so
    correctness of the detection is security-load-bearing."""
    config = MagicMock()

    # Both explicitly False → hatch on
    config.saml_security = (
        '{"wantAssertionsSigned": false, "wantMessagesSigned": false}'
    )
    assert auth_route._saml_unsigned_escape_hatch(config) is True

    # Only one explicit False → hatch off (still hits the floor)
    config.saml_security = '{"wantAssertionsSigned": false}'
    assert auth_route._saml_unsigned_escape_hatch(config) is False

    # Missing keys → hatch off
    config.saml_security = "{}"
    assert auth_route._saml_unsigned_escape_hatch(config) is False

    # Explicit True somewhere → hatch off
    config.saml_security = '{"wantAssertionsSigned": true, "wantMessagesSigned": false}'
    assert auth_route._saml_unsigned_escape_hatch(config) is False


def test_extract_saml_attributes_unsigned_returns_nameid_and_attributes():
    """The unsigned parser must extract the same downstream shape the
    toolkit path produces (single-valued as bare string, multi-valued
    as list, plus a ``name_id`` key) so the rest of the callback runs
    unchanged."""
    xml = (
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>alice@example.com"
        "</saml:NameID></saml:Subject><saml:AttributeStatement>"
        '<saml:Attribute Name="email"><saml:AttributeValue>alice@example.com'
        "</saml:AttributeValue></saml:Attribute>"
        '<saml:Attribute Name="Role"><saml:AttributeValue>engineer'
        "</saml:AttributeValue></saml:Attribute>"
        '<saml:Attribute Name="Role"><saml:AttributeValue>admin'
        "</saml:AttributeValue></saml:Attribute>"
        "</saml:AttributeStatement></saml:Assertion></samlp:Response>"
    ).encode()
    attrs = auth_route._extract_saml_attributes_unsigned(xml)
    assert attrs["name_id"] == "alice@example.com"
    assert attrs["email"] == "alice@example.com"
    # Repeated ``<Attribute Name="Role">`` should be merged into a
    # list, matching ``allowRepeatAttributeName=True`` in the toolkit
    # path.
    assert attrs["Role"] == ["engineer", "admin"]


def test_extract_saml_attributes_unsigned_rejects_xxe():
    """Even in the unsigned escape-hatch path, the parser must not
    resolve external entities — otherwise turning signature
    verification off for local IdP tests would also open an XXE
    file-read vector."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<!DOCTYPE samlp:Response ["
        '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
        "]>"
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>&xxe;</saml:NameID>"
        "</saml:Subject></saml:Assertion></samlp:Response>"
    ).encode()
    attrs = auth_route._extract_saml_attributes_unsigned(xml)
    # No expansion happened — the parser didn't fetch /etc/passwd.
    assert "root:" not in (attrs.get("name_id") or "")
    assert "/etc/passwd" not in (attrs.get("name_id") or "")


_SAML_ENV_WRAPPER = (
    '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
    'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
    'xmlns:ds="http://www.w3.org/2000/09/xmldsig#">'
    "{signatures}"
    "<saml:Assertion>{assertion_signature}<saml:Subject>"
    "<saml:NameID>alice@example.com</saml:NameID>"
    "</saml:Subject></saml:Assertion></samlp:Response>"
)
_FAKE_SIG = "<ds:Signature><ds:SignatureValue>dummy</ds:SignatureValue></ds:Signature>"


def _saml_response_xml(*, response_signed: bool, assertion_signed: bool) -> bytes:
    return _SAML_ENV_WRAPPER.format(
        signatures=_FAKE_SIG if response_signed else "",
        assertion_signature=_FAKE_SIG if assertion_signed else "",
    ).encode()


def test_saml_settings_adapts_to_response_only_signature():
    """When Keycloak / ADFS is configured to sign the outer
    ``<Response>`` but not the inner ``<Assertion>`` (Keycloak's own
    docs describe this as valid: ``Sign Documents=On, Sign
    Assertions=Off``), the callback must ask the toolkit to require
    the message signature rather than reject with "Assertion not
    signed". Adaptive detection sets ``wantMessagesSigned`` based on
    the DOM shape of what actually arrived — no ``--saml-security``
    tweaking needed for this common config."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=True, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantMessagesSigned") is True
    # Assertion-signed floor NOT forced in — the response *is* signed,
    # just at the outer level.
    assert security.get("wantAssertionsSigned") is not True


def test_saml_settings_adapts_to_assertion_only_signature():
    """Symmetric case: only the inner ``<Assertion>`` is signed
    (Keycloak default when ``Sign Assertions=On, Sign Documents=Off``)."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=False, assertion_signed=True)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True
    assert security.get("wantMessagesSigned") is not True


def test_saml_settings_operator_explicit_wins_over_adaptive():
    """Operator strictness must not be dialled *down* by adaptive
    detection: a deployment that mandates Assertion signing (via
    ``--saml-security``) keeps that requirement even if the incoming
    response only signs the outer ``<Response>``. Better to reject a
    non-compliant IdP than silently lower the bar."""

    config = MagicMock()
    config.saml_security = '{"wantAssertionsSigned": true}'
    xml = _saml_response_xml(response_signed=True, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True
    # Adaptive still added the ``wantMessagesSigned`` since the
    # Response *is* signed and the operator didn't explicitly refuse.
    assert security.get("wantMessagesSigned") is True


def test_saml_settings_unsigned_response_still_hits_floor():
    """If the DOM has neither signature and the operator didn't
    opt in either way, the floor kicks in with
    ``wantAssertionsSigned=True`` so the toolkit refuses the
    (attacker-forgeable) unsigned response."""

    config = MagicMock()
    config.saml_security = "{}"
    xml = _saml_response_xml(response_signed=False, assertion_signed=False)
    security = auth_route._saml_settings(config, xml_bytes=xml)["security"]
    assert security.get("wantAssertionsSigned") is True


def test_saml_settings_defaults_allow_repeat_attribute_name():
    """SAML allows multi-valued attributes as either repeated
    ``<AttributeValue>`` inside one ``<Attribute>`` or as multiple
    ``<Attribute Name="X">`` elements. Keycloak's default mappers
    emit the latter (role_list + role_name both write ``Role``);
    the toolkit's out-of-box strict mode rejects that. Default the
    knob on so real IdPs work without extra config; operators who
    want strict mode can opt in via ``--saml-security``."""

    config = MagicMock()

    config.saml_security = "{}"
    assert (
        auth_route._saml_settings(config)["security"]["allowRepeatAttributeName"]
        is True
    )

    # Operator explicit opt-out is respected — strict-mode
    # deployments can still catch mis-configured IdPs.
    config.saml_security = '{"allowRepeatAttributeName": false}'
    assert (
        auth_route._saml_settings(config)["security"]["allowRepeatAttributeName"]
        is False
    )


def _saml_callback_request(saml_response_b64: str, **config_overrides) -> object:
    """Build a request double for the SAML callback tests. The
    callback derives OneLogin's ``current_url`` from ``saml_sp_acs_url``,
    not from ``request.url``, so the URL fields on the request are
    only used for the ``get_data`` payload."""

    request = MagicMock()
    request.method = "POST"

    async def _form():
        return {"SAMLResponse": saml_response_b64}

    request.form = _form
    request.query_params = {}

    cfg = request.app.state.server_config
    cfg.saml_security = "{}"
    cfg.saml_sp_acs_url = "http://localhost:9000/auth/saml/callback"
    cfg.external_auth_name = None
    cfg.external_auth_full_name = None
    cfg.external_auth_avatar_url = None
    cfg.external_auth_default_inactive = False
    for k, v in config_overrides.items():
        setattr(cfg, k, v)
    return request


def _patch_saml_auth(monkeypatch, **auth_overrides):
    """Replace ``OneLogin_Saml2_Auth`` with a scripted fake so tests
    can exercise the callback's flow around signature validation
    without producing real signed XML. The fake returns the caller's
    scripted ``get_errors`` / ``is_authenticated`` / ``get_nameid`` /
    ``get_attributes`` values from ``process_response`` onwards."""

    fake = MagicMock()
    fake.process_response = MagicMock()
    fake.get_errors = MagicMock(return_value=auth_overrides.get("errors", []))
    fake.get_last_error_reason = MagicMock(
        return_value=auth_overrides.get("error_reason", "")
    )
    fake.is_authenticated = MagicMock(
        return_value=auth_overrides.get("authenticated", True)
    )
    fake.get_nameid = MagicMock(return_value=auth_overrides.get("nameid", ""))
    fake.get_attributes = MagicMock(return_value=auth_overrides.get("attributes", {}))
    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", MagicMock(return_value=fake))
    return fake


@pytest.mark.asyncio
async def test_saml_callback_rejects_when_toolkit_reports_errors(monkeypatch):
    """Signature failure surfaces via ``auth.get_errors()``. The
    callback must raise so the decorator produces an ``auth_failed``
    redirect — never mint a JWT for an unverified assertion."""

    _patch_saml_auth(
        monkeypatch,
        errors=["invalid_response"],
        error_reason="Signature validation failed",
    )
    request = _saml_callback_request("dummy-b64")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "Signature validation failed" in str(exc.value.message)


@pytest.mark.asyncio
async def test_saml_callback_rejects_when_not_authenticated(monkeypatch):
    """The toolkit can complete ``process_response`` with an empty
    error list but ``is_authenticated`` still False (e.g. status
    element carries a non-Success code). Treat that the same as a
    signature failure — refuse to trust NameID / attributes."""

    _patch_saml_auth(monkeypatch, errors=[], authenticated=False)
    request = _saml_callback_request("dummy-b64")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "not authenticated" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_saml_callback_missing_configured_username_attribute_rejects(
    monkeypatch,
):
    """When the operator pins ``external_auth_name`` to a specific
    attribute and the (verified) assertion doesn't carry it, fail
    loudly at the source rather than letting ``None`` flow into the
    user resolve/create path."""

    _patch_saml_auth(
        monkeypatch,
        authenticated=True,
        nameid="alice@example.com",
        attributes={"email": ["alice@example.com"]},
    )
    # Operator pointed at ``employeeId`` — the (mocked-verified)
    # assertion above only carries ``email`` and ``name_id``.
    request = _saml_callback_request("dummy-b64", external_auth_name="employeeId")

    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())
    assert "employeeId" in str(exc.value.message)


@pytest.mark.asyncio
async def test_saml_callback_derives_current_url_from_configured_acs(monkeypatch):
    """Reverse-proxy / UI-dev-server setups routinely land the callback
    request on an internal host:port that doesn't match what Keycloak
    signed the assertion for. The toolkit's Destination check would
    then reject valid assertions — the fix is to anchor
    ``current_url`` on the operator's configured ACS URL rather than
    on ``request.url``. This test pins that behaviour: even though
    the request's ``url`` claims a different host / port / scheme,
    ``OneLogin_Saml2_Auth`` is constructed with the ACS-derived
    request_data."""

    _patch_saml_auth(
        monkeypatch,
        authenticated=True,
        nameid="alice@example.com",
        attributes={"email": ["alice@example.com"]},
    )
    captured = {}

    def _capture(req, settings):
        captured["req"] = req
        captured["settings"] = settings
        fake = MagicMock()
        fake.process_response = MagicMock()
        fake.get_errors = MagicMock(return_value=[])
        fake.is_authenticated = MagicMock(return_value=True)
        fake.get_nameid = MagicMock(return_value="alice@example.com")
        fake.get_attributes = MagicMock(return_value={"email": ["alice@example.com"]})
        return fake

    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", _capture)
    monkeypatch.setattr(
        auth_route,
        "_resolve_or_provision_external_user",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        auth_route, "_sync_saml_groups_if_enabled", AsyncMock(return_value=None)
    )

    request = _saml_callback_request(
        "dummy-b64",
        saml_sp_acs_url="https://gpustack.example.com:8443/auth/saml/callback",
    )
    # Whatever the *request* URL looks like (a UI dev server on 9000
    # proxying to a backend on some other port, TLS termination
    # elsewhere, ...) — the toolkit still gets the config-anchored
    # values.
    request.url.scheme = "http"
    request.url.hostname = "localhost"
    request.url.port = None
    request.url.path = "/proxied/path"

    request.app.state.jwt_manager.create_jwt_token = MagicMock(return_value="fake-jwt")

    await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())

    assert captured["req"]["http_host"] == "gpustack.example.com"
    assert captured["req"]["server_port"] == "8443"
    assert captured["req"]["https"] == "on"
    assert captured["req"]["script_name"] == "/auth/saml/callback"
    assert captured["req"]["post_data"] == {"SAMLResponse": "dummy-b64"}


@pytest.mark.asyncio
async def test_saml_callback_unsigned_escape_hatch_skips_toolkit(monkeypatch, caplog):
    """With the operator's explicit both-False opt-out, the callback
    must **not** call ``OneLogin_Saml2_Auth`` at all (its hard-coded
    "No Signature found" check would reject the unsigned response
    regardless of the ``wantX`` flags). It must instead extract
    NameID / attributes via the manual parser, log the loud warning
    on that path, and continue to JWT minting."""
    import base64
    import logging

    unsigned_xml = (
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"'
        ' xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">'
        "<saml:Assertion><saml:Subject><saml:NameID>alice@example.com"
        "</saml:NameID></saml:Subject><saml:AttributeStatement>"
        '<saml:Attribute Name="email"><saml:AttributeValue>alice@example.com'
        "</saml:AttributeValue></saml:Attribute></saml:AttributeStatement>"
        "</saml:Assertion></samlp:Response>"
    ).encode()
    encoded = base64.b64encode(unsigned_xml).decode("ascii")

    # Make the toolkit blow up loudly if it *is* invoked — proves the
    # escape hatch path really bypassed it.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "OneLogin_Saml2_Auth must not be invoked on escape-hatch path"
        )

    monkeypatch.setattr(auth_route, "OneLogin_Saml2_Auth", _fail_if_called)
    monkeypatch.setattr(
        auth_route,
        "_resolve_or_provision_external_user",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        auth_route, "_sync_saml_groups_if_enabled", AsyncMock(return_value=None)
    )

    request = _saml_callback_request(
        encoded,
        saml_security='{"wantAssertionsSigned": false, "wantMessagesSigned": false}',
    )
    request.app.state.jwt_manager.create_jwt_token = MagicMock(return_value="fake-jwt")

    with caplog.at_level(logging.WARNING, logger="gpustack.routes.auth"):
        await auth_route.saml_callback.__wrapped__(request=request, session=MagicMock())

    # Loud, unmissable warning fired on the request that took this path.
    assert any(
        "signature verification is disabled" in rec.message.lower()
        and "production" in rec.message.lower()
        for rec in caplog.records
    )
