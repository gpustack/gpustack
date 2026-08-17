"""``/token-auth`` with authentication moved to the gateway.

Two request shapes reach this endpoint now:

* **form A** — the credential is forwarded as-is, the server authenticates and
  authorizes. Unchanged, and still the path for anything the gateway cannot
  verify locally.
* **form B** — the gateway asserts an identity it already verified and sends no
  credential at all; the server only evaluates policy.

Both must produce the same ``X-Mse-Consumer``, because the plugin also rebuilds
that string locally for PUBLIC routes and a drift there silently corrupts
access-log attribution and the rate-limit consumer dimension.
"""

import logging
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

from gpustack import envs
from gpustack.api.auth import (
    GATEWAY_ASSERTED_ACCESS_KEY_HEADER,
    GATEWAY_ASSERTED_KEY_REF_HEADER,
    GATEWAY_AUTH_TOKEN_HEADER,
    GATEWAY_DOWNSTREAM_CONN_HEADER,
)
from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes.token import server_auth
from gpustack.schemas.api_keys import PermissionScope
from gpustack.schemas.model_routes import AccessPolicyEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.security import AUTH_CACHE_HEADER, JWTManager

GATEWAY_TOKEN = "derived-gateway-token"
ACCESS_KEY = "3192253c1f4a9b7e"
MODEL = "my-org/qwen3-8b"


def _request(headers):
    request = SimpleNamespace()
    request.state = SimpleNamespace()
    # Case-insensitive, like the real thing: the security dependencies read
    # "Authorization" while the gateway sends it lower-cased.
    request.headers = Headers(headers)
    request.cookies = {}
    request.app = SimpleNamespace(
        state=SimpleNamespace(
            jwt_manager=JWTManager(secret_key="jwt-secret"),
            server_config=SimpleNamespace(
                gateway_mode=None,
                get_derived_gateway_token=lambda: GATEWAY_TOKEN,
            ),
        )
    )
    return request


def _principal(**overrides):
    fields = {"id": 7, "is_active": True, "kind": PrincipalType.USER}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _api_key(**overrides):
    fields = {
        "id": 58,
        "access_key": ACCESS_KEY,
        "user_id": 7,
        "expires_at": None,
        "deleted_at": None,
        "scope": [PermissionScope.ALL],
        "is_custom": False,
        "secret_key_digest": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def route_policy(monkeypatch):
    """Stub the route lookup; the returned setter picks the access policy."""
    state = {"policy": AccessPolicyEnum.AUTHED}

    class _ModelRouteService:
        def __init__(self, session):
            pass

        async def get_model_auth_info_by_name(self, name):
            return state["policy"], "registration-token"

    class _UserService:
        def __init__(self, session):
            pass

        async def model_allowed_for_user(self, model_name, user_id, api_key):
            return True

    monkeypatch.setattr("gpustack.routes.token.ModelRouteService", _ModelRouteService)
    monkeypatch.setattr("gpustack.routes.token.UserService", _UserService)
    return state


def _install_asserted_lookups(monkeypatch, api_key, user):
    class _APIKeyService:
        def __init__(self, session):
            pass

        async def get_by_access_key(self, access_key):
            return api_key

    class _AuthUserService:
        def __init__(self, session):
            pass

        async def get_by_id(self, user_id):
            return user

    async def _one_by_id(session, key_id):
        return api_key

    monkeypatch.setattr("gpustack.api.auth.APIKeyService", _APIKeyService)
    monkeypatch.setattr("gpustack.api.auth.UserService", _AuthUserService)
    monkeypatch.setattr("gpustack.api.auth.ApiKey.one_by_id", _one_by_id)


def _forbid_credential_auth(monkeypatch):
    async def _fail(*args, **kwargs):
        raise AssertionError("form B must not re-authenticate the credential")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _fail)


@pytest.mark.asyncio
async def test_form_b_skips_authentication_and_keeps_the_consumer(
    monkeypatch, route_policy
):
    api_key = _api_key()
    _install_asserted_lookups(monkeypatch, api_key, _principal())
    _forbid_credential_auth(monkeypatch)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                GATEWAY_ASSERTED_ACCESS_KEY_HEADER: ACCESS_KEY,
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    assert response.headers["X-Mse-Consumer"] == f"{ACCESS_KEY}.gpustack-7"
    # The gateway already holds the identity it asserted; handing it a ref back
    # would be noise, and the header must never travel further than the plugin.
    assert GATEWAY_ASSERTED_KEY_REF_HEADER not in response.headers


@pytest.mark.asyncio
async def test_form_a_returns_a_key_ref_and_the_same_consumer(
    monkeypatch, route_policy
):
    # The one shape that gets a ref: a custom key in a deployment that publishes
    # none of them, so ``refs`` is the only plugin table it can ever appear in.
    # With the switch on it is bound for ``keys`` instead and gets no ref -- see
    # ``test_key_ref_is_withheld_from_a_key_the_gateway_cannot_index``.
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", False)
    api_key = _api_key(id=58, is_custom=True)
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                "authorization": f"Bearer gpustack_{ACCESS_KEY}_secret",
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    # Byte-identical to form B above.
    assert response.headers["X-Mse-Consumer"] == f"{ACCESS_KEY}.gpustack-7"
    assert response.headers[GATEWAY_ASSERTED_KEY_REF_HEADER] == "58"


@pytest.mark.asyncio
async def test_assertion_without_a_gateway_token_is_inert(monkeypatch, route_policy):
    """An identity header that did not come from the gateway buys nothing: the
    request is authenticated the ordinary way, which for a caller holding no
    credential is a rejection on a non-PUBLIC route.

    Client-supplied copies of this header should never get this far in the
    first place -- the transformer plugin strips them at priority 810, ahead of
    ext-auth at 360; see ``test_gateway_plugins``. This is the second half of
    that pair, and the reason the two must ship together.
    """
    _install_asserted_lookups(monkeypatch, _api_key(), _principal())
    attempted = []

    async def _authenticate(request, **kwargs):
        attempted.append(True)
        raise UnauthorizedException(message="Invalid authentication credentials")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    with pytest.raises(UnauthorizedException):
        await server_auth(
            _request(
                {
                    GATEWAY_ASSERTED_ACCESS_KEY_HEADER: ACCESS_KEY,
                    "x-higress-llm-model": MODEL,
                }
            ),
            session=object(),
        )

    assert attempted, "the assertion must not short-circuit credential auth"


@pytest.mark.asyncio
async def test_anonymous_request_on_a_public_route_stays_anonymous(
    monkeypatch, route_policy
):
    """The consumer the plugin must reproduce for callers it cannot identify."""
    route_policy["policy"] = AccessPolicyEnum.PUBLIC

    async def _authenticate(request, **kwargs):
        raise UnauthorizedException(message="Invalid authentication credentials")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    assert response.headers["X-Mse-Consumer"] == "none"
    assert GATEWAY_ASSERTED_KEY_REF_HEADER not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("is_custom", [False, True])
async def test_key_ref_is_withheld_from_a_key_the_gateway_cannot_index(
    monkeypatch, route_policy, is_custom
):
    """A key still awaiting its digest is in neither plugin table, so a ref for
    it names nothing the gateway can validate. Handing one over is not merely
    useless: the plugin mints a marker for that ref, overwrites the server's,
    then refuses its own marker on the fallback pass and falls back to a
    credential ai-proxy has already replaced -- which the server then resolves
    as the SYSTEM principal.

    Custom or generated makes no difference while the switch is on: both are
    bound for ``keys`` and get there on this very authentication, which is what
    backfills the digest."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", True)
    api_key = _api_key(id=58, is_custom=is_custom, secret_key_digest=None)
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                "authorization": f"Bearer gpustack_{ACCESS_KEY}_secret",
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    assert GATEWAY_ASSERTED_KEY_REF_HEADER not in response.headers


@pytest.mark.asyncio
async def test_the_response_carries_no_upstream_credential(monkeypatch, route_policy):
    """ai-proxy holds the upstream credential statically in ``apiTokens`` since
    the provider config moved there, and the gateway plugin relays no response
    header it was not told to -- so an Authorization here would be a
    cluster-wide credential emitted on every authorization only to be dropped.
    The cookie override is dead for the same reason: the plugin removes client
    cookies itself on the way upstream."""
    api_key = _api_key(id=58, is_custom=True)
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                "authorization": f"Bearer gpustack_{ACCESS_KEY}_secret",
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    assert "authorization" not in {k.lower() for k in response.headers}
    assert "cookie" not in {k.lower() for k in response.headers}


@pytest.mark.asyncio
async def test_the_marker_carries_no_upstream_credential(monkeypatch, route_policy):
    """Same reason, one indirection further: the marker used to carry the
    credential so a fallback pass could replay it."""
    api_key = _api_key()
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    request = _request(
        {
            GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
            "authorization": f"Bearer gpustack_{ACCESS_KEY}_secret",
            "x-higress-llm-model": MODEL,
            GATEWAY_DOWNSTREAM_CONN_HEADER: "10.42.0.1:64586",
        }
    )
    response = await server_auth(request, session=object())

    claims = request.app.state.jwt_manager.decode_jwt_data(
        response.headers[AUTH_CACHE_HEADER]
    )
    assert set(claims) == {"consumer", "model", "conn"}


CONN = "10.42.0.1:64586"


def _mint_marker(monkeypatch, route_policy, conn=CONN):
    """Drive a legitimate authed call and return the server-signed marker it
    emits -- the same value the plugin puts on the upstream request."""
    api_key = _api_key()
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    async def run():
        response = await server_auth(
            _request(
                {
                    GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                    "authorization": f"Bearer gpustack_{ACCESS_KEY}_secret",
                    "x-higress-llm-model": MODEL,
                    GATEWAY_DOWNSTREAM_CONN_HEADER: conn,
                }
            ),
            session=object(),
        )
        return response.headers[AUTH_CACHE_HEADER], response.headers["X-Mse-Consumer"]

    return run


@pytest.mark.asyncio
async def test_a_marker_short_circuits_on_the_connection_it_was_minted_on(
    monkeypatch, route_policy
):
    """The legitimate fallback: an internal redirect on the same client
    connection. It carries no credential and no gateway token -- the marker and
    the matching connection are the whole of what authorises it."""
    marker, consumer = await _mint_marker(monkeypatch, route_policy)()

    # A fresh authenticate that would reject anything reaching it, to prove the
    # short-circuit returns before credential auth is even attempted.
    async def _reject(request, **kwargs):
        raise AssertionError("short-circuit must return before authentication")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _reject)

    response = await server_auth(
        _request(
            {
                AUTH_CACHE_HEADER: marker,
                "x-higress-llm-model": MODEL,
                GATEWAY_DOWNSTREAM_CONN_HEADER: CONN,
            }
        ),
        session=object(),
    )

    assert response.status_code == 200
    assert response.headers["X-Mse-Consumer"] == consumer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_conn",
    ["10.42.0.1:51999", "", None],
    ids=["other-connection", "empty-conn", "no-conn-header"],
)
async def test_a_marker_is_refused_off_its_connection(
    monkeypatch, route_policy, replay_conn, caplog
):
    """A worker or third-party provider that captured the marker upstream and
    replays it here is on a connection of its own. Without the matching
    connection the short-circuit does not fire, and the request -- carrying no
    real credential -- is rejected. An empty or absent connection is unbindable
    and refused rather than matched, so a marker minted where no address was
    available cannot be replayed either.

    The marker here carries the victim's connection, so presenting it on any
    other -- a different address, or none at all -- is a binding that is present
    but does not match, which surfaces at INFO as the replay signal. (The benign
    case, a marker minted with no binding, is
    ``test_a_marker_minted_without_a_connection_never_short_circuits``.)"""
    marker, _ = await _mint_marker(monkeypatch, route_policy)()

    async def _reject(request, **kwargs):
        raise UnauthorizedException(message="no credential")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _reject)

    headers = {AUTH_CACHE_HEADER: marker, "x-higress-llm-model": MODEL}
    if replay_conn is not None:
        headers[GATEWAY_DOWNSTREAM_CONN_HEADER] = replay_conn

    with caplog.at_level(logging.INFO, logger="gpustack.routes.token"):
        with pytest.raises(UnauthorizedException):
            await server_auth(_request(headers), session=object())

    info = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert any("different connection" in r.message for r in info)


@pytest.mark.asyncio
async def test_a_marker_minted_without_a_connection_never_short_circuits(
    monkeypatch, route_policy, caplog
):
    """When the plugin has no source.address it sends an empty connection, so
    the marker it forwards carries an empty binding. That marker must not
    short-circuit even when replayed with the same empty connection -- an empty
    binding binds nothing, matching the plugin's own 'no address, no marker'
    posture."""
    marker, _ = await _mint_marker(monkeypatch, route_policy, conn="")()

    async def _reject(request, **kwargs):
        raise UnauthorizedException(message="no credential")

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _reject)

    with caplog.at_level(logging.DEBUG, logger="gpustack.routes.token"):
        with pytest.raises(UnauthorizedException):
            await server_auth(
                _request(
                    {
                        AUTH_CACHE_HEADER: marker,
                        "x-higress-llm-model": MODEL,
                        GATEWAY_DOWNSTREAM_CONN_HEADER: "",
                    }
                ),
                session=object(),
            )

    # Benign: an absent binding is DEBUG, never the INFO replay signal.
    assert any(
        "no connection binding" in r.message and r.levelno == logging.DEBUG
        for r in caplog.records
    )
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


@pytest.mark.asyncio
async def test_a_key_with_no_access_key_does_not_leave_a_leading_dot(
    monkeypatch, route_policy
):
    """The legacy cluster token's row carries an empty access key. Testing for
    None rather than for emptiness rendered it as ``.gpustack-<id>``, which the
    access log and the rate-limit consumer dimension then carry verbatim."""
    api_key = _api_key(access_key="")
    user = _principal()

    async def _authenticate(request, **kwargs):
        request.state.api_key = api_key
        request.state.user = user
        return user

    monkeypatch.setattr("gpustack.routes.token.authenticate_request", _authenticate)

    response = await server_auth(
        _request(
            {
                GATEWAY_AUTH_TOKEN_HEADER: GATEWAY_TOKEN,
                "authorization": "Bearer legacy-token",
                "x-higress-llm-model": MODEL,
            }
        ),
        session=object(),
    )

    assert response.headers["X-Mse-Consumer"] == "gpustack-7"


def test_consumer_golden_value():
    """Shared with the plugin's TestFailOpenStillDerivesTheConsumer, in
    extensions/gpustack-ext-auth/failure_test.go.

    The plugin rebuilds this string itself on public routes -- where the server
    never sees the request -- so the two are the same value produced twice, with
    no failure signal if they drift. A change here is a change there.
    """
    from gpustack.routes.token import _build_consumer

    assert (
        _build_consumer("3192253c1f4a9b7e", _principal())
        == "3192253c1f4a9b7e.gpustack-7"
    )
