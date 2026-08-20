import asyncio
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import aiohttp
from aiocache import cached
from fastapi import Depends, Request, WebSocket
from starlette.datastructures import Headers
from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.db import async_session
from typing import Annotated, Any, Optional, Set, Tuple, List, Dict
from fastapi.security import (
    APIKeyCookie,
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy import update
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    UnauthorizedException,
    HTTPException,
)
from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.schemas.users import User
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import (
    JWTManager,
    new_secret_key_digest,
    secret_key_digest_usable,
    verify_hashed_secret,
    verify_secret_key_digest,
    get_key_pair,
)
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.passwords import verify_password
from gpustack.server.services import APIKeyService, UserService
from gpustack.websocket_proxy.authenticator import (
    Authenticator as WebsocketAuthenticator,
)

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "gpustack_session"
OIDC_ID_TOKEN_COOKIE_NAME = "gpustack_oidc_id_token"
OIDC_STATE_COOKIE_NAME = "gpustack_oidc_state"
SSO_LOGIN_COOKIE_NAME = "gpustack_sso_login"


def auth_cookie_attrs(request: Request, max_age: int) -> Dict[str, Any]:
    """Security attributes shared by every cookie this server sets.

    One source for them, because they are set from a dozen places and a site
    that spells them out by hand is a site that can fall behind. What makes that
    hard to notice is that Starlette's own defaults are ``samesite="lax"`` and
    ``path="/"`` — the same values the explicit call sites pass. A site can
    therefore omit an attribute and still behave identically, right up until one
    of those values stops being the default we want. ``path`` is exactly that
    case once the server learns its mount prefix.

    ``secure`` has no such cover: it defaults to ``False``, so omitting it
    downgrades the cookie outright.

    It is decided per request from the scheme, which reads ``http`` whenever TLS
    terminates at a proxy — uvicorn only honours ``X-Forwarded-Proto`` from
    ``forwarded_allow_ips``, which defaults to loopback and is not configured.
    So this returns the right answer only for a direct HTTPS listener until that
    is addressed; it is still set here so there is one place to fix rather than
    a dozen.

    ``samesite="lax"`` is load-bearing rather than cosmetic: it keeps the session
    cookie out of a cross-site ``<iframe>``, and since the server sends no
    ``X-Frame-Options`` or CSP ``frame-ancestors``, it is the only thing standing
    between a logged-in admin and a clickjacking page. Stated explicitly here so
    that protection does not rest on a framework default.
    """
    return {
        "httponly": True,
        # Max-Age is authoritative in every browser that matters; Expires is
        # sent alongside for ancient clients that only understand that one.
        "max_age": max_age,
        "expires": max_age,
        "samesite": "lax",
        "secure": request.url.scheme == "https",
    }


SYSTEM_USER_PREFIX = "system/"
SYSTEM_WORKER_USER_PREFIX = "system/worker/"
GATEWAY_AUTH_TOKEN_HEADER = "X-GPUStack-Auth-Token"
# Identity the gateway plugin says it has already verified locally, sent
# *instead of* the credential. Both are request headers a client must never be
# able to set: the transformer plugin strips them at priority 810, before
# ext-auth injects the trusted values at 360 in the same AUTHN phase.
GATEWAY_ASSERTED_ACCESS_KEY_HEADER = "X-GPUStack-Access-Key"
GATEWAY_ASSERTED_KEY_REF_HEADER = "X-GPUStack-Key-Ref"
# The client connection the gateway plugin observed for this request, used to
# bind an auth-cache marker to it. The server never derives this -- only the
# plugin can read source.address -- so it embeds whatever the plugin sends at
# mint time and compares against whatever it sends on the pass that presents the
# marker. Trustworthy for the same reason as the two above: the plugin sets it
# fresh from source.address on every authorization call, overwriting any
# client-supplied copy, so it names the connection the client is actually on.
GATEWAY_DOWNSTREAM_CONN_HEADER = "X-GPUStack-Downstream-Conn"
basic_auth = HTTPBasic(auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)
api_key_header_auth = APIKeyHeader(name="X-API-Key", auto_error=False)
cookie_auth = APIKeyCookie(name=SESSION_COOKIE_NAME, auto_error=False)
_gateway_auth_header = APIKeyHeader(name=GATEWAY_AUTH_TOKEN_HEADER, auto_error=False)


# DO NOT make this a module-level singleton — see issue #5121.
# ``raise existing_instance`` writes the current call-stack traceback onto
# the instance's ``__traceback__``. Because the instance is a module-level
# attribute it is never garbage-collected, and its ``__traceback__`` keeps
# every frame in that call stack alive (along with every frame.f_locals,
# i.e. the entire per-request object stack). Always raise a freshly
# constructed instance instead.
def credentials_exception() -> UnauthorizedException:
    return UnauthorizedException(message="Invalid authentication credentials")


def gateway_token_auth(
    request: Request,
    token: Annotated[Optional[str], Depends(_gateway_auth_header)] = None,
):
    """Verify that a request came from the API gateway.

    What this token authorizes grew with the move of authentication to the
    gateway: it used to say only "this request came from the gateway", and it
    now also lets the caller assert an identity to ``/token-auth`` -- see
    :func:`authenticate_gateway_asserted_identity`.

    Holding it is still much less than being able to act as someone. This
    endpoint answers with a consumer name and a marker, not with inference: a
    client that presents the token and an asserted identity learns who the
    server would call them, and nothing more. Reaching a model that way needs a
    marker the gateway will accept, and the gateway signs those with a
    *different* key -- while stripping any identity header a client sends.

    That other key is the one worth guarding. ``auth_cache.signing_key`` sits in
    the same WasmPlugin resource as this token, and forging markers with it is
    impersonation outright. Read access to the gateway namespace is therefore
    closer to platform-level authority than it looks, which is worth knowing
    before granting it -- the API key documentation says so where it tells an
    operator to edit that resource.

    Rotating this token means rotating ``jwt_secret_key``, which also
    invalidates every user session. A dedicated derivation is the change to make
    if it ever needs rotating on its own, the way the marker signing key already
    has one.
    """
    if not token:
        token = request.headers.get(GATEWAY_AUTH_TOKEN_HEADER)
    if not token:
        raise UnauthorizedException(message="Missing authentication token")
    cfg: Config = request.app.state.server_config
    if token != cfg.get_derived_gateway_token():
        raise UnauthorizedException(message="Invalid gateway token")


def client_ip_getter(request: Request) -> str:
    if request.app.state.server_config.gateway_mode != GatewayModeEnum.disabled:
        try:
            gateway_token_auth(request)
            # Prefer X-Real-IP: the edge proxy (Higress/Envoy) sets it to the
            # immediate downstream connection address, which the client cannot
            # spoof.
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip
            # Fall back to X-Forwarded-For "client, proxy1, proxy2". Take the
            # rightmost entry (the address the trusted edge proxy observed) to
            # avoid trusting client-supplied leftmost values.
            xff = request.headers.get("X-Forwarded-For")
            if xff:
                real_ip = xff.split(",")[-1].strip()
                if real_ip:
                    return real_ip
        except UnauthorizedException:
            pass
    return request.client.host if request.client else ""


@asynccontextmanager
async def _optional_session(session: Optional[AsyncSession]):
    """Yield the caller-provided session as-is, or open (and close) a fresh one."""
    if session is not None:
        yield session  # caller owns it, don't close here
    else:
        async with async_session() as s:
            yield s  # opened here, closed on exit


async def get_current_user(
    request: Request,
    basic_credentials: Annotated[
        Optional[HTTPBasicCredentials], Depends(basic_auth)
    ] = None,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
    cookie_token: Annotated[Optional[str], Depends(cookie_auth)] = None,
) -> User:
    # FastAPI dependency entry point. Keep the signature free of any
    # non-``Depends`` parameter (e.g. ``session``) -- FastAPI would try to
    # build a Pydantic field for it and fail route registration. Callers that
    # already hold a session should call ``authenticate_request`` directly.
    return await authenticate_request(
        request,
        basic_credentials=basic_credentials,
        bearer_token=bearer_token,
        x_api_key=x_api_key,
        cookie_token=cookie_token,
    )


async def authenticate_request(
    request: Request,
    basic_credentials: Optional[HTTPBasicCredentials] = None,
    bearer_token: Optional[HTTPAuthorizationCredentials] = None,
    x_api_key: Optional[str] = None,
    cookie_token: Optional[str] = None,
    session: Optional[AsyncSession] = None,
) -> User:
    if hasattr(request.state, "user"):
        user: User = getattr(request.state, "user")
        return user
    api_key: Optional[ApiKey] = None
    user = None
    try:
        server_config: Config = request.app.state.server_config
        if basic_credentials and is_system_user(basic_credentials.username):
            user = await authenticate_system_principal(server_config, basic_credentials)
        elif basic_credentials or cookie_token or bearer_token or x_api_key:
            # Scoped to just the auth lookup (not Depends(get_session)) so the
            # connection/transaction isn't held open for the lifetime of the
            # request -- otherwise a long-lived StreamingResponse (SSE watch,
            # streaming inference proxy) leaves it idle-in-transaction until
            # the stream ends, which can be hours. See #5678.
            async with _optional_session(session) as session:
                if basic_credentials:
                    user = await authenticate_basic_user(session, basic_credentials)
                elif cookie_token:
                    jwt_manager: JWTManager = request.app.state.jwt_manager
                    user = await get_user_from_jwt_token(
                        session, jwt_manager, cookie_token
                    )
                elif bearer_token or x_api_key:
                    token = (
                        bearer_token.credentials if bearer_token else None
                    ) or x_api_key
                    if token is not None:
                        user, api_key = await get_user_from_api_token(session, token)

        if user:
            if not user.is_active:
                raise UnauthorizedException(message="User account is deactivated")
            request.state.user = user
            if api_key is not None:
                request.state.api_key = api_key
            return user

    except HTTPException:
        raise
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to authenticate user: {e}")

    raise credentials_exception()


async def get_admin_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not current_user.is_admin:
        raise ForbiddenException(message="No permission to access")
    return current_user


def is_server_token_principal(principal: Principal) -> bool:
    """The in-memory principal minted by ``authenticate_system_principal``
    for callers presenting the server token via legacy Basic auth
    (pre-2.0 workers). Never persisted — ``id`` stays None — and trusted
    platform-wide, since the server token is the platform-level secret."""
    return (
        principal is not None
        and principal.kind == PrincipalType.SYSTEM
        and principal.id is None
        and principal.name.startswith(SYSTEM_WORKER_USER_PREFIX)
    )


async def get_cluster_principal(
    current_principal: Annotated[Principal, Depends(get_current_user)],
) -> Principal:
    # A SYSTEM principal that the *cluster* claims (1:1 via
    # ``Cluster.system_principal_id``) is the cluster bootstrap
    # account. ``current_principal.cluster`` is the back-populated
    # relationship — eagerly loaded by ``UserService.get_by_username``
    # so this check is a cheap attribute read. The legacy server-token
    # principal has no cluster row but holds the platform secret, so
    # it passes too.
    if current_principal.kind == PrincipalType.SYSTEM and (
        current_principal.cluster is not None
        or is_server_token_principal(current_principal)
    ):
        return current_principal
    return await get_admin_user(current_principal)


async def get_worker_principal(
    current_principal: Annotated[Principal, Depends(get_current_user)],
) -> Principal:
    if current_principal.kind == PrincipalType.SYSTEM and (
        current_principal.worker is not None
        or is_server_token_principal(current_principal)
    ):
        return current_principal
    return await get_admin_user(current_principal)


def is_system_user(username: str) -> bool:
    return username.startswith(SYSTEM_USER_PREFIX)


async def authenticate_system_principal(
    config: Config,
    credentials: HTTPBasicCredentials,
) -> Optional[Principal]:
    if credentials.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        if credentials.password == config.token:
            # In-memory principal — never persisted. SYSTEM kind is what
            # downstream tenant filters gate on; the worker / cluster
            # client gates accept it via ``is_server_token_principal``.
            # NOTE: must NOT set ``is_admin=True`` — the schema rejects
            # the platform-superuser flag on non-USER principals at
            # construction (PrincipalBase.model_post_init), which is
            # also why this principal carries its trust through its
            # SYSTEM kind + unpersisted identity instead.
            return Principal(
                name=credentials.username,
                kind=PrincipalType.SYSTEM,
            )
    return None


async def authenticate_basic_user(
    session: AsyncSession,
    basic_credentials: HTTPBasicCredentials,
) -> Optional[User]:
    try:
        user = await authenticate_user(
            session, basic_credentials.username, basic_credentials.password
        )
        return user
    except Exception:
        return None


def get_access_token(
    bearer_token: Optional[HTTPAuthorizationCredentials],
    oauth2_bearer_token: Optional[str],
    cookie_token: Optional[str],
) -> str:
    if bearer_token:
        return bearer_token.credentials
    elif oauth2_bearer_token:
        return oauth2_bearer_token
    elif cookie_token:
        return cookie_token
    else:
        raise credentials_exception()


async def get_user_from_jwt_token(
    session: AsyncSession, jwt_manager: JWTManager, access_token: str
) -> Optional[User]:
    try:
        payload = jwt_manager.decode_jwt_token(access_token)
        username = payload.get("sub")
    except Exception:
        logger.debug("Failed to decode JWT token")
        return None

    if username is None:
        return None

    try:
        user = await UserService(session).get_by_username(username)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to get user: {e}")

    return user


def parse_hyphen_uuid(value: str) -> Optional[str]:
    if not re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value, re.I
    ):
        return None
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        return None


_backfill_tasks: Set[asyncio.Task] = set()
_backfill_in_flight: Set[int] = set()
# Keys whose stored digest this build cannot parse. Warned about once per key per
# process: the condition is per-request, and a polling worker would otherwise
# flood the log with the same line.
_warned_unusable_digest: Set[int] = set()


def _schedule_secret_key_digest_backfill(api_key: ApiKey, secret_key: str) -> None:
    """Write ``secret_key_digest`` for a key that has no usable one.

    Off the request path on purpose: this is a pure optimization -- a key without
    a digest keeps working through argon2 -- and the first wave after an upgrade
    would otherwise put a DB write in front of a batch of authentications. For the
    same reason nothing here may raise: the caller has already authenticated the
    request, and turning that into a 500 over a missed optimization would be
    absurd.

    Each key converges after one successful verification, so this is a one-time
    cost per key rather than anything the steady state pays.
    """
    try:
        digest = new_secret_key_digest(
            secret_key=secret_key,
            is_custom=api_key.is_custom,
            access_key=api_key.access_key,
        )
        if digest is None or api_key.id is None:
            # Whatever the eligibility test refuses stays on argon2: the legacy
            # cluster token always, and a custom key while
            # ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` is off. Nothing here knows
            # which -- backfilling a custom key is the same code path as
            # backfilling a generated one, and has to stay that way, since the
            # switch can be turned back on and keys created while it was off
            # have to converge without being touched by hand.
            return
        if api_key.id in _backfill_in_flight:
            return
        # Hold a reference: a bare create_task may be collected mid-flight.
        task = asyncio.create_task(
            _backfill_secret_key_digest(
                api_key.id,
                api_key.access_key,
                digest,
                expected_current=api_key.secret_key_digest,
            )
        )
        # Marked in flight only once the task exists, and safe to do after
        # ``create_task`` because the coroutine cannot run -- and so cannot reach
        # its ``finally`` -- before this synchronous function yields.
        _backfill_in_flight.add(api_key.id)
        _backfill_tasks.add(task)
        task.add_done_callback(_backfill_tasks.discard)
    except Exception:
        logger.exception(
            f"Failed to schedule secret key digest backfill for {api_key.id}"
        )


async def _backfill_secret_key_digest(
    api_key_id: int,
    access_key: str,
    digest: str,
    expected_current: Optional[str] = None,
) -> None:
    """Store ``digest`` for ``api_key_id``, unless the row moved under us.

    The update is a compare-and-set against the value the caller read, so
    concurrent authentications of the same key collapse into one effective write,
    and a value another process wrote in the meantime is never clobbered.
    ``expected_current`` is NULL for a key that predates the column and the old
    string when replacing one this build cannot parse -- an unusable value is
    worth nothing, so leaving it in place would pin the key to argon2 forever.

    ``updated_at`` is assigned its own value on purpose: the column carries
    ``onupdate``, which fires on a bulk UPDATE too, and naming it in the SET
    clause is what suppresses that. Otherwise the first wave after an upgrade
    would restamp every API key -- a user-visible change, and one the API sorts
    on -- for a derived column the user never set.
    """
    try:
        async with async_session() as session:
            guard = (
                ApiKey.secret_key_digest.is_(None)
                if expected_current is None
                else ApiKey.secret_key_digest == expected_current
            )
            await session.execute(
                update(ApiKey)
                .where(ApiKey.id == api_key_id, guard)
                .values(secret_key_digest=digest, updated_at=ApiKey.updated_at)
            )
            await session.commit()
            # ``APIKeyService.get_by_access_key`` is cached, so without this the
            # next requests would keep re-verifying through argon2 until the entry
            # expired, and keep rescheduling this backfill.
            await delete_cache_by_key(
                APIKeyService(session).get_by_access_key, access_key
            )
    except Exception:
        # Purely an optimization: the key keeps verifying through argon2, and the
        # next request retries. Keep the traceback -- a failure here is either a
        # DB problem or a bug, and both need the stack to diagnose.
        logger.exception(f"Failed to backfill secret key digest for {api_key_id}")
    finally:
        _backfill_in_flight.discard(api_key_id)


async def get_user_from_api_token(
    session: AsyncSession, token: str
) -> Tuple[Optional[Principal], Optional[ApiKey]]:
    try:
        access_key, secret_key = get_key_pair(token)
        worker_uuid = parse_hyphen_uuid(access_key)
        if worker_uuid is not None:
            # if access_key is a valid uuid, it's legacy worker re-registering with legacy token
            access_key = ""
        access_keys = [access_key]
        # the custom key should have 32 chars access key as it is generated by security.custom_key_hash which will return 32 chars hex string.
        if len(access_key) == 32 and "" not in access_keys:
            # this means it is custom key or legacy worker token, we should also try to find api key with empty access key for backward compatibility
            access_keys.append("")
        api_key: Optional[ApiKey] = None
        for candidate in access_keys:
            api_key = await APIKeyService(session).get_by_access_key(candidate)
            if api_key:
                logger.trace(f"Found API key for access key: {candidate}")
                break
        if api_key is None:
            return None, None
        if api_key.expires_at is not None and api_key.expires_at <= datetime.now(
            timezone.utc
        ):
            return None, None

        # ``get_by_access_key`` returns a detached ApiKey with all columns
        # already loaded, so reading its fields no longer needs the DB
        # connection. Hand the pooled connection back *before* the argon2
        # verify: the verify is deliberately expensive and CPU-bound, and
        # holding a connection idle across it starves the pool under load.
        # rollback() is safe -- this lookup is read-only.
        await session.rollback()

        if secret_key_digest_usable(api_key.secret_key_digest):
            # A few microseconds, and cheap enough to stay on the event loop.
            # Only secrets this server generated ever get a digest, so a fast
            # hash here is not a downgrade -- see security.new_secret_key_digest.
            verified = verify_secret_key_digest(api_key.secret_key_digest, secret_key)
        else:
            # No digest, or one this build cannot check. argon2 is the permanent
            # fallback verifier, so a malformed or unknown-algorithm value costs
            # this request the slow path instead of locking a valid key out.
            if api_key.secret_key_digest and api_key.id not in _warned_unusable_digest:
                _warned_unusable_digest.add(api_key.id)
                logger.warning(
                    f"API key {api_key.id} has an unusable secret_key_digest; "
                    "verifying via argon2 and replacing the value. The column held "
                    "something this version cannot parse."
                )
            # argon2 verification is synchronous and CPU-bound; run it off the
            # event loop so concurrent requests keep flowing instead of
            # serializing behind each hash.
            verified = await asyncio.to_thread(
                verify_hashed_secret, api_key.hashed_secret_key, secret_key
            )
            if verified:
                # The plaintext is in hand exactly here, and only here, for a key
                # that predates the digest column. Fill it in so the next request
                # takes the fast path.
                _schedule_secret_key_digest_backfill(api_key, secret_key)
        if verified:
            user: Optional[User] = await UserService(session).get_by_id(
                user_id=api_key.user_id,
            )
            if user is not None:
                return user, api_key
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to get user: {e}")

    return None, None


# ``api_keys.id`` is a 32-bit serial. A larger value parses fine as a Python
# int and only fails at the driver, as a DataError the caller would surface as
# a 500 -- so the range is part of "is this a key ref", not a database concern.
_MAX_KEY_REF = 2**31 - 1


def _parse_key_ref(key_ref: str) -> Optional[int]:
    try:
        key_id = int(key_ref)
    except ValueError:
        logger.debug("Ignoring asserted identity: key ref is not an id")
        return None
    if not 0 < key_id <= _MAX_KEY_REF:
        logger.debug("Ignoring asserted identity: key ref is out of range")
        return None
    return key_id


async def authenticate_gateway_asserted_identity(
    request: Request, session: AsyncSession
) -> Tuple[Optional[User], Optional[ApiKey]]:
    """Resolve an identity the gateway states it has already authenticated.

    The gateway plugin verifies the secret locally for keys that carry a
    ``secret_key_digest`` and then calls ``/token-auth`` with the identity in a
    header and no credential at all, so this stands in for authentication on
    that path -- the server only evaluates policy afterwards.

    The assertion is honoured *only* behind ``gateway_token_auth``. That check
    is the whole trust boundary here: authentication no longer happens on this
    path, so without it any client could name any ``access_key`` and nothing
    else would stop it. A missing or wrong gateway token makes this a no-op and
    the caller falls through to ordinary credential authentication, which is
    also what a client forging the header gets.

    Returns ``(None, None)`` whenever the assertion cannot be honoured -- no
    header, no gateway token, or a key that is gone, expired, or owned by a
    deactivated principal. The caller then treats the request as
    unauthenticated, which is what the credential path would have done for a
    revoked key anyway; on a PUBLIC route that still means allowed with the
    ``'none'`` consumer, exactly as today.
    """
    # No gateway, no gateway assertions. Defence in depth rather than a check
    # that carries weight on its own -- the gateway token below is the boundary
    # -- but with the gateway disabled these headers can only be a client's, so
    # there is no reason to be reachable. Mirrors client_ip_getter, which gates
    # its own trust in gateway-set headers the same way.
    server_config: Config = request.app.state.server_config
    if server_config.gateway_mode == GatewayModeEnum.disabled:
        return None, None

    access_key = request.headers.get(GATEWAY_ASSERTED_ACCESS_KEY_HEADER)
    key_ref = request.headers.get(GATEWAY_ASSERTED_KEY_REF_HEADER)
    if not access_key and not key_ref:
        return None, None
    try:
        gateway_token_auth(request)
    except UnauthorizedException:
        logger.debug("Ignoring asserted identity: gateway token missing or invalid")
        return None, None

    api_key: Optional[ApiKey] = None
    if access_key:
        api_key = await APIKeyService(session).get_by_access_key(access_key)
    else:
        key_id = _parse_key_ref(key_ref)
        if key_id is None:
            return None, None
        # Detached like the access-key branch above, which goes through a cached
        # service that expunges. Harmless either way under
        # ``expire_on_commit=False``, but two paths returning objects with
        # different session affinity is a difference waiting to matter.
        api_key = await ApiKey.one_by_id(session, key_id)
        if api_key is not None:
            session.expunge(api_key)
    if api_key is None:
        return None, None
    # Unlike the credential path this also rejects a soft-deleted row. The
    # divergence is deliberate and one-directional: it can only reject a key the
    # gateway's own table has gone stale about, never a live one.
    if api_key.deleted_at is not None:
        return None, None
    if api_key.expires_at is not None and api_key.expires_at <= datetime.now(
        timezone.utc
    ):
        return None, None

    user: Optional[User] = await UserService(session).get_by_id(user_id=api_key.user_id)
    if user is None or not user.is_active:
        return None, None
    # A SYSTEM principal is never something the gateway is in a position to
    # assert: those keys are kept out of its tables precisely because one of
    # them -- the cluster registration token -- is what ai-proxy puts into
    # ``Authorization`` on every fallback trip. If one ever reached the tables
    # anyway, the plugin would authenticate that credential as the system
    # identity and assert it here, and policy would then be evaluated for the
    # platform's own subject. Filtering on the gateway side is the primary
    # defence; refusing it here means a lapse there cannot escalate. Nothing
    # legitimate is lost: workers authenticate with a credential, not an
    # assertion.
    if user.kind == PrincipalType.SYSTEM:
        logger.warning(
            "Refusing a gateway-asserted identity for SYSTEM principal "
            f"{user.id}: these keys must never reach the gateway's tables."
        )
        return None, None

    # Same state the credential path publishes, so ``inference_scope`` and the
    # rest of the authorization checks read the key's scope either way.
    request.state.user = user
    request.state.api_key = api_key
    return user, api_key


async def authenticate_user(
    session: AsyncSession, username: str, password: str
) -> User:
    user = await UserService(session).get_by_username(username)
    if not user:
        raise UnauthorizedException(message="Incorrect username or password")

    if not await verify_password(session, user.id, password):
        raise UnauthorizedException(message="Incorrect username or password")

    if not user.is_active:
        raise UnauthorizedException(message="User account is deactivated")

    return user


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
):
    token_value = (bearer_token.credentials if bearer_token else None) or x_api_key
    if not token_value:
        raise UnauthorizedException(message="Invalid authentication credentials")
    token = request.app.state.token
    config: Config = request.app.state.config
    registration_token = config.token
    server_url = config.get_server_url()
    if token_value in [token, registration_token]:
        return
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is not None:
        cred = token_value
        show_len = max(1, min(6, len(cred)))
        masked_token = f"{'*' * (len(cred) - show_len)}{cred[-show_len:]}"
        logger.debug(f"Verifying worker token {masked_token} via server authentication")
        cached_auth = make_auth_token_via_server(request.app.state.http_client_no_proxy)
        is_valid = await cached_auth(server_url, token_value, model_name)
        if is_valid:
            return
    raise UnauthorizedException(message="Invalid authentication credentials")


def make_auth_token_via_server(client: aiohttp.ClientSession):
    @cached(ttl=60)
    async def inner(server_url: str, token: str, model_name: str) -> bool:
        auth_url = f"{server_url.rstrip('/')}/token-auth"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Higress-Llm-Model": model_name,
        }
        try:
            async with client.get(auth_url, headers=headers) as resp:
                return resp.status == 200
        except aiohttp.ClientError as e:
            logger.error(f"Error verifying token via server: {e}")
            return False

    return inner


def get_scopes(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
) -> List[str]:
    api_key: ApiKey = getattr(request.state, "api_key", None)
    if api_key is not None:
        return api_key.scope
    return [PermissionScope.ALL]


def inference_scope(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
):
    scopes = get_scopes(request, _current_user)
    if PermissionScope.ALL not in scopes and PermissionScope.INFERENCE not in scopes:
        raise ForbiddenException(
            message="API key does not have permission to access inference features"
        )


def management_scope(
    request: Request, _current_user: Annotated[User, Depends(get_current_user)]
):
    scopes = get_scopes(request, _current_user)
    if PermissionScope.ALL not in scopes and PermissionScope.MANAGEMENT not in scopes:
        raise ForbiddenException(
            message="API key does not have permission to access management features"
        )


async def authenticate_worker_by_request_headers(
    header_dict: Dict[str, str],
    validate_proxy: Optional[bool] = None,
) -> Optional[Principal]:
    """
    Authenticate a worker based on request headers, used for both WebSocket and non-WebSocket requests.
    For WebSocket requests, the Bearer token is expected in the "Authorization" header.
    For non-WebSocket requests (e.g. HTTP requests to the proxy), the Bearer token can be in either "Authorization"
    or "Proxy-Authorization" header, with "Proxy-Authorization" taking precedence if both are present.
    """
    headers = Headers(header_dict)
    authorization: Optional[str] = None
    if validate_proxy:
        authorization = headers.get("Proxy-Authorization")
    elif validate_proxy is not None:
        authorization = headers.get("Authorization")
    else:
        # if validate_proxy is None, it means we are in a context where both headers could be used (e.g. WebSocket connection from the proxy)
        # in this case we give precedence to Proxy-Authorization if it exists, otherwise fall back to Authorization
        authorization = headers.get("Proxy-Authorization") or headers.get(
            "Authorization"
        )
    async with async_session() as session:
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials) or scheme.lower() != "bearer":
            return None
        bearer_token = HTTPAuthorizationCredentials(
            scheme=scheme, credentials=credentials
        )
        user, _ = await get_user_from_api_token(session, bearer_token.credentials)
        if user is None:
            return None
        # ``user.worker`` is populated by ``UserService.get_by_id`` via
        # selectinload — no extra fetch needed.
        return user


class BearerTokenAuthenticator(WebsocketAuthenticator):
    """Websocket authenticator that verifies bearer tokens via the main server."""

    token: Optional[str]

    def __init__(
        self,
        token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.token = token
        if not self.token and headers:
            parsed_headers = Headers(headers)
            self.token = parsed_headers.get("Authorization", "").replace("Bearer ", "")

    def inject_headers(
        self,
        headers: Dict[str, str],
    ) -> None:
        # No need to inject headers for outgoing connections from the proxy
        for key in list(headers.keys()):
            if key.lower() == "authorization":
                headers.pop(key)
        if self.token:
            headers.setdefault("Authorization", f"Bearer {self.token}")

    async def authenticate(self, websocket: WebSocket) -> bool:
        user = await authenticate_worker_by_request_headers(
            websocket.headers, validate_proxy=False
        )
        if user is None:
            return False
        if user.worker is None:
            logger.debug(
                f"Authenticated user {user.id} with bearer token but it is not associated with any worker"
            )
            return False
        if websocket.headers.get("x-client-id") != user.worker.worker_uuid:
            logger.debug(
                f"Authenticated worker {user.worker.id} with bearer token but client_id {websocket.headers.get('x-client-id')} does not match worker_uuid {user.worker.worker_uuid}"
            )
            return False
        websocket.scope["user"] = user
        return True
