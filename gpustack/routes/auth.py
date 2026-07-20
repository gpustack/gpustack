import base64
import functools
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import logging
import jwt
from jwt.algorithms import RSAAlgorithm
from sqlalchemy.exc import IntegrityError
from sqlmodel import delete
from gpustack.config.config import Config
from gpustack.utils import captcha as captcha_util
from typing import Annotated, Dict, List, Optional
from fastapi import APIRouter, Form, Request, Response
from gpustack.api.exceptions import (
    ConflictException,
    InvalidException,
    NotFoundException,
    UnauthorizedException,
    BadRequestException,
    TooManyRequestsException,
)
from gpustack.schemas.login_captcha import LoginCaptchaNonce
from gpustack.schemas.users import UpdatePassword
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.users import User, AuthProviderEnum
from gpustack.security import JWTManager
from gpustack import envs
from gpustack.api.auth import (
    SESSION_COOKIE_NAME,
    OIDC_ID_TOKEN_COOKIE_NAME,
    OIDC_STATE_COOKIE_NAME,
    SSO_LOGIN_COOKIE_NAME,
    authenticate_user,
    client_ip_getter,
)
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, SessionDep
from gpustack.server.passwords import change_password, is_password_change_required
from gpustack.server.services import sync_user_group_memberships
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from fastapi.responses import RedirectResponse
from lxml import etree
from starlette.concurrency import run_in_threadpool
from gpustack.utils.convert import safe_b64decode, inflate_data
from urllib.parse import quote, urlencode, urlparse

from gpustack.ssl_context import make_ssl_context
from gpustack.utils.network import use_proxy_env_for_url
from gpustack.utils.rate_limit import KeyedRateLimiter

router = APIRouter()
timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=10.0)
logger = logging.getLogger(__name__)

# The encrypted challenge is valid long enough to complete the form while
# keeping the replay window bounded.
CAPTCHA_TOKEN_TTL_SECONDS = 5 * 60
CAPTCHA_SESSION_COOKIE_NAME = "gpustack_captcha_session"
_CAPTCHA_BINDING_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")

_captcha_issue_limiter = KeyedRateLimiter(max_requests=30, window_seconds=60)
_captcha_audio_limiter = KeyedRateLimiter(max_requests=10, window_seconds=60)
_login_ip_limiter = KeyedRateLimiter(max_requests=30, window_seconds=5 * 60)
_login_account_limiter = KeyedRateLimiter(max_requests=10, window_seconds=5 * 60)


def _issue_captcha(request: Request, length: int, binding: str) -> Dict[str, str]:
    """Generate an image and encrypt its answer into an opaque token."""
    code, image = captcha_util.generate_captcha(length)
    jwt_manager: JWTManager = request.app.state.jwt_manager
    captcha_id = captcha_util.encrypt_challenge(
        secret_key=jwt_manager.secret_key,
        code=code,
        nonce=secrets.token_urlsafe(16),
        binding=binding,
    )
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "captcha_id": captcha_id,
        "image": f"data:image/png;base64,{encoded}",
    }


def _decode_bound_challenge(request: Request, captcha_id: str):
    """Decode a challenge and require the browser-bound HttpOnly cookie."""
    if not captcha_id:
        raise BadRequestException(message="CAPTCHA is required")

    jwt_manager: JWTManager = request.app.state.jwt_manager
    try:
        challenge = captcha_util.decrypt_challenge(
            secret_key=jwt_manager.secret_key,
            token=captcha_id,
            ttl_seconds=CAPTCHA_TOKEN_TTL_SECONDS,
        )
    except captcha_util.InvalidCaptchaToken:
        raise BadRequestException(message="Invalid or expired CAPTCHA") from None

    binding = request.cookies.get(CAPTCHA_SESSION_COOKIE_NAME, "")
    if not binding or not hmac.compare_digest(challenge.binding, binding):
        raise BadRequestException(message="Invalid or expired CAPTCHA")
    return challenge


async def _consume_captcha_nonce(nonce: str) -> None:
    """Atomically spend a nonce in the shared database ledger."""
    now = datetime.now(timezone.utc)
    nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    async with async_session() as session:
        await session.exec(
            delete(LoginCaptchaNonce).where(LoginCaptchaNonce.expires_at <= now)
        )
        session.add(
            LoginCaptchaNonce(
                nonce_hash=nonce_hash,
                expires_at=now + timedelta(seconds=CAPTCHA_TOKEN_TTL_SECONDS + 30),
            )
        )
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise BadRequestException(message="CAPTCHA has already been used") from None


async def _verify_captcha(request: Request, captcha_id: str, captcha: str) -> None:
    """Validate a solved CAPTCHA or raise ``BadRequestException``.

    Decrypts the opaque token, compares the answer case-insensitively, and
    burns the nonce so the same token can't be reused.
    """
    if not captcha_id or not captcha:
        raise BadRequestException(message="CAPTCHA is required")

    challenge = _decode_bound_challenge(request, captcha_id)

    # Burn the nonce before checking the answer so a wrong guess still cannot
    # be retried, including when the next request reaches another replica.
    await _consume_captcha_nonce(challenge.nonce)

    # Validate before comparing: compare_digest rejects non-ASCII str values,
    # and unbounded input should not reach a character-by-character scan.
    answer = captcha.strip().lower()
    if (
        not captcha_util.CAPTCHA_MIN_LENGTH
        <= len(answer)
        <= captcha_util.CAPTCHA_MAX_LENGTH
        or any(char.upper() not in captcha_util.CAPTCHA_ALPHABET for char in answer)
        or not hmac.compare_digest(challenge.code, answer)
    ):
        raise BadRequestException(message="Incorrect CAPTCHA")


def _captcha_binding(request: Request) -> str:
    binding = request.cookies.get(CAPTCHA_SESSION_COOKIE_NAME, "")
    if not _CAPTCHA_BINDING_PATTERN.fullmatch(binding):
        return secrets.token_urlsafe(32)
    return binding


def _set_captcha_binding_cookie(
    request: Request, response: Response, binding: str
) -> None:
    response.set_cookie(
        key=CAPTCHA_SESSION_COOKIE_NAME,
        value=binding,
        path="/auth",
        httponly=True,
        max_age=CAPTCHA_TOKEN_TTL_SECONDS + 30,
        expires=CAPTCHA_TOKEN_TTL_SECONDS + 30,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


def _enforce_rate_limit(limiter: KeyedRateLimiter, key: str) -> None:
    retry_after = limiter.check(key)
    if retry_after:
        raise TooManyRequestsException(
            message=f"Too many requests; retry in {retry_after} seconds"
        )


def _validate_login_origin(request: Request) -> None:
    """Reject browser cross-site form posts while allowing non-browser clients."""
    if request.headers.get("sec-fetch-site", "").casefold() == "cross-site":
        raise BadRequestException(message="Cross-site login requests are not allowed")
    origin = request.headers.get("origin")
    if not origin:
        return
    parsed = urlparse(origin)
    request_host = request.headers.get("host", "").casefold()
    if (
        parsed.scheme.casefold() != request.url.scheme.casefold()
        or not parsed.netloc
        or parsed.netloc.casefold() != request_host
    ):
        raise BadRequestException(message="Cross-site login requests are not allowed")


async def decode_and_validate_token(
    client: httpx.AsyncClient, token: str, config: Config
) -> Dict:
    """
    Decode the JWT token without verification and check if required fields are present.

    Args:
        token: token from OIDC provider
        config: Application configuration
    Returns:
        Dictionary containing decoded token data
    """
    jwks_uri = config.openid_configuration["jwks_uri"]
    jwks_res = await client.get(jwks_uri)
    jwks = jwks_res.json()

    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid", None)

    public_key = None
    if kid:
        for key in jwks['keys']:
            if key['kid'] == kid:
                public_key = RSAAlgorithm.from_jwk(json.dumps(key))
                break
    else:
        public_key = RSAAlgorithm.from_jwk(json.dumps(jwks['keys'][0]))

    if public_key is None:
        raise UnauthorizedException(message="Public key not found in JWKS")

    claims = jwt.decode(
        token,
        public_key,
        algorithms=['RS256'],
        options={"verify_aud": False, "verify_iss": False},
    )
    return claims


async def get_oidc_user_data(
    client: httpx.AsyncClient, token_res, config: Config
) -> Dict:
    """
    Retrieve user data from OIDC token or userinfo endpoint.

    By default, it uses the userinfo endpoint (standard OIDC).
    If `oidc_skip_userinfo` is set to True in config, it retrieves data from the ID token.

    Args:
        client: HTTP client for making requests
        token_res: The token response from OIDC provider
        config: Application configuration

    Returns:
        Dictionary containing user data
    """
    user_data = None
    if not isinstance(token_res, Dict):
        raise InvalidException(message="Invalid token response")

    if config.oidc_skip_userinfo:
        tokens = []
        if access_token := token_res.get("access_token", None):
            tokens.append(access_token)
        if id_token := token_res.get("id_token", None):
            tokens.append(id_token)
        for token in tokens:
            try:
                user_data = await decode_and_validate_token(client, token, config)
                if user_data:
                    break
            except Exception as e:
                logger.warning(f"Token decoding/validation failed: {str(e)}")
    else:
        token = token_res.get("access_token", "")
        userinfo_endpoint = config.openid_configuration["userinfo_endpoint"]
        headers = {'Authorization': f'Bearer {token}'}
        userinfo_res = await client.get(userinfo_endpoint, headers=headers)
        if userinfo_res.status_code == 200:
            user_data = userinfo_res.json()
        else:
            raise UnauthorizedException(
                message="Failed to fetch user info from userinfo endpoint"
            )
    if not user_data:
        raise UnauthorizedException(message="Failed to retrieve valid user data")
    return user_data


# Browser-facing SSO callbacks (CAS / OIDC / SAML) are reached via a
# full-page IdP redirect: a JSON exception raised inside them would
# land the user on a raw error page rather than the login form.
# Failures are surfaced via ``/login?error=<code>`` instead, so the
# login UI can read the code and render an actionable toast.
#
# Two codes are recognised today:
#   * ``source_conflict`` — same username collides with a different
#     auth source; user needs an admin to link or convert.
#   * ``auth_failed`` — anything else (bad ticket, expired state,
#     IdP unreachable, malformed response, …). Generic by design:
#     the underlying detail goes to ``logger`` for diagnosis rather
#     than into a URL exposed to an unauthenticated caller.
SOURCE_CONFLICT_LOGIN_URL = "/login?error=source_conflict"
AUTH_FAILED_LOGIN_URL = "/login?error=auth_failed"


# ``303 See Other`` so a browser arriving via POST (SAML's POST
# binding) switches to GET on the login page; ``302`` would
# technically allow the browser to re-POST.
def _source_conflict_redirect() -> RedirectResponse:
    return RedirectResponse(url=SOURCE_CONFLICT_LOGIN_URL, status_code=303)


def _auth_failed_redirect() -> RedirectResponse:
    return RedirectResponse(url=AUTH_FAILED_LOGIN_URL, status_code=303)


def _sso_callback_errors_to_redirect(fn):
    """Decorator: turn auth failures inside an SSO callback into a
    redirect to ``/login?error=<code>``.

    ``ConflictException`` maps to ``source_conflict`` (specific
    actionable case). Any other exception — typed auth errors,
    malformed-IdP-response errors, even programming errors — maps to
    the generic ``auth_failed`` code and is logged with stacktrace so
    the cause is recoverable from server-side logs without exposing
    the original message to the browser.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ConflictException:
            return _source_conflict_redirect()
        except Exception as exc:
            # ``HTTPException`` subclasses store the description on
            # ``.message`` and don't pass it to ``Exception.__init__``,
            # so ``str(exc)`` (and thus the traceback's tail line) is
            # empty. Pull the descriptive text out explicitly so
            # operators reading logs see e.g. "The Assertion of the
            # Response is not signed and the SP requires it" instead
            # of a bare exception type.
            detail = getattr(exc, "message", None) or str(exc) or type(exc).__name__
            logger.exception("SSO callback %s failed: %s", fn.__name__, detail)
            return _auth_failed_redirect()

    return wrapper


async def _resolve_external_user(
    session, username: str, expected_source: AuthProviderEnum
) -> Optional[User]:
    """Find an existing USER principal by ``name`` for an SSO login and
    enforce that it originated from the expected provider.

    The lookup is scoped to ``kind=USER`` so a same-named ORG / GROUP
    row can't be mistaken for the calling user. Returning ``None`` lets
    the caller fall through to provisioning a fresh row.

    The ``source`` match is a security check, not a convenience one:
    without it, an attacker who controls ``username`` on *any* enabled
    IdP could hand themselves the session of an existing local-admin
    / OIDC / SAML user with the same name. The mismatch message
    stays silent about *which* provider owns the name so it doesn't
    leak that to an unauthenticated caller.
    """
    user = await User.first_by_fields(
        session=session, fields={"name": username, "kind": PrincipalType.USER}
    )
    if user and user.source != expected_source:
        raise ConflictException(
            message=(
                "An account with this username already exists from a different "
                "authentication source. Please contact an administrator to "
                "link or convert it."
            )
        )
    return user


async def _resolve_or_provision_external_user(
    session,
    username: str,
    expected_source: AuthProviderEnum,
    *,
    display_name: str = "",
    avatar_url: Optional[str] = None,
    is_active: bool = True,
):
    """Resolve an SSO-callback user, provisioning a new principal row
    when the username is new.

    :class:`ConflictException` from :func:`_resolve_external_user`
    propagates unchanged; the SSO callback wrapper
    :func:`_sso_callback_errors_to_redirect` converts it into a
    browser-friendly redirect. The three callbacks share this helper
    so the resolve-or-create idiom (and the source-mismatch check
    inside it) stays identical across providers.
    """
    user = await _resolve_external_user(session, username, expected_source)
    if user is not None:
        return user
    user_info = User(
        kind=PrincipalType.USER,
        name=username,
        display_name=display_name or username,
        avatar_url=avatar_url,
        is_admin=False,
        is_active=is_active,
        source=expected_source,
    )
    return await User.create(session, user_info)


def _saml_response_signature_kinds(xml_bytes: bytes) -> "tuple[bool, bool]":
    """Report which SAML signature elements are present on the raw
    response — as ``(response_signed, assertion_signed)``.

    Presence-only inspection: does *not* validate the signature, only
    checks the DOM shape. The toolkit does the actual crypto check
    once the corresponding ``wantMessagesSigned`` / ``wantAssertionsSigned``
    flag is set, and rejects a faked ``<ds:Signature>`` element there.

    The toolkit's two ``wantX`` knobs are directional — "reject if
    this position is unsigned" — so there's no direct way to express
    "either position signed is fine". Different IdPs sign different
    combinations, so the callback flips the flag matching whichever
    signature is actually present.

    XXE-safe parser: this is the callback's first read of attacker-
    controlled bytes, and needs the same hardening the toolkit applies
    internally.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        root = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError:
        return (False, False)
    ns = {
        "ds": "http://www.w3.org/2000/09/xmldsig#",
        "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
    }
    response_signed = root.find("ds:Signature", ns) is not None
    assertion_signed = root.find(".//saml:Assertion/ds:Signature", ns) is not None
    return (response_signed, assertion_signed)


def _saml_unsigned_escape_hatch(config: Config) -> bool:
    """Whether the operator has explicitly opted the callback out of
    signature verification via ``--saml-security``.

    Only the exact combination of *both* ``wantAssertionsSigned`` and
    ``wantMessagesSigned`` set to ``False`` counts — a missing key or
    a half-opt-out doesn't. Intended purely for local IdP-integration
    testing (e.g. Keycloak with all three signing switches off) where
    turning on signing is inconvenient; enabling this mode makes the
    /auth/saml/callback endpoint accept forged SAMLResponses from any
    client.
    """
    try:
        operator_security = json.loads(config.saml_security)
    except (TypeError, ValueError):
        return False
    return (
        operator_security.get("wantAssertionsSigned") is False
        and operator_security.get("wantMessagesSigned") is False
    )


def _extract_saml_attributes_unsigned(xml_bytes: bytes) -> Dict:
    """Pull NameID + attributes out of the assertion without any
    signature check. Used by the testing escape hatch — see
    :func:`_saml_unsigned_escape_hatch`.

    XXE-safe parser so "signature off" doesn't double as an XXE hole.

    Output shape matches what the toolkit's ``get_attributes()``
    yields after we collapse single-item lists, so the callback code
    downstream doesn't need to branch: bare string for single-valued
    attributes, list for multi-valued.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.fromstring(xml_bytes, parser=parser)
    ns = {"saml": "urn:oasis:names:tc:SAML:2.0:assertion"}

    nameid_elem = root.find(".//saml:NameID", ns)
    nameid = nameid_elem.text if nameid_elem is not None else ""

    attributes: Dict = {"name_id": nameid}
    for attr in root.iterfind(".//saml:Attribute", ns):
        attr_name = attr.get("Name")
        if not attr_name:
            continue
        values = [
            v.text
            for v in attr.iterfind("saml:AttributeValue", ns)
            if v.text is not None
        ]
        # Multiple ``<Attribute Name="X">`` elements with the same
        # Name are valid multi-valued SAML — merge values into one
        # list rather than overwriting.
        existing = attributes.get(attr_name)
        if existing is None:
            attributes[attr_name] = values[0] if len(values) == 1 else values
        else:
            merged = existing if isinstance(existing, list) else [existing]
            merged.extend(values)
            attributes[attr_name] = merged
    return attributes


def _saml_settings(config: Config, *, xml_bytes: Optional[bytes] = None) -> Dict:
    """Assemble the OneLogin toolkit's SP+IdP settings dict.

    The toolkit ships with both ``wantAssertionsSigned`` and
    ``wantMessagesSigned`` defaulted to ``False``, and ``strict=True``
    on its own doesn't enforce either — so an unsigned (i.e. forged)
    assertion would pass. Two-step policy on those flags:

    * **Adaptive detection** (callback path only, when ``xml_bytes``
      is given): flip ``wantMessagesSigned`` / ``wantAssertionsSigned``
      to match whichever ``<ds:Signature>`` is actually present. Lets
      IdPs that sign only the outer ``<Response>`` and IdPs that sign
      only the inner ``<Assertion>`` both work without any operator
      config. An operator's explicit setting always wins — a strict
      deployment can require Assertion signing regardless.

    * **Floor** (applied last): if neither ends up True, force
      ``wantAssertionsSigned=True``. Layered strictness (encrypted
      NameID, both-signed, …) is available via ``--saml-security``.
    """
    operator_security = json.loads(config.saml_security)
    security = dict(operator_security)
    if xml_bytes is not None:
        response_signed, assertion_signed = _saml_response_signature_kinds(xml_bytes)
        if "wantMessagesSigned" not in operator_security and response_signed:
            security["wantMessagesSigned"] = True
        if "wantAssertionsSigned" not in operator_security and assertion_signed:
            security["wantAssertionsSigned"] = True
    # Floor: if neither ``wantX`` ends up True, force
    # ``wantAssertionsSigned=True`` so an unsigned response is refused.
    if not security.get("wantAssertionsSigned") and not security.get(
        "wantMessagesSigned"
    ):
        security["wantAssertionsSigned"] = True
    # Default-tolerate duplicate ``<Attribute Name="X">`` elements in
    # the assertion (values are merged into one list). SAML allows
    # multi-valued attributes as either repeated ``<AttributeValue>``
    # or repeated ``<Attribute>`` — Keycloak's default mappers emit
    # the latter (role list + role name mappers both write ``Role``),
    # and the toolkit's out-of-box behaviour is to reject that.
    # Rejecting isn't a security control, it's a compatibility knob:
    # operators can still opt into strict mode via ``--saml-security``.
    security.setdefault("allowRepeatAttributeName", True)
    return {
        "strict": True,
        "sp": {
            "entityId": config.saml_sp_entity_id,
            "assertionConsumerService": {
                "url": config.saml_sp_acs_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "singleLogoutService": {
                "url": config.saml_sp_slo_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": config.saml_sp_x509_cert,
            "privateKey": config.saml_sp_private_key,
        },
        "idp": {
            "entityId": config.saml_idp_entity_id,
            "singleSignOnService": {
                "url": config.saml_idp_server_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "singleLogoutService": {
                "url": config.saml_idp_logout_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": config.saml_idp_x509_cert,
        },
        "security": security,
    }


async def init_saml_auth(request: Request):
    """
    Initialize SAML authentication configuration.
    """
    config: Config = request.app.state.server_config
    form_data = await request.form()
    req = {
        "http_host": request.client.host,
        "script_name": request.url.path,
        "get_data": dict(request.query_params),
        "post_data": dict(form_data),
    }
    return OneLogin_Saml2_Auth(req, _saml_settings(config))


# SAML login and callback endpoints


@router.get("/saml/login")
async def saml_login(request: Request):
    auth = await init_saml_auth(request)
    return RedirectResponse(url=auth.login())


@router.api_route("/saml/callback", methods=["GET", "POST"])
@_sso_callback_errors_to_redirect
async def saml_callback(request: Request, session: SessionDep):
    logger.debug("Invoke saml callback.")
    config: Config = request.app.state.server_config

    # Normalise the SAMLResponse to plain base64 (HTTP-POST binding
    # shape) so the OneLogin toolkit can consume it via
    # ``request_data['post_data']``. The HTTP-Redirect binding (GET)
    # additionally DEFLATE-compresses the assertion, which the
    # toolkit doesn't unwrap on its own — inflate here and re-encode.
    if request.method == "GET":
        raw = request.query_params.get("SAMLResponse", "")
        xml_bytes = inflate_data(safe_b64decode(raw))
        saml_response = base64.b64encode(xml_bytes).decode("ascii")
    else:
        form_data = await request.form()
        saml_response = form_data.get("SAMLResponse", "")
        xml_bytes = safe_b64decode(saml_response) if saml_response else b""

    if _saml_unsigned_escape_hatch(config):
        # ``OneLogin_Saml2_Auth`` has a hard-coded "at least one
        # signature element must be present" check (``response.py``
        # ``No Signature found``) that fires regardless of the ``wantX``
        # flags, so the escape hatch has to bypass the toolkit
        # entirely rather than adjust its settings.
        logger.warning(
            "SAML signature verification is disabled via --saml-security "
            "(both wantAssertionsSigned and wantMessagesSigned set to false). "
            "Any client can log in as any user by posting a forged "
            "SAMLResponse. Do NOT use in production."
        )
        attributes = _extract_saml_attributes_unsigned(xml_bytes)
    else:
        # ``current_url`` (used by the toolkit for the ``Destination`` /
        # ``Recipient`` checks) is derived from the operator's
        # configured ``saml_sp_acs_url`` rather than from ``request.url``:
        # behind a reverse proxy or UI dev-server the request the
        # callback sees often has its host/port/scheme rewritten and
        # no longer matches what the IdP signed the assertion for. The
        # ACS URL is what the SP publicly claims to be, so anchoring
        # the check to it is both more reliable and no weaker —
        # Destination binding still prevents a response signed for
        # this SP from being replayed against a *different* SP.
        acs = urlparse(config.saml_sp_acs_url or "")
        req = {
            "http_host": acs.hostname or "",
            "server_port": str(
                acs.port
                if acs.port is not None
                else (443 if acs.scheme == "https" else 80)
            ),
            "https": "on" if acs.scheme == "https" else "off",
            "script_name": acs.path,
            "get_data": dict(request.query_params),
            "post_data": {"SAMLResponse": saml_response},
        }
        saml_auth = OneLogin_Saml2_Auth(
            req, _saml_settings(config, xml_bytes=xml_bytes)
        )
        saml_auth.process_response()

        errors = saml_auth.get_errors()
        if errors:
            # The descriptive reason goes to the server log via the
            # decorator's ``logger.exception``; the browser only sees
            # a generic ``auth_failed`` redirect. Prefer the reason
            # string over the coarse code list when present.
            raise UnauthorizedException(
                message="SAML response validation failed: "
                + (saml_auth.get_last_error_reason() or ", ".join(errors))
            )
        if not saml_auth.is_authenticated():
            raise UnauthorizedException(message="SAML response is not authenticated")

        # Collapse ``dict[str, list[str]]`` (what the toolkit returns)
        # into "bare string for single-valued, list for multi-valued"
        # so downstream code doesn't need to branch on arity — matches
        # the shape ``_extract_saml_attributes_unsigned`` produces.
        attrs_raw = saml_auth.get_attributes()
        attributes = {
            k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in attrs_raw.items()
        }
        attributes["name_id"] = saml_auth.get_nameid()

    if config.external_auth_name:
        # If external_auth_name is set, use it as username.
        username = get_saml_attributes(config, attributes, config.external_auth_name)
        if not username:
            # Operator pointed us at a specific attribute, but it
            # isn't present on this assertion. Bail with a clear
            # message instead of letting ``None`` flow into the
            # resolve / create path.
            raise UnauthorizedException(
                message=(
                    f"Username attribute "
                    f"'{config.external_auth_name}' not found in SAML "
                    f"attributes"
                )
            )
    else:
        # Try email or name_id for username if external_auth_name is not set.
        for key in ["email", "emailaddress", "name_id", "nameidentifier"]:
            username = get_saml_attributes(config, attributes, key)
            if username:
                break
        else:
            raise UnauthorizedException(
                message="No valid username found in saml attributes"
            )

    if config.external_auth_full_name and '+' not in config.external_auth_full_name:
        # If external_auth_full_name is set, use it as user's full name.
        full_name = get_saml_attributes(
            config, attributes, config.external_auth_full_name
        )
    elif config.external_auth_full_name:
        # external_auth_full_name is set with concat symbol '+'.
        full_name = ' '.join(
            [
                get_saml_attributes(config, attributes, v.strip())
                for v in config.external_auth_full_name.split('+')
            ]
        )
    else:
        full_name = ""
        # Try common claims. These are not guaranteed to be present.
        for key in ["displayName", "name"]:
            full_name = get_saml_attributes(config, attributes, key)
            if full_name:
                break

    avatar_url = None
    if config.external_auth_avatar_url:
        avatar_url = get_saml_attributes(
            config, attributes, config.external_auth_avatar_url
        )

    user = await _resolve_or_provision_external_user(
        session,
        username,
        AuthProviderEnum.SAML,
        display_name=full_name or username,
        avatar_url=avatar_url,
        is_active=not config.external_auth_default_inactive,
    )

    await _sync_saml_groups_if_enabled(session, user, attributes, config)

    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=username,
    )
    response = RedirectResponse(url='/', status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    response.set_cookie(
        key=SSO_LOGIN_COOKIE_NAME,
        value="true",
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )

    return response


@router.api_route("/saml/logout/callback", methods=["GET", "POST"])
async def saml_logout_callback(request: Request):
    try:
        auth = await init_saml_auth(request)
        auth.process_slo(False)
    except Exception:
        pass
    response = RedirectResponse(url="/")
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    response.delete_cookie(key=SSO_LOGIN_COOKIE_NAME)
    return response


def get_saml_attributes(
    config: Config, attributes: Dict[str, str], name: str
) -> Optional[str]:
    search_keys = []

    if config.saml_sp_attribute_prefix:
        search_keys.append(config.saml_sp_attribute_prefix + name)

    search_keys.extend(
        [
            f"http://schemas.xmlsoap.org/ws/2005/05/identity/claims/{name}",
            name,
        ]
    )

    for key in search_keys:
        if key in attributes:
            return attributes[key]
    return None


def _group_sync_enabled(config: Config) -> bool:
    """Sync runs only when both ``external_auth_group_sync`` is true
    *and* ``external_auth_groups`` is set. The second check prevents
    the silent footgun where the toggle is flipped but the claim
    name was forgotten — every lookup returns ``None`` → empty list
    → sync authoritatively clears the user's provider-sourced
    memberships. Better to skip-with-log than to delete.
    """
    if not config.external_auth_group_sync:
        return False
    if not config.external_auth_groups:
        logger.warning(
            "external_auth_group_sync is on but external_auth_groups "
            "is unset; skipping group sync."
        )
        return False
    return True


async def _sync_oidc_groups_if_enabled(
    session, user, user_data, config: Config
) -> None:
    """OIDC group-sync wrapper. Extracted out of the callback so the
    callback's cyclomatic complexity doesn't push past flake8's
    ceiling — the sync logic itself is one straight-line block but
    the surrounding callback already has 15+ branches handling
    token exchange, error mapping, claim fallbacks, and user
    provisioning.
    """
    if not _group_sync_enabled(config):
        return
    group_names = _coerce_group_claim(user_data.get(config.external_auth_groups))
    await sync_user_group_memberships(
        session, user.id, AuthProviderEnum.OIDC, group_names
    )
    await session.commit()


async def _sync_saml_groups_if_enabled(
    session, user, attributes: Dict[str, str], config: Config
) -> None:
    """SAML counterpart of ``_sync_oidc_groups_if_enabled``. Same
    rationale: keep the callback's complexity below flake8's
    threshold by lifting the sync wrap into its own function."""
    if not _group_sync_enabled(config):
        return
    raw = get_saml_attributes(config, attributes, config.external_auth_groups)
    group_names = _coerce_group_claim(raw)
    await sync_user_group_memberships(
        session, user.id, AuthProviderEnum.SAML, group_names
    )
    await session.commit()


def _coerce_group_claim(raw) -> List[str]:
    """Normalise an OIDC claim / SAML attribute into a list[str].

    IdPs return groups in three shapes:
    - missing key → ``None`` here; we return ``[]`` (user has no groups
      in this provider). In sync mode this clears any existing
      provider-sourced memberships, which is correct for an
      authoritative IdP.
    - single string → wrap to ``[value]``; if the string contains
      commas it's split (some IdPs concatenate this way).
    - list of strings → trimmed copy.

    Non-string entries are dropped, both at the top level and inside
    a list — Group identifiers are strings, numeric group ids
    shouldn't be silently coerced. Empty / whitespace-only entries
    are also dropped; values are trimmed so accidental surrounding
    whitespace from XML / CSV marshalling can't produce two
    "different" Groups for what the admin sees as the same name.
    We do not lower-case or normalise further — case is significant.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = raw.split(",") if "," in raw else [raw]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        # Non-string scalar (int, bool, ...) — reject; same rule we
        # apply to list members below.
        return []
    out: List[str] = []
    for v in candidates:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if s:
            out.append(s)
    return out


# OIDC login and callback endpoints


@router.get("/oidc/login")
async def oidc_login(request: Request):
    config: Config = request.app.state.server_config
    authorization_endpoint = config.openid_configuration["authorization_endpoint"]
    state = secrets.token_urlsafe(32)
    params = urlencode(
        {
            "response_type": "code",
            "client_id": config.oidc_client_id,
            "redirect_uri": config.oidc_redirect_uri,
            "scope": "openid profile email",
            "state": state,
        }
    )
    authUrl = f'{authorization_endpoint}?{params}'

    response = RedirectResponse(url=authUrl)
    response.set_cookie(
        key=OIDC_STATE_COOKIE_NAME,
        value=state,
        httponly=True,
        max_age=600,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@router.get("/oidc/callback")
@_sso_callback_errors_to_redirect
async def oidc_callback(request: Request, session: SessionDep):
    logger.debug("Invoke oidc callback.")
    config: Config = request.app.state.server_config
    query = dict(request.query_params)

    expected_state = request.cookies.get(OIDC_STATE_COOKIE_NAME)
    received_state = query.get("state")
    if (
        not expected_state
        or not received_state
        or not hmac.compare_digest(expected_state, received_state)
    ):
        raise UnauthorizedException(message="Invalid OIDC state")

    code = query['code']
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": config.oidc_client_id,
        "client_secret": config.oidc_client_secret,
        "redirect_uri": config.oidc_redirect_uri,
    }
    token_endpoint = config.openid_configuration["token_endpoint"]
    use_proxy_env = use_proxy_env_for_url(token_endpoint)
    verify = (
        False if config.external_auth_insecure_skip_tls_verify else make_ssl_context()
    )
    async with httpx.AsyncClient(
        timeout=timeout, verify=verify, trust_env=use_proxy_env
    ) as client:
        try:
            token_res = await client.request("POST", token_endpoint, data=data)
            res_data = json.loads(token_res.text)
            if token_res.status_code != 200:
                raise BadRequestException(
                    message=f"Failed to get token, {res_data['error_description']}"
                )

            # Get user data from token or userinfo endpoint
            user_data = await get_oidc_user_data(client, res_data, config)

            if config.external_auth_name:
                # If external_auth_name is set, use it as username.
                username = user_data.get(config.external_auth_name)
            else:
                # Try common OIDC fields for username if external_auth_name is not set.
                # Ref: https://openid.net/specs/openid-connect-core-1_0.html#rfc.section.18.1.1
                for key in ["email", "sub"]:
                    if key in user_data:
                        username = user_data[key]
                        break
                else:
                    raise UnauthorizedException(
                        message="No valid username found in user data"
                    )

            if (
                config.external_auth_full_name
                and '+' not in config.external_auth_full_name
            ):
                full_name = user_data.get(config.external_auth_full_name)
            elif config.external_auth_full_name:
                full_name = ' '.join(
                    [
                        user_data.get(v.strip())
                        for v in config.external_auth_full_name.split('+')
                    ]
                )
            else:
                full_name = user_data.get("name", "")

            if config.external_auth_avatar_url:
                avatar_url = user_data.get(config.external_auth_avatar_url)
            else:
                avatar_url = user_data.get("picture", None)
        except Exception as e:
            logger.error(f"Get OIDC user info error: {str(e)}")
            raise UnauthorizedException(message=str(e))
    user = await _resolve_or_provision_external_user(
        session,
        username,
        AuthProviderEnum.OIDC,
        display_name=full_name or username,
        avatar_url=avatar_url,
        is_active=not config.external_auth_default_inactive,
    )

    await _sync_oidc_groups_if_enabled(session, user, user_data, config)

    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=username,
    )
    response = RedirectResponse(url='/')
    response.delete_cookie(key=OIDC_STATE_COOKIE_NAME)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    try:
        id_token = res_data.get("id_token")
        if id_token:
            response.set_cookie(
                key=OIDC_ID_TOKEN_COOKIE_NAME,
                value=id_token,
                httponly=True,
                max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
            response.set_cookie(
                key=SSO_LOGIN_COOKIE_NAME,
                value="true",
                httponly=True,
                max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
    except Exception as e:
        logger.warning(f"Failed to set id_token cookie: {str(e)}")
    return response


# CAS (Central Authentication Service) — protocol 2.0 / 3.0
#
# Login flow is a browser redirect to ``{cas_server_url}/login?service=…``;
# the CAS server hands the browser back to ``/auth/cas/callback?ticket=…``,
# we exchange the ticket via ``serviceValidate``, parse the XML response,
# and provision / sign in the user. CAS validation responses come back
# with or without the ``cas:`` XML namespace depending on server version
# — we try both.

_CAS_NS = {"cas": "http://www.yale.edu/tp/cas"}


def _cas_find(elem: etree._Element, local_name: str) -> Optional[etree._Element]:
    """Look up a CAS XML child by local name, namespace-or-not."""
    found = elem.find(f"cas:{local_name}", _CAS_NS)
    if found is None:
        found = elem.find(local_name)
    return found


def _add_cas_attribute(
    target: Dict[str, "str | List[str]"], name: str, value: str
) -> None:
    """Record ``value`` under ``name`` in ``target``. CAS 3.0 emits
    repeated XML elements for multi-valued attributes (``groups``,
    ``roles``, …) — lift the dict entry to a list on the second value
    so we preserve the whole set instead of silently overwriting."""
    existing = target.get(name)
    if existing is None:
        target[name] = value
    elif isinstance(existing, list):
        existing.append(value)
    else:
        target[name] = [existing, value]


def _cas_service_url(request: Request, config: Config) -> str:
    """Build the ``service`` URL sent to the CAS server. The exact
    value must match between ``/login`` and ``serviceValidate``, so
    derive it from one source.

    Preference order: explicit ``cas_callback_url`` override →
    ``{server_external_url}/<callback path>`` (robust behind a reverse
    proxy that rewrites Host or terminates TLS) → ``request.url_for``
    (last-resort derivation from the incoming request, fine for
    direct-access dev / single-host setups).
    """
    if config.cas_callback_url:
        return config.cas_callback_url
    callback_path = request.app.url_path_for("cas_callback")
    if config.server_external_url:
        return f"{config.server_external_url.rstrip('/')}{callback_path}"
    return str(request.url_for("cas_callback"))


async def validate_cas_ticket(
    client: httpx.AsyncClient,
    ticket: str,
    service: str,
    config: Config,
) -> Dict[str, "str | List[str]"]:
    """Exchange a CAS service ticket for the authenticated user's
    attributes. Raises ``UnauthorizedException`` on any failure mode
    (network, malformed XML, ``authenticationFailure`` element, missing
    ``cas:user``).
    """
    validate_endpoint = config.cas_validate_endpoint or "/p3/serviceValidate"
    # Tolerate operators omitting the leading slash on the configured
    # endpoint — otherwise ``{base}{endpoint}`` glues ``/cas`` and
    # ``p3/serviceValidate`` into ``/casp3/serviceValidate``.
    if not validate_endpoint.startswith("/"):
        validate_endpoint = "/" + validate_endpoint
    base = config.cas_server_url.rstrip("/")
    # ``safe=''`` so slashes in the service URL are percent-encoded —
    # strict CAS servers / reverse proxies can reject raw ``/`` inside
    # a query-parameter value.
    validate_url = (
        f"{base}{validate_endpoint}"
        f"?ticket={quote(ticket, safe='')}"
        f"&service={quote(service, safe='')}"
    )
    logger.debug("Validating CAS ticket against %s", validate_url)

    try:
        response = await client.get(validate_url)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise UnauthorizedException(
            message=f"CAS validation HTTP error: {e.response.status_code}"
        )
    except httpx.HTTPError as e:
        raise UnauthorizedException(message=f"CAS validation request failed: {e}")

    try:
        # Parse the raw bytes so the XML declaration's ``encoding=``
        # is respected (httpx-decoded ``.text`` may use the HTTP
        # Content-Type charset or chardet, which can disagree with
        # what the document declares).
        #
        # ``lxml`` resolves entities and loads external DTDs by
        # default — neither is needed for CAS responses, and both
        # widen the XXE attack surface. Disable explicitly.
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, load_dtd=False
        )
        root = etree.fromstring(response.content, parser=parser)
    except etree.XMLSyntaxError as e:
        raise UnauthorizedException(message=f"Failed to parse CAS response: {e}")

    success_elem = _cas_find(root, "authenticationSuccess")
    if success_elem is None:
        failure_elem = _cas_find(root, "authenticationFailure")
        if failure_elem is not None:
            code = failure_elem.get("code", "UNKNOWN")
            message = (failure_elem.text or "Authentication failed").strip()
            raise UnauthorizedException(
                message=f"CAS authentication failed: {code} - {message}"
            )
        raise UnauthorizedException(
            message="CAS authentication failed: invalid response"
        )

    user_data: Dict[str, "str | List[str]"] = {}
    user_elem = _cas_find(success_elem, "user")
    if user_elem is not None and user_elem.text:
        user_data["username"] = user_elem.text.strip()

    attributes_elem = _cas_find(success_elem, "attributes")
    if attributes_elem is not None:
        for child in attributes_elem:
            # ``lxml`` exposes XML comments / processing instructions
            # via a non-string ``tag`` (the ``etree.Comment`` factory
            # itself). Skip those so we don't ``AttributeError`` on
            # ``.split`` and crash the whole login flow.
            if not isinstance(child.tag, str):
                continue
            if not child.text:
                continue
            # Strip ``{namespace}`` prefix from the tag if present so
            # callers can address attributes by their unqualified name.
            local_name = child.tag.split("}", 1)[-1]
            _add_cas_attribute(user_data, local_name, child.text.strip())

    if not user_data.get("username"):
        raise UnauthorizedException(message="No username found in CAS response")

    return user_data


@router.get("/cas/login")
async def cas_login(request: Request):
    config: Config = request.app.state.server_config
    if not config.cas_server_url:
        raise BadRequestException(message="CAS is not configured")
    service = _cas_service_url(request, config)
    base = config.cas_server_url.rstrip("/")
    return RedirectResponse(url=f"{base}/login?service={quote(service, safe='')}")


@router.get("/cas/callback")
@_sso_callback_errors_to_redirect
async def cas_callback(request: Request, session: SessionDep):
    logger.debug("Invoke CAS callback.")
    config: Config = request.app.state.server_config
    if not config.cas_server_url:
        raise BadRequestException(message="CAS is not configured")

    ticket = request.query_params.get("ticket")
    if not ticket:
        raise BadRequestException(message="Missing CAS ticket parameter")

    service = _cas_service_url(request, config)
    use_proxy_env = use_proxy_env_for_url(config.cas_server_url)
    verify = (
        False if config.external_auth_insecure_skip_tls_verify else make_ssl_context()
    )
    async with httpx.AsyncClient(
        timeout=timeout, verify=verify, trust_env=use_proxy_env
    ) as client:
        user_data = await validate_cas_ticket(client, ticket, service, config)

    def _first(value):
        """Pick the first entry when an attribute came back as a list
        (CAS 3.0 multi-valued attributes), otherwise return as-is.
        Username / full name / avatar URL all have single-valued
        semantics — silently collapsing extras is the right call here;
        future group sync will want the full list and read raw."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    username_attr = config.cas_username_attribute
    username = _first(
        user_data.get(username_attr) if username_attr else user_data.get("username")
    )
    if not username:
        raise UnauthorizedException(
            message="Failed to extract username from CAS response"
        )

    full_name = ""
    if config.cas_full_name_attribute:
        full_name = _first(user_data.get(config.cas_full_name_attribute, "")) or ""
    elif "displayName" in user_data:
        full_name = _first(user_data.get("displayName", "")) or ""

    avatar_url: Optional[str] = None
    if config.cas_avatar_attribute:
        avatar_url = _first(user_data.get(config.cas_avatar_attribute))

    # The JWT is keyed by ``username``; CAS has no group-sync step
    # that would need the principal row, so the helper's return value
    # is discarded once the resolve-or-provision side effects have
    # run.
    await _resolve_or_provision_external_user(
        session,
        username,
        AuthProviderEnum.CAS,
        display_name=full_name or username,
        avatar_url=avatar_url,
        is_active=not config.external_auth_default_inactive,
    )

    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(username=username)
    response = RedirectResponse(url="/")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    response.set_cookie(
        key=SSO_LOGIN_COOKIE_NAME,
        value="true",
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


# Local authentication endpoints


@router.get("/captcha")
async def get_captcha(request: Request, response: Response):
    """Issue a graphic CAPTCHA challenge for the local login form.

    Only served when ``enable_login_captcha`` is on; otherwise the login form
    never asks for one, so a request here is a misconfiguration and returns
    404 rather than a usable challenge.
    """
    config: Config = request.app.state.server_config
    if not config.enable_login_captcha:
        raise NotFoundException(message="CAPTCHA is not enabled")
    _enforce_rate_limit(_captcha_issue_limiter, client_ip_getter(request))
    binding = _captcha_binding(request)
    _set_captcha_binding_cookie(request, response, binding)
    response.headers["Cache-Control"] = "no-store"
    return await run_in_threadpool(
        _issue_captcha, request, config.login_captcha_length, binding
    )


@router.post("/captcha/audio")
async def get_captcha_audio(
    request: Request,
    response: Response,
    captcha_id: Annotated[str, Form()] = "",
):
    """Return a lazy WAV alternative for the browser-bound challenge."""
    config: Config = request.app.state.server_config
    if not config.enable_login_captcha:
        raise NotFoundException(message="CAPTCHA is not enabled")
    _enforce_rate_limit(_captcha_audio_limiter, client_ip_getter(request))
    challenge = _decode_bound_challenge(request, captcha_id)
    audio = await run_in_threadpool(captcha_util.generate_audio, challenge.code)
    response.headers["Cache-Control"] = "no-store"
    encoded = base64.b64encode(audio).decode("ascii")
    return {"audio": f"data:audio/wav;base64,{encoded}"}


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    session: SessionDep,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    captcha_id: Annotated[str, Form()] = "",
    captcha: Annotated[str, Form()] = "",
):
    config: Config = request.app.state.server_config
    if config.enable_login_captcha:
        _validate_login_origin(request)
        client_ip = client_ip_getter(request)
        username_hash = hashlib.sha256(
            username.strip().casefold().encode("utf-8")
        ).hexdigest()
        _enforce_rate_limit(_login_ip_limiter, client_ip)
        _enforce_rate_limit(_login_account_limiter, f"{client_ip}:{username_hash}")
        await _verify_captcha(request, captcha_id, captcha)
    user = await authenticate_user(session, username, password)
    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=user.name,
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    if config.enable_login_captcha:
        response.delete_cookie(key=CAPTCHA_SESSION_COOKIE_NAME, path="/auth")


@router.post("/logout")
async def logout(request: Request):
    config: Config = request.app.state.server_config
    external_logout_url = None
    if (
        config.external_auth_type == AuthProviderEnum.OIDC
        and config.openid_configuration
    ):
        end_session_endpoint = config.openid_configuration.get("end_session_endpoint")
        if end_session_endpoint:
            redirect_uri = str(config.server_external_url or request.base_url)
            params = {
                "client_id": config.oidc_client_id,
                "post_logout_redirect_uri": redirect_uri,
                "id_token_hint": request.cookies.get(OIDC_ID_TOKEN_COOKIE_NAME),
            }
            if config.external_auth_post_logout_redirect_key:
                params[config.external_auth_post_logout_redirect_key] = redirect_uri
            query = urlencode({k: v for k, v in params.items() if v})
            external_logout_url = (
                end_session_endpoint if not query else f"{end_session_endpoint}?{query}"
            )
    elif config.external_auth_type == AuthProviderEnum.SAML:
        try:
            auth = await init_saml_auth(request)
            redirect_uri = str(config.server_external_url or request.base_url)
            params = {}
            if config.external_auth_post_logout_redirect_key:
                params[config.external_auth_post_logout_redirect_key] = redirect_uri
            external_logout_url = auth.logout(return_to=redirect_uri)
            query = urlencode({k: v for k, v in params.items() if v})
            if query:
                external_logout_url += f"&{query}"
        except Exception as e:
            logger.error(f"Failed to get SAML logout url: {str(e)}")
            external_logout_url = None
    elif config.external_auth_type == AuthProviderEnum.CAS and config.cas_server_url:
        # CAS Single Sign-Out: the browser hits ``{cas}/logout?service=…``
        # which terminates the CAS session and bounces back to ``service``.
        base = config.cas_server_url.rstrip("/")
        redirect_uri = str(config.server_external_url or request.base_url)
        external_logout_url = f"{base}/logout?service={quote(redirect_uri, safe='')}"
    sso_login = request.cookies.get(SSO_LOGIN_COOKIE_NAME)
    content = json.dumps({"logout_url": external_logout_url}) if sso_login else ""
    resp = Response(content=content, media_type="application/json")
    resp.delete_cookie(key=SESSION_COOKIE_NAME)
    resp.delete_cookie(key=OIDC_ID_TOKEN_COOKIE_NAME)
    resp.delete_cookie(key=SSO_LOGIN_COOKIE_NAME)
    return resp


@router.post("/update-password")
async def update_password(
    request: Request,
    session: SessionDep,
    user: CurrentUserDep,
    update_in: UpdatePassword,
):
    ok = await change_password(
        session, user.id, update_in.current_password, update_in.new_password
    )
    if not ok:
        raise InvalidException(message="Incorrect current password")

    remove_initial_password_file_if_exists(request.app.state.server_config)


# Public auth providers, in :meth:`Config.init_auth` precedence order.
# Drives ``/auth/config`` so a new provider only needs an entry here and
# a matching ``/auth/<type>/login`` route — the UI renders the SSO
# button from the response without any per-provider conditionals.
_EXTERNAL_AUTH_PROVIDERS = (
    AuthProviderEnum.OIDC,
    AuthProviderEnum.SAML,
    AuthProviderEnum.CAS,
)


@router.get("/config")
async def get_auth_config(request: Request, session: SessionDep):
    config: Config = request.app.state.server_config

    external_auth = None
    auth_type = config.external_auth_type
    if auth_type in _EXTERNAL_AUTH_PROVIDERS:
        external_auth = {
            "type": auth_type,
            "login_url": f"/auth/{auth_type.lower()}/login",
        }

    req_dict = {
        "external_auth": external_auth,
        # Drives whether the login form renders the CAPTCHA row and fetches a
        # challenge. The login endpoint enforces it regardless; this just lets
        # the UI avoid asking for a CAPTCHA that would never be checked.
        "captcha_enabled": config.enable_login_captcha,
        # Deprecated: per-provider booleans kept for UI bundles that
        # pre-date the data-driven ``external_auth`` field. Only the
        # providers that were already public are exposed here — newer
        # providers (e.g. CAS) are advertised solely via
        # ``external_auth``. Remove these once we no longer ship UI
        # builds that read them.
        "is_oidc": auth_type == AuthProviderEnum.OIDC,
        "is_saml": auth_type == AuthProviderEnum.SAML,
    }

    # First-login state is keyed off the shared DB ``require_password_change``
    # flag on the bootstrap admin, so every replica reports it consistently
    # behind a load balancer — a file check would not, since the password file
    # only exists on replicas where it was written / mounted. The retrieval
    # command is only advertised while that password file is still present.
    #
    # Scope the lookup to the default ``admin`` account that bootstrap creates,
    # not any admin: extra admins added later must not resurface the guide, and
    # once the default admin is renamed (only possible after logging in) the
    # first-login window is over.
    #
    # FastAPI always injects a session for real requests; direct (non-FastAPI)
    # callers that only need the external-auth config pass ``session=None`` and
    # skip the first-login lookup.
    if session is None:
        return req_dict

    admin = await User.first_by_fields(
        session=session,
        fields={
            "name": "admin",
            "is_admin": True,
            "kind": PrincipalType.USER,
            "is_active": True,
        },
    )
    if admin and await is_password_change_required(session, admin.id):
        command = _get_initial_password_command(config)
        if command:
            req_dict["first_time_setup"] = True
            req_dict["get_initial_password_command"] = command

    return req_dict


def _get_initial_password_command(config: Config) -> Optional[str]:
    """Command an operator runs to retrieve the machine-generated initial admin
    password, or ``None`` when the password file is no longer present.

    In HA the Helm chart mounts a shared Secret at this path on every replica;
    single-node installs write it there when generating the password.
    """
    initial_password_file = Path(config.data_dir) / "initial_admin_password"
    if not initial_password_file.exists():
        return None
    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        pod_name = os.getenv("HOSTNAME", "<pod_name>")
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        namespace = (
            namespace_file.read_text().strip()
            if namespace_file.exists()
            else "<namespace>"
        )
        return f"kubectl exec {pod_name} -n {namespace} -- cat {initial_password_file}"
    elif Path("/.dockerenv").exists():
        return f"docker exec <container_name_or_id> cat {initial_password_file}"
    return f"cat {initial_password_file}"


def remove_initial_password_file_if_exists(config: Config) -> None:
    """Remove the initial admin password file if it exists.

    A shared/read-only mount (e.g. the HA password mounted into every replica)
    may be un-removable; that is fine and not treated as an error. The
    first-login guide hides via the DB ``require_password_change`` flag, not the
    file's presence, so leaving the file in place is harmless.
    """
    initial_password_file = Path(config.data_dir) / "initial_admin_password"
    try:
        initial_password_file.unlink(missing_ok=True)
    except OSError as e:
        logger.debug(f"Left initial password file in place: {e}")
