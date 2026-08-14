import logging
from datetime import timedelta
from typing import Optional, Annotated, Tuple
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import APIRouter, Request, Response, Depends
from gpustack.api.exceptions import (
    NotFoundException,
    ForbiddenException,
    UnauthorizedException,
    BadRequestException,
)
from gpustack.server.services import ModelRouteService, UserService
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.users import User
from gpustack.schemas.models import AccessPolicyEnum
from gpustack.server.deps import SessionDep
from gpustack.api.auth import (
    api_key_header_auth,
    basic_auth,
    cookie_auth,
    bearer_auth,
    authenticate_gateway_asserted_identity,
    authenticate_request,
    credentials_exception,
    gateway_token_auth,
    inference_scope,
    GATEWAY_ASSERTED_KEY_REF_HEADER,
    GATEWAY_DOWNSTREAM_CONN_HEADER,
)
from gpustack.security import JWTManager, AUTH_CACHE_HEADER
from gpustack.server.gateway_auth_reconciler import gateway_ref_indexable

logger = logging.getLogger(__name__)

router = APIRouter()


# DO NOT make these module-level singletons — see issue #5121. Raising the
# same instance repeatedly keeps every prior call-stack alive on the
# instance's ``__traceback__`` (the instance lives forever as a module
# attribute), retaining every ``frame.f_locals`` (Request, AsyncSession,
# user, api_key, ...) and burning ~30 KB per request. Always raise a freshly
# constructed instance instead.
def model_name_missing_exception() -> BadRequestException:
    return BadRequestException(
        message="Missing 'model' field",
        is_openai_exception=True,
    )


def model_not_found_exception() -> NotFoundException:
    return NotFoundException(
        message="Model not found",
        is_openai_exception=True,
    )


def _build_consumer(access_key: Optional[str], user: User) -> str:
    """The caller identity that travels upstream as ``X-Mse-Consumer``.

    Single definition on purpose: the gateway plugin rebuilds this string
    locally for PUBLIC routes, where it never calls this endpoint at all, and
    the two must agree byte for byte or access-log attribution and the
    rate-limit consumer dimension silently drift apart.

    The key part is omitted when there is nothing to name: for identities with
    no API key behind them (cookie / basic), and equally for the legacy cluster
    token, whose row carries an empty access key. Testing for emptiness rather
    than for None is what keeps the second case from rendering as a leading
    dot, ``.gpustack-<id>``.
    """
    return ".".join([part for part in [access_key, f"gpustack-{user.id}"] if part])


async def _resolve_caller(
    request: Request, session: SessionDep
) -> Tuple[Optional[User], Optional[ApiKey], str, bool]:
    """Who is calling, in the two shapes this endpoint now accepts.

    Form B first: the gateway verified the credential itself and asserts the
    identity, so there is nothing to authenticate here and the request carries
    no credential at all. Everything the gateway cannot verify locally arrives
    as form A, which is the original path, unchanged.

    Failing to identify the caller is not an error -- a PUBLIC route serves
    anonymous requests -- so the unauthenticated outcome is a None user with the
    ``'none'`` consumer, and the policy check downstream decides.

    Returns ``(user, api_key, consumer, identity_asserted)``.
    """
    user, api_key = await authenticate_gateway_asserted_identity(request, session)
    if user is not None:
        return (
            user,
            api_key,
            _build_consumer(getattr(api_key, "access_key", None), user),
            True,
        )

    cookie_token = await cookie_auth(request)
    x_api_key = await api_key_header_auth(request)
    try:
        user = await authenticate_request(
            request=request,
            basic_credentials=await basic_auth(request),
            bearer_token=await bearer_auth(request),
            x_api_key=x_api_key,
            cookie_token=cookie_token,
            session=session,
        )
    except UnauthorizedException:
        logger.debug("Unauthenticated request to server token-auth endpoint")
        return None, None, "none", False
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        raise e
    api_key = getattr(request.state, "api_key", None)
    return (
        user,
        api_key,
        _build_consumer(getattr(api_key, "access_key", None), user),
        False,
    )


@router.get("")
async def server_auth(
    request: Request,
    session: SessionDep,
):
    jwt_manager: JWTManager = request.app.state.jwt_manager
    cached = request.headers.get(AUTH_CACHE_HEADER)
    request_model = request.headers.get("x-higress-llm-model")
    request_conn = request.headers.get(GATEWAY_DOWNSTREAM_CONN_HEADER, "")
    if cached and request_model:
        try:
            data = jwt_manager.decode_jwt_data(cached)
            # This is where an unresolved caller's marker is honoured on its
            # fallback pass: the plugin cannot verify a server-signed marker
            # (the two use different keys) so it forwards it here, and the
            # server resolves it. A resolved caller's marker never reaches this
            # point -- the plugin verifies its own and never asks.
            #
            # The marker rides the upstream request, so a worker or third-party
            # provider that received one could otherwise replay it here on a
            # connection of its own. Binding it to the connection it was minted
            # on is what stops that: the plugin sends the current connection on
            # every call, and only the genuine fallback -- an internal redirect
            # on the same client connection -- carries the one the marker names.
            # An empty binding binds nothing, so it is refused rather than
            # matched, which is also the plugin's posture when no address is
            # available.
            marker_conn = data.get("conn") or ""
            if data.get("model") == request_model:
                if marker_conn and marker_conn == request_conn:
                    logger.debug(
                        "Gateway auth-cache hit for consumer %r on model %r; "
                        "honoured on its minting connection, re-authentication "
                        "skipped",
                        data.get("consumer"),
                        request_model,
                    )
                    return Response(
                        status_code=200,
                        headers={"X-Mse-Consumer": data["consumer"]},
                    )
                # Model matches but the connection binding does not, so the
                # marker is not honoured and the request falls through to normal
                # authentication. Two shapes, logged apart because only one is
                # notable: an absent binding is benign -- a marker minted with no
                # source.address, or by a server predating this field during a
                # rolling upgrade -- while a *present but different* connection is
                # what a marker replayed from elsewhere looks like, and should be
                # rare enough in isolation to be worth an INFO line.
                if not marker_conn:
                    logger.debug(
                        "Gateway auth-cache marker for consumer %r on model %r "
                        "carries no connection binding; not honoured",
                        data.get("consumer"),
                        request_model,
                    )
                else:
                    logger.info(
                        "Gateway auth-cache marker for consumer %r on model %r "
                        "presented on a different connection than it was minted "
                        "on; not honoured",
                        data.get("consumer"),
                        request_model,
                    )
        except Exception:
            pass

    user, api_key, consumer, identity_asserted = await _resolve_caller(request, session)

    if user is None:
        gateway_token_auth(request)

    model_name = request.headers.get("x-higress-llm-model")
    if model_name is None or model_name == "":
        logger.debug(
            "Missing x-higress-llm-model header for token authentication",
        )
        raise (
            credentials_exception() if user is None else model_name_missing_exception()
        )
    pair = await ModelRouteService(session=session).get_model_auth_info_by_name(
        model_name
    )
    if pair is None:
        raise credentials_exception() if user is None else model_not_found_exception()
    # ``pair[1]`` is the cluster registration token, no longer read here: see
    # the response headers below.
    policy = pair[0]

    if user is None and policy != AccessPolicyEnum.PUBLIC:
        logger.debug(
            f"Unauthenticated request to access model {model_name} with policy {policy}",
        )
        raise credentials_exception()

    if policy != AccessPolicyEnum.PUBLIC:
        # llm_scope will raise exception if the api key is not allowed to access llm.
        inference_scope(request, user)
        if not await UserService(session).model_allowed_for_user(
            model_name=model_name,
            user_id=user.id,
            api_key=api_key,
        ):
            raise ForbiddenException(
                message=f"Not allowed to access model {model_name}"
            )
    # No ``token`` claim: the upstream credential is held statically by
    # ai-proxy's ``apiTokens`` and no longer travels per request.
    #
    # ``conn`` binds this marker to the connection the plugin observed. It is
    # only consulted for a marker the plugin has to forward here -- an
    # unresolved caller's -- and it is the sole thing the verify path above adds
    # to the model check, so a marker leaked upstream cannot be replayed from
    # another connection. Empty when the plugin had no address to send; such a
    # marker simply never satisfies the verify path, matching the plugin's own
    # "no address, no marker" behaviour.
    cache_token = jwt_manager.create_token(
        {
            "consumer": consumer,
            "model": model_name,
            "conn": request_conn,
        },
        expires_delta=timedelta(minutes=5),
    )
    # Neither the upstream credential nor a cookie override travels here any
    # more. ai-proxy holds the credential statically in ``apiTokens``, and the
    # gateway plugin drops client cookies itself via
    # ``upstream_request.headers_to_remove`` -- it relays no response header it
    # was not told to, so both of these were being sent only to be discarded,
    # the first of them a cluster-wide credential on every authorization.
    headers = {
        "X-Mse-Consumer": consumer,
        AUTH_CACHE_HEADER: cache_token,
    }
    if (
        not identity_asserted
        and api_key is not None
        and api_key.id is not None
        and gateway_ref_indexable(api_key, user)
    ):
        # Form A only, and only for the gateway itself: it hands the plugin a
        # handle it can cache and later re-assert for a credential it cannot
        # verify locally. It must never reach the upstream model -- it would
        # land in the access log -- so it is deliberately absent from the
        # plugin's allowed_upstream_headers.
        #
        # Gated on the same predicate that decides the plugin's ``refs`` table,
        # because a ref the gateway cannot look up does active harm: it mints a
        # marker naming that ref, then rejects its own marker on the fallback
        # pass and falls back to a credential ai-proxy has already replaced.
        headers[GATEWAY_ASSERTED_KEY_REF_HEADER] = str(api_key.id)
    return Response(status_code=200, headers=headers)


async def worker_auth(
    request: Request,
    bearer_token: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_auth)
    ] = None,
    x_api_key: Annotated[Optional[str], Depends(api_key_header_auth)] = None,
):
    token: str = request.app.state.token
    registration_token = request.app.state.config.token
    model_name = request.headers.get("X-Higress-Llm-Model")
    if model_name is None:
        logger.warning("Missing X-Higress-Llm-Model header for token authentication")
        raise credentials_exception()
    token_value = (bearer_token.credentials if bearer_token else None) or x_api_key
    if token_value is None:
        raise credentials_exception()
    if token_value != token and token_value != registration_token:
        raise credentials_exception()
    return Response(
        status_code=200,
        headers={
            "X-Mse-Consumer": "gpustack-server",
        },
    )
