import json
import httpx
import logging
import jwt
from jwt.algorithms import RSAAlgorithm
from gpustack.config.config import Config
from typing import Annotated, Dict, Optional
from fastapi import APIRouter, Form, Request, Response
from gpustack.api.exceptions import (
    InvalidException,
    UnauthorizedException,
    BadRequestException,
)
from gpustack.schemas.users import UpdatePassword
from gpustack.schemas.users import User, AuthProviderEnum
from gpustack.security import (
    JWTManager,
    get_secret_hash,
    verify_hashed_secret,
)
from gpustack import envs
from gpustack.api.auth import (
    SESSION_COOKIE_NAME,
    OIDC_ID_TOKEN_COOKIE_NAME,
    SSO_LOGIN_COOKIE_NAME,
    authenticate_user,
)
from gpustack.server.deps import CurrentUserDep, SessionDep
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from fastapi.responses import RedirectResponse
from lxml import etree
from gpustack.utils.convert import safe_b64decode, inflate_data
from urllib.parse import urlencode

from gpustack.utils.network import use_proxy_env_for_url

router = APIRouter()
timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=10.0)
logger = logging.getLogger(__name__)


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


async def init_saml_auth(request: Request):
    """
    Initialize SAML authentication configuration.
    """
    config: Config = request.app.state.server_config
    form_data = await request.form()
    form_dict = dict(form_data)
    saml_settings = {
        "strict": True,
        "sp": {
            "entityId": config.saml_sp_entity_id,  # sp_entityId
            "assertionConsumerService": {
                "url": config.saml_sp_acs_url,  # callback url
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "singleLogoutService": {
                "url": config.saml_sp_slo_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": config.saml_sp_x509_cert,  # SP public key
            "privateKey": config.saml_sp_private_key,  # sp privateKey
        },
        "idp": {
            "entityId": config.saml_idp_entity_id,  # idp_entityId
            "singleSignOnService": {
                "url": config.saml_idp_server_url,  # server url
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "singleLogoutService": {
                "url": config.saml_idp_logout_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
            },
            "x509cert": config.saml_idp_x509_cert,  # idp public key
        },
        "security": json.loads(config.saml_security),
    }  # Signature configuration
    req = {
        "http_host": request.client.host,
        "script_name": request.url.path,
        "get_data": dict(request.query_params),
        "post_data": form_dict,
    }
    return OneLogin_Saml2_Auth(req, saml_settings)


# SAML login and callback endpoints


@router.get("/saml/login")
async def saml_login(request: Request):
    auth = await init_saml_auth(request)
    return RedirectResponse(url=auth.login())


@router.api_route("/saml/callback", methods=["GET", "POST"])
async def saml_callback(request: Request, session: SessionDep):
    logger.debug("Invoke saml callback.")
    try:
        if request.method == "GET":
            query = dict(request.query_params)
            SAMLResponse = query['SAMLResponse']
            decoded = safe_b64decode(SAMLResponse)
            xml_bytes = inflate_data(decoded)
        else:
            form_data = await request.form()
            form_dict = dict(form_data)
            SAMLResponse = form_dict.get('SAMLResponse')
            xml_bytes = safe_b64decode(SAMLResponse)

        root = etree.fromstring(xml_bytes)
        name_id = root.find('.//{*}NameID').text
        ns = {'saml': 'urn:oasis:names:tc:SAML:2.0:assertion'}
        attributes = {}
        attributes['name_id'] = name_id
        for attr in root.xpath('//saml:Attribute', namespaces=ns):
            attr_name = attr.get('Name')
            values = [v.text for v in attr.xpath('saml:AttributeValue', namespaces=ns)]
            attributes[attr_name] = values[0] if len(values) == 1 else values

        config: Config = request.app.state.server_config

        if config.external_auth_name:
            # If external_auth_name is set, use it as username.
            username = get_saml_attributes(
                config, attributes, config.external_auth_name
            )
        else:
            # Try email or name_id for username if external_auth_name is not set.
            for key in ["email", "emailaddress", "name_id", "nameidentifier"]:
                username = get_saml_attributes(config, attributes, key)
                if username:
                    break
            else:
                raise Exception(message="No valid username found in saml attributes")

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

        # determine whether the user already exists
        user = await User.first_by_field(
            session=session, field="username", value=username
        )
        # create user
        if not user:
            user_info = User(
                username=username,
                full_name=full_name,
                avatar_url=avatar_url,
                hashed_password="",
                is_admin=False,
                is_active=not config.external_auth_default_inactive,
                source=AuthProviderEnum.SAML,
                require_password_change=False,
            )
            await User.create(session, user_info)
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
        )
        response.set_cookie(
            key=SSO_LOGIN_COOKIE_NAME,
            value="true",
            httponly=True,
            max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
            expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        )
    except Exception as e:
        logger.error(f"SAML callback error: {str(e)}")
        raise UnauthorizedException(message=str(e))

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


# OIDC login and callback endpoints


@router.get("/oidc/login")
async def oidc_login(request: Request):
    config: Config = request.app.state.server_config
    authorization_endpoint = config.openid_configuration["authorization_endpoint"]
    authUrl = (
        f'{authorization_endpoint}?response_type=code&'
        f'client_id={config.oidc_client_id}&'
        f'redirect_uri={config.oidc_redirect_uri}&'
        f'scope=openid profile email&state=random_state_string'
    )

    return RedirectResponse(url=authUrl)


@router.get("/oidc/callback")
async def oidc_callback(request: Request, session: SessionDep):
    logger.debug("Invoke oidc callback.")
    config: Config = request.app.state.server_config
    query = dict(request.query_params)
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
    async with httpx.AsyncClient(timeout=timeout, trust_env=use_proxy_env) as client:
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
    # determine whether the user already exists
    user = await User.first_by_field(session=session, field="username", value=username)
    # create user
    if not user:
        user_info = User(
            username=username,
            full_name=full_name,
            avatar_url=avatar_url,
            hashed_password="",
            is_admin=False,
            is_active=not config.external_auth_default_inactive,
            source=AuthProviderEnum.OIDC,
            require_password_change=False,
        )
        await User.create(session, user_info)
    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=username,
    )
    response = RedirectResponse(url='/')
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
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
            )
            response.set_cookie(
                key=SSO_LOGIN_COOKIE_NAME,
                value="true",
                httponly=True,
                max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
            )
    except Exception as e:
        logger.warning(f"Failed to set id_token cookie: {str(e)}")
    return response


# Local authentication endpoints


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    session: SessionDep,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
):
    user = await authenticate_user(session, username, password)
    user_name = user.username
    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=user_name,
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
    )


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
    sso_login = request.cookies.get(SSO_LOGIN_COOKIE_NAME)
    content = json.dumps({"logout_url": external_logout_url}) if sso_login else ""
    resp = Response(content=content, media_type="application/json")
    resp.delete_cookie(key=SESSION_COOKIE_NAME)
    resp.delete_cookie(key=OIDC_ID_TOKEN_COOKIE_NAME)
    resp.delete_cookie(key=SSO_LOGIN_COOKIE_NAME)
    return resp


@router.post("/update-password")
async def update_password(
    session: SessionDep,
    user: CurrentUserDep,
    update_in: UpdatePassword,
):
    if not verify_hashed_secret(user.hashed_password, update_in.current_password):
        raise InvalidException(message="Incorrect current password")

    hashed_password = get_secret_hash(update_in.new_password)
    patch = {"hashed_password": hashed_password, "require_password_change": False}
    await user.update(session, patch)
