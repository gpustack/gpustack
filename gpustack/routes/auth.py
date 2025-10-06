import json
import httpx
import logging
import jwt
from gpustack.config.config import Config
from typing import Annotated, Dict, Optional
from fastapi import APIRouter, Form, Request, Response
from gpustack.api.exceptions import InvalidException, UnauthorizedException
from gpustack.schemas.users import UpdatePassword
from gpustack.schemas.users import User, AuthProviderEnum
from gpustack.security import (
    JWT_TOKEN_EXPIRE_MINUTES,
    JWTManager,
    get_secret_hash,
    verify_hashed_secret,
)
from gpustack.api.auth import SESSION_COOKIE_NAME, authenticate_user
from gpustack.server.deps import CurrentUserDep, SessionDep
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from fastapi.responses import RedirectResponse
from lxml import etree
from gpustack.utils.convert import safe_b64decode, inflate_data

router = APIRouter()
timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=10.0)
logger = logging.getLogger(__name__)


async def get_oidc_user_data(
    client: httpx.AsyncClient, token: str, userinfo_endpoint: str, config: Config
) -> Dict:
    """
    Retrieve user data from OIDC token or userinfo endpoint.

    First attempts to decode the JWT token to get user data directly.
    If the token contains the required fields, validates it against the userinfo endpoint.
    Falls back to fetching user data from the userinfo endpoint if needed.

    Args:
        client: HTTP client for making requests
        token: The access token from OIDC provider
        userinfo_endpoint: URL of the userinfo endpoint
        config: Application configuration

    Returns:
        Dictionary containing user data
    """
    headers = {'Authorization': f'Bearer {token}'}

    # Try to decode the token first and check if required fields are present
    user_data = None
    try:
        # Decode JWT token without verification (we'll validate via userinfo endpoint)
        decoded_token = jwt.decode(token, options={"verify_signature": False})

        # Check if the required user data is in the token
        has_required_data = False
        if config.external_auth_name and config.external_auth_name in decoded_token:
            has_required_data = True
        elif "email" in decoded_token:
            has_required_data = True

        if has_required_data:
            # Validate token by checking with userinfo endpoint
            validation_res = await client.request(
                'get', userinfo_endpoint, headers=headers
            )
            if validation_res.status_code == 200:
                user_data = decoded_token
                logger.debug("Using decoded token data for user information")
            else:
                logger.debug(
                    "Token validation failed, falling back to userinfo endpoint"
                )

    except jwt.DecodeError:
        logger.debug("Failed to decode token, falling back to userinfo endpoint")
    except Exception as e:
        logger.debug(
            f"Error processing token: {str(e)}, falling back to userinfo endpoint"
        )

    # Fallback to userinfo endpoint if token decoding failed or required data not present
    if user_data is None:
        user_res = await client.request('get', userinfo_endpoint, headers=headers)
        user_data = json.loads(user_res.text)

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
            "x509cert": config.saml_sp_x509_cert,  # SP public key
            "privateKey": config.saml_sp_private_key,  # sp privateKey
        },
        "idp": {
            "entityId": config.saml_idp_entity_id,  # idp_entityId
            "singleSignOnService": {
                "url": config.saml_idp_server_url,  # server url
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
            max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
            expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
        )
    except Exception as e:
        logger.error(f"SAML callback error: {str(e)}")
        raise UnauthorizedException(message=str(e))

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
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            token_endpoint = config.openid_configuration["token_endpoint"]
            userinfo_endpoint = config.openid_configuration["userinfo_endpoint"]
            token_res = await client.request("POST", token_endpoint, data=data)
            res_data = json.loads(token_res.text)
            if "access_token" not in res_data:
                raise UnauthorizedException(message=res_data['error_description'])
            token = res_data['access_token']

            # Get user data from token or userinfo endpoint
            user_data = await get_oidc_user_data(
                client, token, userinfo_endpoint, config
            )

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
        max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
    )
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
        max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key=SESSION_COOKIE_NAME)


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
