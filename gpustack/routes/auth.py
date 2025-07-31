import os
import json
import httpx
from typing import Annotated
from fastapi import APIRouter, Form, Request, Response
from pydantic import BaseModel
from gpustack.api.exceptions import InvalidException, UnauthorizedException
from gpustack.schemas.users import UpdatePassword
from gpustack.schemas.users import User
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
from gpustack.utils.process import safe_b64decode, inflate_data

router = APIRouter()
timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=10.0)
# get authentication configuration
authentication_info = json.loads(os.getenv('GPUSTACK_EXTERNAL_AUTH', '{}'))


# Initialize Saml authentication configuration
def init_saml_auth(request: Request):
    saml_settings = {
        "strict": True,
        "sp": {
            "entityId": authentication_info.get("sp_entityId"),  # sp_entityId
            "assertionConsumerService": {
                "url": authentication_info.get("sp_asc_url"),  # callback url
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": authentication_info.get("sp_x509cert", ""),  # SP public key
            "privateKey": authentication_info.get("sp_privateKey", "")  # sp privateKey
        },
        "idp": {
            "entityId": authentication_info.get("idp_entityId"),  # idp_entityId
            "singleSignOnService": {
                "url": authentication_info.get("idp_server_url"),  # server url
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": authentication_info.get("idp_x509cert", "")  # idp public key
        },
        "security": authentication_info.get("security", {})}  # Signature configuration
    req = {
        "http_host": request.client.host,
        "script_name": request.url.path,
        "get_data": dict(request.query_params),
        "post_data": request.form()
    }
    return OneLogin_Saml2_Auth(req, saml_settings)


# saml login
@router.get("/saml/login")
async def saml_login(request: Request):
    auth = init_saml_auth(request)
    return RedirectResponse(url=auth.login())


@router.get("/saml/callback")
async def saml_callback(request: Request, session: SessionDep):
    query = dict(request.query_params)
    SAMLResponse = query['SAMLResponse']
    decoded = safe_b64decode(SAMLResponse)
    xml_data = inflate_data(decoded).decode('utf-8-sig')
    root = etree.fromstring(xml_data)
    name_id = root.find('.//{*}NameID').text
    ns = {'saml': 'urn:oasis:names:tc:SAML:2.0:assertion'}
    attributes = {}
    attributes['name_id'] = name_id
    for attr in root.xpath('//saml:Attribute', namespaces=ns):
        attr_name = attr.get('Name')
        values = [v.text for v in attr.xpath('saml:AttributeValue', namespaces=ns)]
        attributes[attr_name] = values[0] if len(values) == 1 else values
    username = attributes.get(authentication_info['username'])
    if '+' not in authentication_info['full_name']:
        full_name = attributes.get(authentication_info['full_name'])
    else:
        full_name = ' '.join([attributes.get(v.strip()) for v in authentication_info['full_name'].split('+')])
    # determine whether the user already exists
    user = await User.first_by_fields(session=session,
                                      fields={"username": username,
                                              "source": authentication_info.get('type')})
    # create user
    if not user:
        user_info = User(
            username=username,
            full_name=full_name,
            hashed_password="",
            is_admin=False,
            source=authentication_info.get('type'),
            require_password_change=False,
        )
        await User.create(session, user_info)
    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=username,
    )
    response = RedirectResponse(url='/#/login?sso=saml')
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response


# oidc login
@router.get("/oidc/login")
async def oidc_login():
    authUrl = f'{authentication_info.get("base_entrypoint")}auth?response_type=code&' \
              f'client_id={authentication_info.get("CLIENT_ID")}&' \
              f'redirect_uri={authentication_info.get("redirect_uri")}&' \
              f'scope=openid profile email&state=random_state_string'

    return RedirectResponse(url=authUrl)


@router.get("/oidc/callback")
async def oidc_callback(request: Request,session: SessionDep):
    query = dict(request.query_params)
    code = query['code']
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": authentication_info.get('CLIENT_ID'),
        "client_secret": authentication_info.get('CLIENT_SECRET'),
        "redirect_uri": authentication_info.get('redirect_uri')
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            token_res = await client.request(
                "POST", authentication_info.get('base_entrypoint') + "token", data=data
            )
            res_data = json.loads(token_res.text)
            if "access_token" not in res_data:
                raise UnauthorizedException(message=res_data['error_description'])
            token = res_data['access_token']
            headers = {'Authorization': f'Bearer {token}'}
            user_res = await client.request(
                'get', authentication_info.get('base_entrypoint') + "userinfo", headers=headers
            )
            user_data = json.loads(user_res.text)
            username = user_data.get(authentication_info['username'])
            if '+' not in authentication_info['full_name']:
                full_name = user_data.get(authentication_info['full_name'])
            else:
                full_name = ' '.join([user_data.get(v.strip()) for v in authentication_info['full_name'].split('+')])
        except Exception as e:
            raise UnauthorizedException(message=str(e))
    # determine whether the user already exists
    user = await User.first_by_fields(session=session,
                                      fields={"username": username,
                                              "source": authentication_info.get('type')})
    # create user
    if not user:
        user_info = User(
            username=username,
            full_name=full_name,
            hashed_password="",
            is_admin=False,
            source=authentication_info.get('type'),
            require_password_change=False,
        )
        await User.create(session, user_info)
    jwt_manager: JWTManager = request.app.state.jwt_manager
    access_token = jwt_manager.create_jwt_token(
        username=username,
    )
    response = RedirectResponse(url='/#/login?sso=oidc')
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
        expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    session: SessionDep,
    username: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = ""
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
