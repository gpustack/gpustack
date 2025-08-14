import os
from gpustack.config.config import Config
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def register(app: FastAPI):
    ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
    if not os.path.isdir(ui_dir):
        raise RuntimeError(f"directory '{ui_dir}' does not exist")

    for name in ["css", "js", "static"]:
        app.mount(
            f"/{name}",
            StaticFiles(directory=os.path.join(ui_dir, name)),
            name=name,
        )

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(ui_dir, "index.html"))

    # Provide configuration interface
    @app.get("/auth-config")
    async def get_auth_config(request: Request):
        req_dict = {}
        config: Config = request.app.state.server_config
        auth_type = 'Local'
        if config.external_auth_type:
            auth_type = config.external_auth_type
        if auth_type.lower() == 'oidc':
            req_dict = {"is_oidc": True, "is_saml": False}
        if auth_type.lower() == 'saml':
            req_dict = {"is_oidc": False, "is_saml": True}

        return req_dict
