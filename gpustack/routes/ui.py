import os
import json
from fastapi import FastAPI
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
    @app.get("/get_config")
    async def get_config():
        authentication_info = json.loads(os.getenv('GPUSTACK_authentication_info', '{}'))
        req_dict = {}
        if authentication_info.get('type') == 'oidc':
            req_dict = {"base_entrypoint": authentication_info.get('base_entrypoint'),
                        "redirect_uri": authentication_info.get('redirect_uri'),
                        "CLIENT_ID": authentication_info.get('CLIENT_ID'),
                        "is_oidc": True,
                        "is_saml": False
                        }
        if authentication_info.get('type') == 'saml':
            req_dict = {"is_oidc": False,
                        "is_saml": True}

        return req_dict
