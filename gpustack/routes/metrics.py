import os
from fastapi import APIRouter, Request
from gpustack.config.config import get_global_config
from gpustack.server.deps import CurrentUserDep
import yaml

from gpustack.utils.metrics import get_builtin_metrics_config_file_path

router = APIRouter()


@router.get("/default-config")
async def get_default_metrics_config(user: CurrentUserDep):
    builtin_metrics_config_path = get_builtin_metrics_config_file_path()
    with open(builtin_metrics_config_path, "r") as f:
        return yaml.safe_load(f)


@router.get("/config")
async def get_metrics_config(user: CurrentUserDep):
    data_dir = get_global_config().data_dir
    custom_metrics_config_path = f"{data_dir}/custom_metrics_config.yaml"

    builtin_metrics_config_path = get_builtin_metrics_config_file_path()
    file_path = (
        custom_metrics_config_path
        if os.path.exists(custom_metrics_config_path)
        else builtin_metrics_config_path
    )

    with open(file_path, "r") as f:
        return yaml.safe_load(f)


@router.post("/config")
async def update_metrics_config(user: CurrentUserDep, request: Request):
    data_dir = get_global_config().data_dir
    custom_metrics_config_path = f"{data_dir}/custom_metrics_config.yaml"

    new_config = await request.json()
    with open(custom_metrics_config_path, "w") as f:
        yaml.safe_dump(new_config, f)
    return {"status": "ok"}
