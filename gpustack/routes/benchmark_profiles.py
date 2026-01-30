from fastapi import APIRouter
from gpustack.server.deps import CurrentUserDep
from gpustack.utils.compat_importlib import pkg_resources
import yaml


router = APIRouter()


@router.get("/default-config")
async def get_default_profiles_config(user: CurrentUserDep):
    builtin_profiles_config_path = get_builtin_profiles_config_file_path()
    with open(builtin_profiles_config_path, "r") as f:
        return yaml.safe_load(f)


def get_builtin_profiles_config_file_path() -> str:
    profiles_config_file_name = "profiles_config.yaml"
    profiles_config_file_path = str(
        pkg_resources.files("gpustack.assets.profiles_config").joinpath(
            profiles_config_file_name
        )
    )
    return profiles_config_file_path
