import logging
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
import requests
import yaml

from gpustack.schemas.models import (
    ModelBase,
)
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelTemplate(BaseModel):
    name: str
    description: Optional[str] = None
    home: Optional[str] = None
    icon: Optional[str] = None
    template: ModelBase

    model_config = ConfigDict(protected_namespaces=())


model_catalog: List[ModelTemplate] = []


def get_model_catalog() -> List[ModelTemplate]:
    return model_catalog


def init_model_catalog(model_catalog_file: Optional[str]):
    global model_catalog
    try:
        if model_catalog_file is None:
            model_catalog_file = get_builtin_model_catalog_file()

        with open(model_catalog_file, "r") as f:
            model_catalog = yaml.safe_load(f)
            logger.debug(
                f"Loaded {len(model_catalog)} templates from model catalog: {model_catalog_file}"
            )
    except Exception as e:
        raise Exception(f"Failed to load model catalog: {e}")


def get_builtin_model_catalog_file() -> str:
    file_name = "model-catalog-huggingface.yaml"
    if not can_access("https://huggingface.co") and can_access("https://modelscope.cn"):
        logger.info("Cannot access huggingface.co, using built-in ModelScope catalog.")
        file_name = "model-catalog-modelscope.yaml"

    return pkg_resources.files("gpustack.catalogs").joinpath(file_name)


def can_access(url: str) -> bool:
    try:
        requests.get(url, timeout=3)
        return True
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to connect to {url}: {e}")
        return False
