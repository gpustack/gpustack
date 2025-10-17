from datetime import date
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import urlparse
from pathlib import Path
from fastapi import APIRouter
import requests
import yaml

from gpustack.schemas.model_sets import (
    Catalog,
    ModelSet,
    DraftModel,
    ModelSetPublic,
    ModelSpec,
    ModelRecipe,
)
from gpustack.utils import file
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)

router = APIRouter()


model_catalog: Catalog = None
model_set_recipes: Dict[int, List[ModelRecipe]] = {}
model_set_recipes_by_key: Dict[str, ModelRecipe] = {}


def get_model_sets() -> List[ModelSet]:
    return model_catalog.model_sets if model_catalog else []


def get_catalog_draft_models() -> List[DraftModel]:
    return model_catalog.draft_models if model_catalog else []


def get_model_set_recipes() -> Dict[int, List[ModelRecipe]]:
    return model_set_recipes


def convert_to_public(model_sets: List[ModelSet]) -> List[ModelSetPublic]:
    return [
        ModelSetPublic(**model_set.model_dump(exclude={"templates"}))
        for model_set in model_sets
    ]


def init_model_catalog(model_catalog_file: Optional[str] = None):
    model_sets: List[ModelSet] = []
    try:
        if model_catalog_file is None:
            model_catalog_file = get_builtin_model_catalog_file()

        raw_data = None
        parsed_url = urlparse(model_catalog_file)
        if parsed_url.scheme in ("http", "https"):
            response = requests.get(model_catalog_file)
            response.raise_for_status()
            raw_data = yaml.safe_load(response.text)
        else:
            with open(model_catalog_file, "r") as f:
                raw_data = yaml.safe_load(f)

        global model_catalog
        model_catalog = Catalog(**raw_data)
        logger.debug(
            f"Loaded {len(model_catalog.model_sets)} model sets from model catalog: {model_catalog_file}"
        )
        model_sets = model_catalog.model_sets

        # Use index as the id for each model set
        for idx, model_set in enumerate(model_sets):
            model_set.id = idx + 1

        model_sets = sort_model_sets(model_sets)
        model_catalog.model_sets = convert_to_public(model_sets)
        init_model_set_specs(model_sets)
    except Exception as e:
        raise Exception(f"Failed to load model catalog: {e}")


def sort_model_sets(model_sets: List[ModelSet]) -> List[ModelSet]:
    """
    Sort model sets by order asc, then by release_date desc
    """
    return sorted(
        model_sets,
        key=lambda x: (
            x.order if x.order is not None else float('inf'),
            -(x.release_date.toordinal() if x.release_date else date.min.toordinal()),
        ),
    )


def init_model_set_specs(model_sets: List[ModelSet]):
    global model_set_recipes
    model_set_recipes = {}
    for model_set in model_sets:
        model_set_recipes[model_set.id] = model_set.recipes


def prepare_chat_templates(data_dir: str):
    source_dir = pkg_resources.files("gpustack").joinpath("assets/chat_templates")
    target_dir = Path(data_dir).joinpath("chat_templates")

    if not os.path.exists(source_dir):
        return

    file.copy_with_owner(source_dir, target_dir)


def get_builtin_model_catalog_file() -> str:
    huggingface_url = "https://huggingface.co"
    modelscope_url = "https://modelscope.cn"

    model_catalog_file_name = "model-catalog.yaml"
    if not can_access(huggingface_url) and can_access(modelscope_url):
        model_catalog_file_name = "model-catalog-modelscope.yaml"
        logger.info(f"Cannot access {huggingface_url}, using ModelScope model catalog.")

    return str(pkg_resources.files("gpustack.assets").joinpath(model_catalog_file_name))


def can_access(url: str) -> bool:
    """
    Check if the URL is accessible
    """
    try:
        response = requests.get(url, timeout=3)
        return response.status_code >= 200 and response.status_code < 300
    except requests.RequestException:
        return False


def resolve_model_template(
    model_template: ModelRecipe,
    size: Optional[float],
    quantization: Optional[str],
) -> ModelSpec:
    fields = model_template.model_dump(exclude={"sizes", "quantizations"})

    # Interpolating the size and quantization into template fields
    resolved_fields = {}
    for key, value in fields.items():
        if isinstance(value, str):
            value = resolve_value(value, size, quantization)
        if isinstance(value, list):
            value = [resolve_value(v, size, quantization) for v in value]
        resolved_fields[key] = value

    return ModelSpec(**resolved_fields, size=size, quantization=quantization)


def resolve_value(
    value: str,
    size: Optional[float],
    quantization: Optional[str],
) -> str:
    if size is not None:
        value = value.replace("{size}", format_size(size))
    if quantization:
        value = value.replace("{quantization}", quantization)
    return value


def format_size(size: Optional[float]) -> str:
    if size is None:
        return ""
    # Remove decimal point if the size is an integer
    return str(int(size)) if size.is_integer() else str(size)
