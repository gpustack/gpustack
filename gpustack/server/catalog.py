from datetime import date
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import urlparse
from pathlib import Path
from fastapi import APIRouter
import requests
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_sets import (
    ModelSet,
    DraftModel,
    ModelSetPublic,
    ModelSpec,
)
from gpustack.schemas.catalog_source import (
    CatalogModelEntry,
    KIND_DRAFT,
    KIND_MODEL_SET,
)
from gpustack.utils import file
from gpustack.utils.compat_importlib import pkg_resources

logger = logging.getLogger(__name__)

router = APIRouter()

# Bounded because this read is on the start-up path, and on every leader start.
_CATALOG_FETCH_TIMEOUT = 30


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


# ---- DB-backed catalog accessors ----
# The catalog is materialized into CatalogModelEntry by CatalogSourceController
# (from managed CatalogSource rows). These accessors read that table; the
# catalog is small (hundreds of rows) so loading + Python filtering per request
# is cheap and mirrors the previous in-memory behavior.


def _entry_to_model_set(entry: CatalogModelEntry) -> ModelSet:
    """Reconstruct a ModelSet (specs inline) from an entry, with its stable id."""
    model_set = ModelSet(**entry.payload)
    model_set.id = entry.id
    return model_set


def _entry_to_public(entry: CatalogModelEntry) -> ModelSetPublic:
    """Public model set (no specs) with the stable id and source stamp."""
    data = {
        **entry.payload,
        "id": entry.id,
        "source_name": entry.source_name,
        "source_type": entry.source_type,
    }
    return ModelSetPublic(**data)


async def _model_set_entries(session: AsyncSession) -> List[CatalogModelEntry]:
    return await CatalogModelEntry.all_by_fields(session, {"kind": KIND_MODEL_SET})


async def get_model_sets(session: AsyncSession) -> List[ModelSetPublic]:
    """All catalog model sets as public objects, in catalog display order."""
    entries = await _model_set_entries(session)
    return sort_model_sets([_entry_to_public(entry) for entry in entries])


async def get_model_set_specs(session: AsyncSession, id: int) -> List[ModelSpec]:
    """The specs of one model set, or [] when the id is not a model set."""
    entry = await CatalogModelEntry.one_by_id(session, id)
    if entry is None or entry.kind != KIND_MODEL_SET:
        return []
    return ModelSet(**entry.payload).specs


async def get_catalog_draft_models(session: AsyncSession) -> List[DraftModel]:
    entries = await CatalogModelEntry.all_by_fields(session, {"kind": KIND_DRAFT})
    return [DraftModel(**entry.payload) for entry in entries]


async def get_catalog_spec_by_source_key(
    session: AsyncSession, model_source_key: str
) -> Optional[ModelSpec]:
    """The catalog spec matching a model_source_key.

    Within a set (specs reversed) and across sets in display order, the first
    match wins — prioritizing standard specs, matching the old in-memory lookup.
    """
    model_sets = sort_model_sets(
        [_entry_to_model_set(entry) for entry in await _model_set_entries(session)]
    )
    by_key: Dict[str, ModelSpec] = {}
    for model_set in model_sets:
        for spec in reversed(model_set.specs):
            by_key.setdefault(spec.model_source_key, spec)
    return by_key.get(model_source_key)


def prepare_chat_templates(data_dir: str):
    source_dir = pkg_resources.files("gpustack").joinpath("assets/chat_templates")
    target_dir = Path(data_dir).joinpath("chat_templates")

    if not os.path.exists(source_dir):
        return

    file.copy_with_owner(source_dir, target_dir)


def read_builtin_catalog_text(model_catalog_file: Optional[str] = None) -> str:
    """Read the raw catalog YAML text from a local file or URL.

    An explicit ``model_catalog_file`` (path or URL) wins, else the packaged
    file with HF/ModelScope variant detection. Blocking I/O (and the variant
    network probe) — the leader seed calls it via ``asyncio.to_thread``.
    """
    if model_catalog_file is None:
        model_catalog_file = get_builtin_model_catalog_file()

    parsed_url = urlparse(model_catalog_file)
    if parsed_url.scheme in ("http", "https"):
        response = requests.get(model_catalog_file, timeout=_CATALOG_FETCH_TIMEOUT)
        response.raise_for_status()
        return response.text
    with open(model_catalog_file, "r") as f:
        return f.read()


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
