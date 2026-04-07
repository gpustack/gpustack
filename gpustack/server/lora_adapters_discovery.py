"""Discover LoRA adapters by base model id (HF Hub API / ModelScope lineage / local cache)."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from sqlalchemy import func, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.models import SourceEnum
from gpustack.routes.proxy import hf_hub_api_headers, replace_hf_endpoint
from gpustack.server.cache import locked_cached
from gpustack.utils.network import use_proxy_env_for_url

logger = logging.getLogger(__name__)

HF_MODELS_API = "https://huggingface.co/api/models"
MODELSCOPE_LINEAGE_LIST = "https://www.modelscope.cn/api/v1/models/lineage/list"

_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

_LOCAL_LORA_NAME_FIELD: Dict[str, str] = {
    SourceEnum.HUGGING_FACE.value: "huggingface_repo_id",
    SourceEnum.MODEL_SCOPE.value: "model_scope_model_id",
    SourceEnum.LOCAL_PATH.value: "local_path",
}


def _normalize_base(base: str) -> str:
    return (base or "").strip().lower()


def _looks_like_hf_or_ms_repo_id(base: str) -> bool:
    s = (base or "").strip()
    return "/" in s and not s.startswith("/")


async def list_local_loras(session: AsyncSession, base: str) -> List[Dict[str, Any]]:
    nb = _normalize_base(base)
    if not nb:
        return []
    stmt = select(ModelFile).where(
        ModelFile.is_lora.is_(True),  # noqa: E712
        or_(
            func.lower(ModelFile.base_model) == nb,
            ModelFile.base_model.is_(None),
            ModelFile.base_model == "",
        ),
    )
    files = (await session.exec(stmt)).all()
    out: List[Dict[str, Any]] = []
    for f in files:
        src = f.source.value if isinstance(f.source, SourceEnum) else f.source
        name_attr = _LOCAL_LORA_NAME_FIELD.get(src)
        if not name_attr:
            continue
        name = getattr(f, name_attr, None)
        if not name:
            continue
        out.append(
            {
                "lora_repo_name": name,
                "source": src,
                "is_local": True,
            }
        )
    seen = set()
    deduped = []
    for item in out:
        key = (item["lora_repo_name"], item["source"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_hf_models_json(payload: Any) -> List[str]:
    if not isinstance(payload, list):
        return []
    ids: List[str] = []
    for m in payload:
        if not isinstance(m, dict):
            continue
        rid = m.get("id") or m.get("modelId")
        if rid and "/" in str(rid):
            ids.append(str(rid))
    return ids


async def list_huggingface_adapter_repos(
    base_model_id: str,
    *,
    q: Optional[str] = None,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """
    List adapter repos for a base model via GET /api/models (same filters as the Hub website).
    """
    base_model_id = (base_model_id or "").strip()
    if not base_model_id or not _looks_like_hf_or_ms_repo_id(base_model_id):
        return []

    params: Dict[str, str] = {
        "filter": f"base_model:{base_model_id}",
        "sort": "trendingScore",
        "limit": str(limit),
    }
    if q and q.strip():
        params["search"] = q.strip()

    url = f"{HF_MODELS_API}?{urlencode(params)}"
    url = replace_hf_endpoint(url)
    headers = hf_hub_api_headers(url)

    try:
        trust_env = use_proxy_env_for_url(url)
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, trust_env=trust_env
        ) as client:
            resp = await client.get(url, headers=headers)
    except Exception as e:
        logger.warning(
            "HuggingFace /api/models request failed for base=%s: %s", base_model_id, e
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            "HuggingFace /api/models returned %s for base=%s",
            resp.status_code,
            base_model_id,
        )
        return []

    try:
        payload = resp.json()
    except Exception as e:
        logger.warning(
            "HuggingFace /api/models invalid JSON for base=%s: %s", base_model_id, e
        )
        return []

    out: List[Dict[str, Any]] = []
    for rid in _parse_hf_models_json(payload)[:limit]:
        out.append(
            {
                "lora_repo_name": rid,
                "source": SourceEnum.HUGGING_FACE.value,
                "is_local": False,
            }
        )
    return out


def _parse_modelscope_lineage_adapter_names(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    ml = data.get("ModelLineageList")
    if not isinstance(ml, list):
        return []
    names: List[str] = []
    for block in ml:
        if not isinstance(block, dict):
            continue
        if block.get("LineAgeType") != "Adapter":
            continue
        for m in block.get("ModelInfoList") or []:
            if isinstance(m, dict) and m.get("Name"):
                names.append(str(m["Name"]))
    return names


async def _modelscope_lineage_put(
    base_model_id: str, body: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return Data payload on success, or None on failure."""
    try:
        trust_env = use_proxy_env_for_url(MODELSCOPE_LINEAGE_LIST)
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, trust_env=trust_env
        ) as client:
            resp = await client.put(
                MODELSCOPE_LINEAGE_LIST,
                json=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
    except Exception as e:
        logger.warning(
            "ModelScope lineage/list failed for base=%s: %s", base_model_id, e
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "ModelScope lineage/list returned %s for base=%s",
            resp.status_code,
            base_model_id,
        )
        return None

    try:
        root = resp.json()
    except Exception as e:
        logger.warning(
            "ModelScope lineage/list invalid JSON for base=%s: %s", base_model_id, e
        )
        return None

    if not isinstance(root, dict) or root.get("Code") != 200:
        msg = root.get("Message") if isinstance(root, dict) else None
        logger.warning(
            "ModelScope lineage/list error for base=%s: %s", base_model_id, msg
        )
        return None

    return root.get("Data") or {}


async def list_modelscope_adapter_repos(
    base_model_id: str,
    *,
    q: Optional[str] = None,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """
    List adapter repos under a base model via PUT .../models/lineage/list (ModelScope web parity).
    """
    base_model_id = (base_model_id or "").strip()
    if not base_model_id or not _looks_like_hf_or_ms_repo_id(base_model_id):
        return []

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    page_number = 1
    max_page_size = 100

    while len(out) < limit:
        page_size = min(max_page_size, limit - len(out))
        body: Dict[str, Any] = {
            "BaseModel": base_model_id,
            "LineageType": "Adapters",
            "PageNumber": page_number,
            "PageSize": page_size,
        }
        if q and q.strip():
            body["Search"] = q.strip()

        data = await _modelscope_lineage_put(base_model_id, body)
        if data is None:
            return out

        new_names = [
            n for n in _parse_modelscope_lineage_adapter_names(data) if n not in seen
        ]
        if not new_names:
            break
        for name in new_names:
            if len(out) >= limit:
                break
            seen.add(name)
            out.append(
                {
                    "lora_repo_name": name,
                    "source": SourceEnum.MODEL_SCOPE.value,
                    "is_local": False,
                }
            )
        if len(new_names) < page_size:
            break
        page_number += 1

    return out


def _discovery_cache_key(f, *args, **kwargs):
    """Cache key for standalone discovery functions (not class methods)."""
    return f"{f.__qualname__}:{args}:{sorted(kwargs.items())}"


@locked_cached(ttl=300, key=_discovery_cache_key)
async def _cached_hf_adapters(
    base_model_id: str, q: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    return await list_huggingface_adapter_repos(base_model_id, q=q, limit=limit)


@locked_cached(ttl=300, key=_discovery_cache_key)
async def _cached_ms_adapters(
    base_model_id: str, q: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    return await list_modelscope_adapter_repos(base_model_id, q=q, limit=limit)


async def list_adapters_for_base(
    session: AsyncSession,
    base_model_id: str,
    *,
    q: Optional[str] = None,
    limit: int = 40,
) -> Dict[str, Any]:
    """
    Merge Hugging Face (Hub /api/models), ModelScope (lineage/list), and local ModelFile LoRAs.

    ``base_model_id`` should be a repo id like ``Qwen/Qwen3-8B`` for remote sources.
    Optional ``q`` maps to Hub ``search`` and ModelScope ``Search``.
    """
    base_model_id = (base_model_id or "").strip()
    loc = await list_local_loras(session, base_model_id)

    if _looks_like_hf_or_ms_repo_id(base_model_id):
        hf_task = _cached_hf_adapters(base_model_id, q, limit)
        ms_task = _cached_ms_adapters(base_model_id, q, limit)
        hf, ms = await asyncio.gather(hf_task, ms_task)
    else:
        hf, ms = [], []
        if base_model_id:
            logger.debug(
                "Skipping HF/ModelScope adapter discovery (expected org/name base id): %s",
                base_model_id,
            )

    # Local first; skip HF/ModelScope items whose lora_repo_name already exists locally.
    merged: List[Dict[str, Any]] = []
    seen = set()
    local_names = {item["lora_repo_name"] for item in loc}
    for item in loc:
        key = (item["lora_repo_name"], item["source"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    for bucket in (hf, ms):
        for item in bucket:
            if item["lora_repo_name"] in local_names:
                continue
            key = (item["lora_repo_name"], item["source"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return {"lora_list": merged}
