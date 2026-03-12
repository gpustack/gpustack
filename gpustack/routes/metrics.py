import asyncio
import os
from fastapi import APIRouter, Request
from gpustack.config.config import get_global_config
from gpustack.server.deps import CurrentUserDep
import yaml

from gpustack.utils.metrics import get_builtin_metrics_config_file_path

router = APIRouter()

# Cache for parsed YAML configs: {file_path: parsed_data}
_config_cache: dict[str, dict] = {}
# Locks for each file path to ensure async-safe cache access
_cache_locks: dict[str, asyncio.Lock] = {}


def _load_yaml_sync(file_path: str) -> dict:
    """Synchronous YAML loading function to be run in thread pool."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml_sync(file_path: str, data: dict) -> None:
    """Synchronous YAML saving function to be run in thread pool."""
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f)


async def _load_yaml_cached(file_path: str) -> dict:
    """Load YAML file with caching. Async-safe and non-blocking.

    Cache is only invalidated via _invalidate_cache(), typically called after POST updates.
    External file changes will not be detected automatically.
    """
    # Get or create lock for this file path (setdefault is atomic in CPython)
    lock = _cache_locks.setdefault(file_path, asyncio.Lock())

    async with lock:
        # Check if we have a cached version
        if file_path in _config_cache:
            return _config_cache[file_path]

        # Load and cache the file in thread pool to avoid blocking event loop
        data = await asyncio.to_thread(_load_yaml_sync, file_path)

        _config_cache[file_path] = data
        return data


async def _invalidate_cache(file_path: str) -> None:
    """Invalidate cache for a specific file. Async-safe."""
    lock = _cache_locks.setdefault(file_path, asyncio.Lock())
    async with lock:
        _config_cache.pop(file_path, None)


@router.get("/default-config")
async def get_default_metrics_config(user: CurrentUserDep):
    builtin_metrics_config_path = get_builtin_metrics_config_file_path()
    return await _load_yaml_cached(builtin_metrics_config_path)


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

    return await _load_yaml_cached(file_path)


@router.post("/config")
async def update_metrics_config(user: CurrentUserDep, request: Request):
    data_dir = get_global_config().data_dir
    custom_metrics_config_path = f"{data_dir}/custom_metrics_config.yaml"

    new_config = await request.json()

    # Write file in thread pool to avoid blocking event loop
    await asyncio.to_thread(_save_yaml_sync, custom_metrics_config_path, new_config)

    # Invalidate cache after updating the config
    await _invalidate_cache(custom_metrics_config_path)

    return {"status": "ok"}
