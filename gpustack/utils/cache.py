import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.parse
from gpustack.config.config import get_global_config
import logging

logger = logging.getLogger(__name__)


def _make_filepath(namespace: str, key: str) -> Path:
    config = get_global_config()
    safe_key = urllib.parse.quote(key, ".")
    return Path(config.cache_dir) / namespace / safe_key


def is_cached(namespace: str, key: str, cache_expiration: Optional[int] = None) -> bool:
    if not namespace or not key:
        return False

    dst_file = _make_filepath(namespace, key)
    if not dst_file.exists():
        return False

    if cache_expiration and time.time() - os.path.getmtime(dst_file) > cache_expiration:
        return False

    return True


def load_cache(
    namespace: str, key: str, cache_expiration: Optional[int] = None
) -> Tuple[Optional[str], bool]:
    try:
        if not is_cached(namespace, key, cache_expiration):
            return None, False

        dst_file = _make_filepath(namespace, key)
        with dst_file.open('r') as f:
            data = f.read()
        return data, True
    except Exception as e:
        logger.warning(f"Failed to load cache {namespace} {key} : {e}")
        return None, False


def save_cache(namespace: str, key: str, value: str) -> bool:
    if not namespace or not key:
        return False

    dst_file = _make_filepath(namespace, key)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst_file.open('w') as f:
            f.write(value)
        return True
    except Exception as e:
        logger.warning(f"Failed to save cache {namespace} {key} : {e}")
        return False
