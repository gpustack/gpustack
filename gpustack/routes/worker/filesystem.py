import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query

from gpustack.api.auth import worker_auth
from gpustack.schemas.filesystem import FileExistsResponse


router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)

ALLOWED_CONFIG_FILES = {
    "config.json",
    "model_index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "adapter_config.json",
    "preprocessor_config.json",
}


def is_config_file(filename: str) -> bool:
    """Check if a file is a model config file."""
    return filename in ALLOWED_CONFIG_FILES


@router.get("/model-config")
async def read_model_config(path: str = Query(..., description="File path to read")):
    """
    Read and parse a model config file.
    Only model config files (config.json, model_index.json, etc.) can be read for security.
    Returns the parsed configuration object.
    """
    try:
        # Normalize the path
        normalized_path = os.path.normpath(path)

        # Check if path exists
        if not os.path.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        # Check if path is a file
        if not os.path.isfile(normalized_path):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")

        # Check if file is a config file for security
        filename = os.path.basename(normalized_path)
        if not is_config_file(filename):
            raise HTTPException(
                status_code=403,
                detail="Access denied: Only model config files are allowed to be read",
            )

        # Read and parse JSON file
        try:
            with open(normalized_path, "r", encoding="utf-8") as f:
                import json

                config_data = json.load(f)
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to read file: {str(e)}"
            )

        return config_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/file-exists", response_model=FileExistsResponse)
async def file_exists(path: str = Query(..., description="Path to check")):
    """
    Check if a path exists.
    """
    try:
        # Normalize the path
        normalized_path = os.path.normpath(path)

        # Check if path exists
        exists = os.path.exists(normalized_path)
        is_file = os.path.isfile(normalized_path) if exists else False
        is_dir = os.path.isdir(normalized_path) if exists else False

        return FileExistsResponse(
            exists=exists, path=normalized_path, is_file=is_file, is_dir=is_dir
        )

    except Exception as e:
        logger.error(f"Error checking path {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check path: {str(e)}")
