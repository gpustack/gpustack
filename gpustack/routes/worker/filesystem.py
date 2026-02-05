import asyncio
import json
import logging
import os
import subprocess
import traceback

from fastapi import APIRouter, Depends, HTTPException, Query

from gpustack.api.auth import worker_auth
from gpustack.config.config import get_global_config
from gpustack.schemas.filesystem import (
    FileExistsResponse,
    GGUFParseRequest,
    GGUFParseResponse,
)
from gpustack.schemas.models import Model
from gpustack.scheduler.calculator import (
    _gguf_parser_command,
    _gguf_parser_env,
    GPUOffloadEnum,
    calculate_local_model_weight_size,
)


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


def validate_path_security(path: str, base_path: str = None) -> str:
    """
    Validate path security to prevent directory traversal attacks.

    This function:
    1. Resolves the absolute path (following symlinks)
    2. Validates the path is within the allowed base directory (if provided)
    3. Prevents directory traversal attacks

    Args:
        path: The path to validate
        base_path: Optional base directory that the path must be within

    Returns:
        The validated absolute path

    Raises:
        HTTPException: If the path is invalid or outside the allowed directory

    Security:
    - Uses os.path.realpath to resolve symlinks and get absolute path
    - Validates path is within base_path if provided
    - Prevents directory traversal attacks (../, symlinks, etc.)
    """
    try:
        # Resolve to absolute path, following symlinks
        # This is more secure than os.path.normpath which doesn't resolve symlinks
        resolved_path = os.path.realpath(path)

        # If base_path is provided, ensure the resolved path is within it
        if base_path:
            resolved_base = os.path.realpath(base_path)
            # Use os.path.commonpath to check if resolved_path is under resolved_base
            # This prevents directory traversal attacks
            try:
                common = os.path.commonpath([resolved_base, resolved_path])
                if common != resolved_base:
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: Path is outside allowed directory",
                    )
            except ValueError:
                # Paths are on different drives (Windows)
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: Path is outside allowed directory",
                )

        return resolved_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")


@router.get("/files/model-config")
async def read_model_config(path: str = Query(..., description="File path to read")):
    """
    Read and parse a model config file.
    Only model config files (config.json, model_index.json, etc.) can be read for security.
    Returns the parsed configuration object.

    Security:
    - Uses os.path.realpath to resolve symlinks and prevent directory traversal
    - Only allows reading of whitelisted config files
    - Validates file exists and is a regular file
    """
    try:
        # Validate path security (resolves symlinks, prevents directory traversal)
        validated_path = validate_path_security(path)

        # Check if path exists
        if not os.path.exists(validated_path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        # Check if path is a file
        if not os.path.isfile(validated_path):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")

        # Check if file is a config file for security
        filename = os.path.basename(validated_path)
        if not is_config_file(filename):
            raise HTTPException(
                status_code=403,
                detail="Access denied: Only model config files are allowed to be read",
            )

        # Read and parse JSON file
        try:
            with open(validated_path, "r", encoding="utf-8") as f:
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


@router.get("/files/file-exists", response_model=FileExistsResponse)
async def file_exists(path: str = Query(..., description="Path to check")):
    """
    Check if a path exists.

    Security:
    - Uses os.path.realpath to resolve symlinks and prevent directory traversal
    """
    try:
        # Validate path security (resolves symlinks, prevents directory traversal)
        validated_path = validate_path_security(path)

        # Check if path exists
        exists = os.path.exists(validated_path)
        is_file = os.path.isfile(validated_path) if exists else False
        is_dir = os.path.isdir(validated_path) if exists else False

        return FileExistsResponse(
            exists=exists, path=validated_path, is_file=is_file, is_dir=is_dir
        )

    except Exception as e:
        logger.error(f"Error checking path {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check path: {str(e)}")


@router.get("/files/model-weight-size")
async def get_model_weight_size(
    path: str = Query(..., description="Directory path to scan"),
    is_diffusion: bool = Query(False, description="Whether this is a diffusion model"),
):
    """
    Calculate the total size of model weight files in a directory.

    For LLM models (is_diffusion=False): Scans only the root directory.
    For diffusion models (is_diffusion=True): Scans subdirectories defined in model_index.json.

    Security:
    - Uses os.path.realpath to resolve symlinks and prevent directory traversal
    - Only scans the specified directory (not recursive for LLM, component dirs for diffusion)
    """
    try:
        # Validate path security (resolves symlinks, prevents directory traversal)
        validated_path = validate_path_security(path)

        if not os.path.exists(validated_path):
            raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

        if not os.path.isdir(validated_path):
            raise HTTPException(
                status_code=400, detail=f"Path is not a directory: {path}"
            )

        # Calculate size using utility function
        try:
            total_size = calculate_local_model_weight_size(validated_path, is_diffusion)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except NotADirectoryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid model_index.json: {str(e)}"
            )

        return {"size": total_size}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating model weight size for {path}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate size: {str(e)}"
        )


@router.post("/files/parse-gguf", response_model=GGUFParseResponse)
async def parse_gguf_file(request: GGUFParseRequest):
    """
    Parse a GGUF file using gguf-parser binary on the worker.

    Security:
    - Uses os.path.realpath to resolve symlinks and prevent directory traversal
    - Only allow parsing of existing files
    - 60 second timeout to prevent long-running processes
    """
    try:
        # 1. Deserialize Model object
        model = Model.model_validate(request.model_dict)

        # 2. Path validation - use validate_path_security for robust security
        validated_path = validate_path_security(model.local_path)

        # Check if file exists
        if not os.path.exists(validated_path):
            raise HTTPException(
                status_code=404, detail=f"File not found: {model.local_path}"
            )

        # Check if path is a file
        if not os.path.isfile(validated_path):
            raise HTTPException(
                status_code=400, detail=f"Path is not a file: {model.local_path}"
            )

        # Update model.local_path to use validated path
        model.local_path = validated_path

        # 3. Build offload enum
        offload_enum = GPUOffloadEnum(request.offload)

        # 4. Prepare kwargs (override parameters)
        kwargs = {}
        if request.tensor_split:
            kwargs["tensor_split"] = request.tensor_split
        if request.rpc:
            kwargs["rpc"] = request.rpc

        # Worker should use its own cache_dir from config, not from server.
        # The cache_dir is node-local and server's path may not exist on worker.
        worker_config = get_global_config()
        kwargs["cache_dir"] = worker_config.cache_dir

        # 5. Reuse _gguf_parser_command to build command
        command = await _gguf_parser_command(model, offload_enum, **kwargs)
        env = _gguf_parser_env(model)

        # 6. Execute command
        logger.debug(f"Executing gguf-parser command: {' '.join(map(str, command))}")

        # Use subprocess.run in a thread to avoid asyncio event loop issues
        # This is more reliable than asyncio.create_subprocess_exec in worker threads
        def run_command():
            """Run command synchronously in a thread."""
            try:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
                return result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return -1, b"", b"Parsing timed out after 60 seconds"

        # Run in thread pool to avoid blocking
        returncode, stdout, stderr = await asyncio.to_thread(run_command)
        logger.debug("Process completed, processing output")

        if returncode == -1:
            # Timeout
            logger.error(f"GGUF parsing timed out for {model.local_path}")
            return GGUFParseResponse(success=False, error=stderr.decode())

        if returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"GGUF parsing failed for {model.local_path}: {error_msg}")
            return GGUFParseResponse(success=False, error=error_msg)

        output_str = stdout.decode()
        logger.debug(f"GGUF parsing succeeded for {model.local_path}")
        return GGUFParseResponse(success=True, output=output_str)

    except HTTPException:
        raise
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Error parsing GGUF file: {e}\nTraceback:\n{error_detail}")
        return GGUFParseResponse(success=False, error=f"{type(e).__name__}: {str(e)}")
