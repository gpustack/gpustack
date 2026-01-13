import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from gpustack.api.auth import worker_auth
from gpustack.schemas.filesystem import FileEntry, FileListResponse, FileExistsResponse


router = APIRouter(dependencies=[Depends(worker_auth)])

logger = logging.getLogger(__name__)


@router.get("/files/list", response_model=FileListResponse)
async def list_files(path: str = Query(..., description="Directory path to list")):
    """
    List files and directories in the specified path.
    """
    try:
        # Normalize the path
        normalized_path = os.path.normpath(path)

        # Check if path exists
        if not os.path.exists(normalized_path):
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")

        # Check if path is a directory
        if not os.path.isdir(normalized_path):
            raise HTTPException(
                status_code=400, detail=f"Path is not a directory: {path}"
            )

        # List files and directories
        files = []
        try:
            entries = os.listdir(normalized_path)
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

        for entry_name in entries:
            entry_path = os.path.join(normalized_path, entry_name)
            try:
                stat_info = os.stat(entry_path)
                is_file = os.path.isfile(entry_path)
                is_dir = os.path.isdir(entry_path)
                is_symlink = os.path.islink(entry_path)

                file_entry = FileEntry(
                    name=entry_name,
                    path=entry_path,
                    size=stat_info.st_size if is_file else None,
                    is_file=is_file,
                    is_dir=is_dir,
                    is_symlink=is_symlink,
                )
                files.append(file_entry)
            except OSError as e:
                logger.warning(f"Failed to stat {entry_path}: {e}")
                continue

        return FileListResponse(files=files, path=normalized_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files in {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/files/read")
async def read_file(
    path: str = Query(..., description="File path to read"),
    offset: Optional[int] = Query(0, description="Offset in bytes"),
    length: Optional[int] = Query(None, description="Length in bytes to read"),
):
    """
    Read content of a file.
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

        # Read file content
        try:
            with open(normalized_path, "rb") as f:
                if offset > 0:
                    f.seek(offset)

                if length is not None:
                    content = f.read(length)
                else:
                    content = f.read()
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to read file: {str(e)}"
            )

        return {"content": content, "path": normalized_path}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@router.get("/files/exists", response_model=FileExistsResponse)
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
