from typing import List, Optional
from pydantic import BaseModel, Field


class FileEntry(BaseModel):
    """Represents a file or directory entry."""

    name: str = Field(..., description="Name of the file or directory")
    path: str = Field(..., description="Full path of the file or directory")
    size: Optional[int] = Field(None, description="Size of the file in bytes")
    is_file: bool = Field(..., description="Whether this is a file")
    is_dir: bool = Field(..., description="Whether this is a directory")
    is_symlink: bool = Field(
        default=False, description="Whether this is a symbolic link"
    )


class FileListResponse(BaseModel):
    """Response containing a list of files and directories."""

    files: List[FileEntry] = Field(..., description="List of files and directories")
    path: str = Field(..., description="The path that was listed")


class FileExistsResponse(BaseModel):
    """Response indicating whether a path exists."""

    exists: bool = Field(..., description="Whether the path exists")
    path: str = Field(..., description="The path that was checked")
    is_file: bool = Field(default=False, description="Whether the path is a file")
    is_dir: bool = Field(default=False, description="Whether the path is a directory")
