from pydantic import BaseModel, Field


class FileExistsResponse(BaseModel):
    """Response indicating whether a path exists."""

    exists: bool = Field(..., description="Whether the path exists")
    path: str = Field(..., description="The path that was checked")
    is_file: bool = Field(default=False, description="Whether the path is a file")
    is_dir: bool = Field(default=False, description="Whether the path is a directory")
