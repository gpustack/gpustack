from typing import List, Optional
from pydantic import BaseModel


class ImageData(BaseModel):
    index: int
    object: str
    progress: float
    b64_json: Optional[str] = None
    finish_reason: Optional[str] = None


class ImageGenerationChunk(BaseModel):
    created: int
    object: str
    model: Optional[str] = None
    data: List[ImageData] = []
    usage: Optional[dict] = None
