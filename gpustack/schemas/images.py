from typing import List, Optional
from pydantic import BaseModel


class ImageData(BaseModel):
    b64_json: str
    finish_reason: Optional[str]
    index: int
    object: str
    progress: float


class ImageGenerationChunk(BaseModel):
    created: int
    model: Optional[str]
    data: List[ImageData]
    object: str
    usage: dict
