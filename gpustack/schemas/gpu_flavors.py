from typing import Dict, List
from pydantic import BaseModel


class GPUFlavorsResponse(BaseModel):
    type_map: Dict[str, int] = {}
    allow_gpu_count: List[int] = []
