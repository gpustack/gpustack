from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.model_sets import ModelSpec


class ResourceClaim(BaseModel):
    ram: int  # in bytes
    vram: int  # in bytes


class ModelEvaluationRequest(BaseModel):
    cluster_id: Optional[int] = None
    model_specs: Optional[List[ModelSpec]] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelEvaluationResult(BaseModel):
    compatible: bool = True
    compatibility_messages: Optional[List[str]] = []
    scheduling_messages: Optional[List[str]] = []
    default_spec: Optional[ModelSpec] = None
    resource_claim: Optional[ResourceClaim] = None
    resource_claim_by_cluster_id: Optional[Dict[int, ResourceClaim]] = None

    error: Optional[bool] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelEvaluationResponse(BaseModel):
    results: List[ModelEvaluationResult] = []

    model_config = ConfigDict(protected_namespaces=())
