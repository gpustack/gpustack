from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict

from gpustack.schemas.model_sets import ModelSpec


class ResourceClaim(BaseModel):
    ram: int  # in bytes
    vram: int  # in bytes


class RoleResourceClaim(BaseModel):
    """One role's share of a group's footprint.

    ``ram`` / ``vram`` are the role's WHOLE demand -- every replica summed --
    because that is the number that decides whether the group fits, and a
    per-replica figure presented as the answer is what made a 4P4D deployment
    read like a single instance.

    ``per_replica`` is what one member costs, and it is ``None`` when the
    members disagree: a role spread over two accelerator types sizes
    differently on each, and there a single number would be a summary of
    nothing. The UI shows the split only when it exists.
    """

    role: str
    replicas: int
    ram: int  # in bytes, all replicas of this role
    vram: int  # in bytes, all replicas of this role
    per_replica: Optional[ResourceClaim] = None


class RoleResourceDemand(BaseModel):
    """One role's ask, when the group did not fit and there is nothing to claim.

    The refusal counterpart of ``RoleResourceClaim``: same shape, different
    provenance. A claim is read off a placement that was proved; a demand is
    what the role's own selector priced before anything was placed, which is
    the only honest figure available once the solve has failed.

    ``placeable`` is how many of this role's ``replicas`` the cluster was
    measured to hold **with nothing else of the group standing in**. Every role
    is measured the same way, so the counts are comparable with each other --
    and a breakdown where every role is individually satisfied is itself the
    answer: the members do not fit *together*, which is what the group-level
    message above says.
    """

    role: str
    replicas: int
    placeable: int
    ram: int  # in bytes, all replicas of this role
    vram: int  # in bytes, all replicas of this role
    # What one member costs. ``None`` means no selector ever priced this role
    # -- no worker was eligible for it, so none ever ran -- and the count above
    # is then the whole of what is known. Unlike ``RoleResourceClaim``, where
    # None means the members disagree: a demand is one price per role, so there
    # is nothing for them to disagree about.
    per_replica: Optional[ResourceClaim] = None


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

    # Only a role-bearing (PD) deployment fills these, and then
    # `resource_claim` above is the group's TOTAL rather than one instance's.
    # A role-less model leaves them None and its claim keeps meaning exactly
    # what it always did -- one replica -- so no existing reader changes.
    role_resource_claims: Optional[List[RoleResourceClaim]] = None
    role_resource_claims_by_cluster_id: Optional[Dict[int, List[RoleResourceClaim]]] = (
        None
    )

    # The mirror image of the two above, filled when a role-bearing deployment
    # does NOT fit. It breaks a refusal down role by role for the same reason
    # the claims above do: a bare count -- "the group needs 2 placements and
    # the cluster has room for 0" -- answers the same question in units that
    # cannot be compared with the success case, and never says what a member
    # costs.
    role_resource_demands_by_cluster_id: Optional[
        Dict[int, List[RoleResourceDemand]]
    ] = None

    error: Optional[bool] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class ModelEvaluationResponse(BaseModel):
    results: List[ModelEvaluationResult] = []

    model_config = ConfigDict(protected_namespaces=())
