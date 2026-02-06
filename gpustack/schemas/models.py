from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel, Text

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.links import (
    ModelInstanceDraftModelFileLink,
    ModelInstanceModelFileLink,
)
from gpustack.utils.command import find_parameter, find_bool_parameter
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    AccessPolicyEnum,
)

if TYPE_CHECKING:
    from gpustack.schemas.model_files import ModelFile
    from gpustack.schemas.clusters import Cluster

# Models


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    MODEL_SCOPE = "model_scope"
    LOCAL_PATH = "local_path"


class CategoryEnum(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    UNKNOWN = "unknown"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class BackendEnum(str, Enum):
    VLLM = "vLLM"
    VOX_BOX = "VoxBox"
    ASCEND_MINDIE = "MindIE"
    SGLANG = "SGLang"
    CUSTOM = "Custom"


class BackendSourceEnum(str, Enum):
    CUSTOM = "custom"
    BUILT_IN = "built_in"
    COMMUNITY = "community"


class SpeculativeAlgorithmEnum(str, Enum):
    EAGLE3 = "eagle3"
    MTP = "mtp"
    NGRAM = "ngram"


class GPUSelector(BaseModel):
    # format of each element: "worker_name:device:gpu_index", example: "worker1:cuda:0"
    gpu_ids: Optional[List[str]] = None
    gpus_per_replica: Optional[int] = None


class ExtendedKVCacheConfig(BaseModel):
    enabled: bool = False
    """ Enable extended KV cache for the model."""

    ram_ratio: Optional[float] = 1.2
    """ RAM-to-VRAM ratio for KV cache. For example, 2.0 means the RAM is twice the size of the VRAM. """

    ram_size: Optional[int] = None
    """ Maximum size of the KV cache to be stored in local CPU memory (unit: GiB). Overrides ram_ratio if both are set. """

    chunk_size: Optional[int] = None
    """ Size for each KV cache chunk (unit: number of tokens). """


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def model_source_key(self) -> str:
        """Returns a unique identifier for the model, independent of quantization."""
        if self.source == SourceEnum.HUGGING_FACE:
            return self.huggingface_repo_id or ""
        elif self.source == SourceEnum.MODEL_SCOPE:
            return self.model_scope_model_id or ""
        elif self.source == SourceEnum.LOCAL_PATH:
            return self.local_path or ""
        return ""

    @property
    def readable_source(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend([self.model_scope_model_id, self.model_scope_file_path])
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        return "/".join([value for value in values if value is not None])

    @property
    def model_source_index(self) -> str:
        # Include source type to differentiate between different sources
        values = [str(self.source)]

        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend([self.model_scope_model_id, self.model_scope_file_path])
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        # Filter out None values and join
        filtered_values = [v for v in values if v is not None]
        source_string = "/".join(filtered_values)
        return hashlib.sha256(source_string.encode()).hexdigest()

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided "
                    "when source is 'huggingface'"
                )

        if self.source == SourceEnum.MODEL_SCOPE:
            if not self.model_scope_model_id:
                raise ValueError(
                    "model_scope_model_id must be provided when source is 'model_scope'"
                )

        if self.source == SourceEnum.LOCAL_PATH:
            if not self.local_path:
                raise ValueError(
                    "local_path must be provided when source is 'local_path'"
                )
        return self

    model_config = ConfigDict(protected_namespaces=())


class SpeculativeConfig(BaseModel):
    """Configuration for speculative decoding."""

    enabled: bool = False
    """Whether speculative decoding is enabled."""
    algorithm: Optional[SpeculativeAlgorithmEnum] = None
    """The algorithm to use for speculative decoding."""
    draft_model: Optional[str] = None
    """The draft model to use for speculative decoding.

    It can be a draft model name from the model catalog, a local path or a model ID from the main model source."""
    num_draft_tokens: Optional[int] = None
    """The number of draft tokens."""
    # For ngram only
    ngram_min_match_length: Optional[int] = None
    """Minimum length of the n-gram to match."""
    ngram_max_match_length: Optional[int] = None
    """Maximum length of the n-gram to match."""


class ModelSpecBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    meta: Optional[Dict[str, Any]] = Field(sa_type=JSON, default={})

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    categories: List[str] = Field(sa_type=JSON, default=[])
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: Optional[bool] = None
    distributed_inference_across_workers: Optional[bool] = None
    worker_selector: Optional[Dict[str, str]] = Field(sa_type=JSON, default={})
    gpu_selector: Optional[GPUSelector] = Field(
        sa_type=pydantic_column_type(GPUSelector), default=None
    )

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)

    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)
    restart_on_error: Optional[bool] = True
    distributable: Optional[bool] = False

    # Extended KV Cache configuration. Currently maps to LMCache config in vLLM and SGLang.
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # Enable generic proxy for model, the control of generic proxy
    # is migrated to ModelAccess. Keeping this field for backward compatibility
    generic_proxy: Optional[bool] = Field(default=False)

    @model_validator(mode="after")
    def set_defaults(self):
        backend = get_backend(self)
        if self.distributed_inference_across_workers is None:
            self.distributed_inference_across_workers = (
                True
                if backend
                in [BackendEnum.VLLM, BackendEnum.ASCEND_MINDIE, BackendEnum.SGLANG]
                else False
            )
        return self


class ModelBase(ModelSpecBase):
    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")
    # Deprecated field, kept for backward compatibility
    access_policy: AccessPolicyEnum = Field(default=AccessPolicyEnum.AUTHED)


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    id: Optional[int] = Field(default=None, primary_key=True)

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="model",
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_models",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_route_targets: List["ModelRouteTarget"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "models",
            "cascade": "delete",
        },
    )

    model_routes: List["ModelRoute"] = Relationship(
        back_populates="models",
        link_model=ModelRouteTarget,
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "model,model_route_targets,route_targets,model_route",
        },
    )


class ModelListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "source",
        "cluster_id",
        "replicas",
        "ready_replicas",
        "created_at",
        "updated_at",
    ]


class ModelCreate(ModelBase):
    enable_model_route: Optional[bool] = Field(default=None)


class ModelUpdate(ModelBase):
    pass


class ModelPublic(
    ModelBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelsPublic = PaginatedList[ModelPublic]


# Model Instances


class ModelInstanceStateEnum(str, Enum):
    r"""
    Enum for Model Instance State

    Transitions:

       |- - - - - Scheduler - - - - |- - ServeManager - -|- - - - Controller - - - -|- ServeManager -|
       |                            |                    |                          |                |
    PENDING ---> ANALYZING ---> SCHEDULED ---> INITIALIZING ---> DOWNLOADING ---> STARTING ---> RUNNING
                     |            ^  |               |                |               |          ^
                     |            |  |               |                |               |          |(Worker ready)
                     |------------|--|---------------|----------------|---------------|----------|
                     \____________|_____________________________________________________________/|
                                  |                  ERROR                                       |(Worker unreachable)
                                  └--------------------┘                                         v
                                    (Restart on Error)                                       UNREACHABLE
    """

    INITIALIZING = "initializing"
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    total_layers: Optional[int] = None
    ram: Optional[int] = Field(default=None)  # in bytes
    vram: Optional[Dict[int, int]] = Field(default=None)  # in bytes
    tensor_split: Optional[List[int]] = Field(default=None)
    vram_utilization: Optional[float] = Field(default=None)


class ModelInstanceSubordinateWorker(BaseModel):
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    total_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    # - For model file preparation
    download_progress: Optional[float] = None
    # - For model instance serving preparation
    pid: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    arguments: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class DistributedServerCoordinateModeEnum(Enum):
    # DELEGATED means that the subordinate workers' coordinate is by-pass to other framework.
    DELEGATED = "delegated"
    # INITIALIZE_LATER means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance SHOULD start after the main worker initializes.
    # For example, Ascend MindIE/vLLM/SGLang instances need to start their subordinate workers after the main worker initializes.
    INITIALIZE_LATER = "initialize_later"
    # RUN_FIRST means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance MUST get ready before the main worker starts.
    RUN_FIRST = "run_first"


class DistributedServers(BaseModel):
    # Indicates how the distributed servers coordinate with the main worker.
    mode: DistributedServerCoordinateModeEnum = (
        DistributedServerCoordinateModeEnum.DELEGATED
    )
    # Indicates if subordinate workers should download model files.
    download_model_files: Optional[bool] = True
    subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = Field(
        sa_column=Column(JSON), default=[]
    )
    model_config = ConfigDict(from_attributes=True)


@dataclass
class ModelInstanceDeploymentMetadata:
    """
    Metadata for model instance deployment.
    """

    name: str
    """
    Name for model instance deployment.
    """
    distributed: bool = False
    """
    Whether the model instance is deployed in distributed mode.
    """
    distributed_leader: bool = False
    """
    Whether the model instance is the leader in distributed mode.
    """
    distributed_follower: bool = False
    """
    Whether the model instance is a follower in distributed mode.
    """
    distributed_follower_index: Optional[int] = None
    """
    Index of the follower in distributed mode.
    It is None for leader or non-distributed mode.
    """


class ModelInstanceBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_advertise_address: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    pid: Optional[int] = None
    # FIXME: Migrate to ports.
    port: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    download_progress: Optional[float] = None
    resolved_path: Optional[str] = None
    draft_model_source: Optional[ModelSource] = Field(
        sa_column=Column(pydantic_column_type(ModelSource)), default=None
    )
    draft_model_download_progress: Optional[float] = None
    draft_model_resolved_path: Optional[str] = None
    restart_count: Optional[int] = 0
    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])

    model_id: int = Field(default=None, foreign_key="models.id")
    model_name: str

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None

    distributed_servers: Optional[DistributedServers] = Field(
        sa_column=Column(pydantic_column_type(DistributedServers)), default=None
    )
    # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
    # Disable it given that it's not a real issue for this particular field.
    model_config = ConfigDict(protected_namespaces=())

    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")

    def get_deployment_metadata(
        self,
        worker_id: int,
    ) -> Optional[ModelInstanceDeploymentMetadata]:
        """
        Get the deployment metadata for the model instance.

        Args:
            worker_id:
                The ID of the worker to get the deployment metadata for.

        Returns:
            The deployment metadata,
            or None if the model instance is not handling by the given `worker_id` worker.
        """

        dservers = self.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )

        name = self.name
        distributed = bool(subworkers)
        distributed_leader = distributed and self.worker_id == worker_id
        distributed_follower = distributed and not distributed_leader
        distributed_follower_index = None
        if distributed_follower:
            for idx, subworker in enumerate(subworkers):
                if subworker.worker_id == worker_id:
                    distributed_follower_index = idx
                    break
            if distributed_follower_index is not None:
                # Mutate the name to include the follower index,
                # so that each follower has a unique name.
                name += f"-f{distributed_follower_index}"

        if self.worker_id != worker_id and distributed_follower_index is None:
            # This model instance is not handling by the given worker.
            return None

        return ModelInstanceDeploymentMetadata(
            name=name,
            distributed=distributed,
            distributed_leader=distributed_leader,
            distributed_follower=distributed_follower,
            distributed_follower_index=distributed_follower_index,
        )


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    __tablename__ = 'model_instances'
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_files: List["ModelFile"] = Relationship(
        back_populates="instances",
        link_model=ModelInstanceModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    draft_model_files: List["ModelFile"] = Relationship(
        back_populates="draft_instances",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_model_instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        return self.id


class ModelInstanceCreate(ModelInstanceBase):
    pass


class ModelInstanceUpdate(ModelInstanceBase):
    pass


class ModelInstancePublic(
    ModelInstanceBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelInstancesPublic = PaginatedList[ModelInstancePublic]


def is_gguf_model(model: Union[Model, ModelSource]):
    """
    Check if the model is a GGUF model.
    Args:
        model: Model to check.
    """
    return (
        (
            model.source == SourceEnum.HUGGING_FACE
            and model.huggingface_filename
            and model.huggingface_filename.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.MODEL_SCOPE
            and model.model_scope_file_path
            and model.model_scope_file_path.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.LOCAL_PATH
            and model.local_path
            and model.local_path.endswith(".gguf")
        )
    )


def is_audio_model(model: Model):
    """
    Check if the model is a STT or TTS model.
    Args:
        model: Model to check.
    """
    if model.backend == BackendEnum.VOX_BOX:
        return True

    if model.categories:
        return (
            'speech_to_text' in model.categories or 'text_to_speech' in model.categories
        )

    return False


def is_llm_model(model: Model):
    """
    Check if the model is an LLM model.
    Args:
        model: Model to check.
    """
    return not model.categories or CategoryEnum.LLM in model.categories


def is_omni_model(model: Model) -> bool:
    """
    Check if the model is an omni model (Image or Audio category).
    Args:
        model: Model to check.
    """

    if model.backend == BackendEnum.VLLM and find_bool_parameter(
        model.backend_parameters, ["omni"]
    ):
        return True

    OMNI_CATEGORIES = (
        CategoryEnum.IMAGE,
        CategoryEnum.TEXT_TO_SPEECH,
    )
    return any(cat in model.categories for cat in OMNI_CATEGORIES)


def is_image_model(model: Model):
    """
    Check if the model is an image model.
    Args:
        model: Model to check.
    """
    return "image" in model.categories


def is_embedding_model(model: Model):
    """
    Check if the model is an embedding model.
    Args:
        model: Model to check.
    """
    return "embedding" in model.categories


def is_renaker_model(model: Model):
    """
    Check if the model is a reranker model.
    Args:
        model: Model to check.
    """
    return "reranker" in model.categories


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    if is_gguf_model(model):
        return BackendEnum.CUSTOM

    if is_audio_model(model):
        return BackendEnum.VOX_BOX

    return BackendEnum.VLLM


def get_mmproj_filename(model: Union[Model, ModelSource]) -> Optional[str]:
    """
    Get the mmproj filename for the model. If the mmproj is not provided in the model's
    backend parameters, it will try to find the default mmproj file.
    """
    if not is_gguf_model(model):
        return None

    if hasattr(model, "backend_parameters"):
        mmproj = find_parameter(model.backend_parameters, ["mmproj"])
        if mmproj and Path(mmproj).name == mmproj:
            return mmproj

    return "*mmproj*.gguf"
