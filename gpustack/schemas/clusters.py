import secrets
from urllib.parse import urlparse
from enum import Enum
from typing import ClassVar, Optional, Dict, Any, List
from pydantic import BaseModel, computed_field, field_validator, ConfigDict
from sqlmodel import (
    Field,
    Relationship,
    Column,
    SQLModel,
    Text,
    Integer,
    ForeignKey,
    JSON,
    String,
)
import sqlalchemy as sa
from typing import TYPE_CHECKING

from gpustack.schemas.config import (
    SensitivePredefinedConfig,
    PredefinedConfigNoDefaults,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    PublicFields,
    ListParams,
    PaginatedList,
    pydantic_column_type,
)

if TYPE_CHECKING:
    from gpustack.schemas.models import Model, ModelInstance
    from gpustack.schemas.workers import Worker
    from gpustack.schemas.users import User


class WorkerPoolUpdate(SQLModel):
    name: str
    batch_size: Optional[int] = Field(default=None, ge=1)
    replicas: int = Field(default=1, ge=0)
    labels: Optional[Dict[str, str]] = Field(sa_column=Column(JSON), default={})


class Volume(BaseModel):
    format: Optional[str] = None
    size_gb: Optional[int] = None
    name: Optional[str] = None

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            return v
        # the worker id will be appended to the name to ensure uniqueness
        # so the max length is 60 characters to leave room for the worker id
        if len(v) > 60:
            raise ValueError("Volume name too long, max 60 characters")
        # allow alphanumeric characters, dashes, and periods
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
        )
        if not all(c in allowed_chars for c in v):
            raise ValueError("Volume name contains invalid characters")
        return v


class CloudOptions(BaseModel):
    volumes: Optional[List[Volume]] = None


class WorkerPoolCreate(WorkerPoolUpdate):
    instance_type: str
    os_image: str
    image_name: str
    cloud_options: Optional[CloudOptions] = Field(
        default={}, sa_column=Column(pydantic_column_type(CloudOptions))
    )
    zone: Optional[str] = None
    # instance_spec is for UI to store the instance_type's extended specifications for display.
    instance_spec: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )


class WorkerPoolBase(WorkerPoolCreate):
    cluster_id: int = Field(
        sa_column=Column(Integer, ForeignKey("clusters.id", ondelete="CASCADE"))
    )


class WorkerPool(WorkerPoolBase, BaseModelMixin, table=True):
    __tablename__ = "worker_pools"
    __table_args__ = (
        sa.Index("idx_worker_pools_deleted_at_created_at", "deleted_at", "created_at"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    cluster: Optional["Cluster"] = Relationship(
        back_populates="cluster_worker_pools",
        sa_relationship_kwargs={"lazy": "noload"},
    )
    pool_workers: list["Worker"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
        back_populates="worker_pool",
    )
    _workers: int = 0
    _ready_workers: int = 0

    @computed_field()
    @property
    def workers(self) -> int:
        if self.pool_workers is not None:
            return len(self.pool_workers)

        return self._workers

    @computed_field()
    @property
    def ready_workers(self) -> int:
        if self.pool_workers is not None:
            return len([w for w in self.pool_workers if w.state.value == 'ready'])

        return self._ready_workers

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, WorkerPool):
            return self.id == other.id
        return False

    def __init__(
        self,
        workers: int = 0,
        ready_workers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._workers = workers
        self._ready_workers = ready_workers


class WorkerPoolPublic(WorkerPoolBase, PublicFields):
    workers: int = Field(default=0)
    ready_workers: int = Field(default=0)


WorkerPoolsPublic = PaginatedList[WorkerPoolPublic]


class ClusterProvider(Enum):
    Docker = "Docker"
    Kubernetes = "Kubernetes"
    DigitalOcean = "DigitalOcean"


class CloudCredentialBase(SQLModel):
    """
    Supports providers other than Kubernetes and Docker.
    """

    name: str
    description: Optional[str] = None
    provider: ClusterProvider = Field(default=ClusterProvider.DigitalOcean)
    key: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class CloudCredentialUpdate(CloudCredentialBase):
    secret: Optional[str] = None


class CloudCredentialCreate(CloudCredentialUpdate):
    pass


class CloudCredential(CloudCredentialCreate, BaseModelMixin, table=True):
    __tablename__ = "cloud_credentials"
    __table_args__ = (
        sa.Index(
            "idx_cloud_credentials_deleted_at_created_at", "deleted_at", "created_at"
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, CloudCredential):
            return self.id == other.id
        return False


class CloudCredentialListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "provider",
        "created_at",
        "updated_at",
    ]


class CloudCredentialPublic(CloudCredentialBase, PublicFields):
    pass


CloudCredentialsPublic = PaginatedList[CloudCredentialPublic]


class ClusterStateEnum(str, Enum):
    PENDING = 'pending'
    PROVISIONING = 'provisioning'
    PROVISIONED = 'provisioned'
    READY = 'ready'


class ClusterUpdate(SQLModel):
    name: str
    description: Optional[str] = None
    gateway_endpoint: Optional[str] = None
    server_url: Optional[str] = None
    worker_config: Optional[PredefinedConfigNoDefaults] = Field(
        default=None,
        sa_column=Column(
            pydantic_column_type(
                PredefinedConfigNoDefaults,
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
            )
        ),
    )

    @field_validator("server_url")
    def validate_server_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) == 0:
            return None
        if v is not None:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid server_url format")
        return v


class ClusterCreateBase(ClusterUpdate):
    provider: ClusterProvider = Field(default=ClusterProvider.Docker)
    credential_id: Optional[int] = Field(
        default=None, foreign_key="cloud_credentials.id"
    )
    region: Optional[str] = None


class ClusterCreate(ClusterCreateBase):
    worker_pools: Optional[List[WorkerPoolCreate]] = Field(default=None)


class ClusterBase(ClusterCreateBase):
    state: ClusterStateEnum = ClusterStateEnum.PROVISIONING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    reported_gateway_endpoint: Optional[str] = None
    is_default: bool = Field(default=False)


class Cluster(ClusterBase, BaseModelMixin, table=True):
    __tablename__ = "clusters"
    __table_args__ = (
        sa.Index("idx_clusters_deleted_at_created_at", "deleted_at", "created_at"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_suffix: str = Field(nullable=False, default=secrets.token_hex(6))
    registration_token: str = Field(nullable=False, default=secrets.token_hex(16))
    cluster_worker_pools: List[WorkerPool] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="cluster",
    )
    cluster_models: List["Model"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"}, back_populates="cluster"
    )
    cluster_model_instances: List["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"}, back_populates="cluster"
    )
    cluster_users: list["User"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="cluster",
    )
    cluster_workers: List["Worker"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="cluster",
    )
    _models: int = 0
    _workers: int = 0
    _ready_workers: int = 0
    _gpus: int = 0

    @computed_field()
    @property
    def workers(self) -> int:
        if self.cluster_workers is not None:
            return len(self.cluster_workers)

        return self._workers

    @computed_field()
    @property
    def ready_workers(self) -> int:
        if self.cluster_workers is not None:
            return len([w for w in self.cluster_workers if w.state.value == 'ready'])

        return self._ready_workers

    @computed_field(alias="gpus")
    @property
    def gpus(self) -> int:
        if self.cluster_workers is not None:
            count = 0
            for worker in self.cluster_workers:
                if worker.status is None or worker.status.gpu_devices is None:
                    continue
                count += len(worker.status.gpu_devices)
            return count

        return self._gpus

    @computed_field(alias="models")
    @property
    def models(self) -> int:
        if self.cluster_models is not None:
            return len(self.cluster_models)

        return self._models

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Cluster):
            return self.id == other.id
        return False

    def __init__(
        self,
        workers: int = 0,
        ready_workers: int = 0,
        gpus: int = 0,
        models: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._workers = workers
        self._ready_workers = ready_workers
        self._gpus = gpus
        self._models = models


class ClusterListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "provider",
        "state",
        "workers",
        "ready_workers",
        "gpus",
        "models",
        "created_at",
        "updated_at",
    ]


class ClusterPublic(ClusterBase, PublicFields):
    workers: int = Field(default=0)
    ready_workers: int = Field(default=0)
    gpus: int = Field(default=0)
    models: int = Field(default=0)
    worker_config: Optional[PredefinedConfigNoDefaults] = Field(default=None)


ClustersPublic = PaginatedList[ClusterPublic]


class SensitiveRegistrationConfig(SensitivePredefinedConfig):
    model_config = ConfigDict(extra="ignore")
    token: str


class ClusterRegistrationTokenPublic(BaseModel):
    """
    The arguments of docker run command to register a worker.
    The env attribute is basically a dict of environment variables parsed from SensitiveRegistrationConfig.
    """

    token: str
    server_url: str
    image: str
    env: Dict[str, str]
    args: List[str]


class CredentialType(str, Enum):
    SSH = "ssh"
    CA = "ca"
    X509 = "x509"


class SSHKeyOptions(BaseModel):
    algorithm: str = Field(default="RSA")
    length: int = Field(default=2048)


class CredentialBase(SQLModel):
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    credential_type: CredentialType = Field(default=CredentialType.SSH)
    # pem format public key
    public_key: str = Field(sa_column=Column(Text, nullable=False))
    # base64 encoded private key
    encoded_private_key: str = Field(default="", sa_column=Column(Text, nullable=False))
    # e.g. RSA, ED25519
    ssh_key_options: Optional[SSHKeyOptions] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(SSHKeyOptions), nullable=True),
    )


class Credential(CredentialBase, BaseModelMixin, table=True):
    __tablename__ = "credentials"
    __table_args__ = (sa.Index("idx_credentials_external_id", "external_id"),)
    id: Optional[int] = Field(default=None, primary_key=True)
