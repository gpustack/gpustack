import secrets
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, computed_field
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

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, pydantic_column_type

if TYPE_CHECKING:
    from gpustack.schemas.models import Model, ModelInstance
    from gpustack.schemas.workers import Worker
    from gpustack.schemas.users import User


def is_in_session(obj: SQLModel) -> bool:
    insp = sa.inspect(obj)
    return insp.session is not None


class PublicFields:
    id: int
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class WorkerPoolUpdate(SQLModel):
    name: str
    batch_size: Optional[int] = Field(default=None, ge=1)
    replicas: int = Field(default=1, ge=0)


class Volume(BaseModel):
    format: Optional[str] = None
    size_gb: Optional[int] = None
    name: Optional[str] = None


class CloudOptions(BaseModel):
    volumes: Optional[List[Volume]] = None


class WorkerPoolCreate(WorkerPoolUpdate):
    instance_type: str
    os_image: str
    image_name: str
    labels: Optional[Dict[str, str]] = Field(sa_column=Column(JSON), default={})
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
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    pool_workers: list["Worker"] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"},
        back_populates="worker_pool",
    )
    _workers: Optional[int] = None
    _ready_workers: Optional[int] = None

    @computed_field()
    @property
    def workers(self) -> int:
        if not is_in_session(self):
            return 0 if not self._workers else self._workers
        return len(self.pool_workers) if self.pool_workers else 0

    @computed_field()
    @property
    def ready_workers(self) -> int:
        if not is_in_session(self):
            return 0 if not self._workers else self._workers
        if self.pool_workers is None or len(self.pool_workers) == 0:
            return 0
        return len([w for w in self.pool_workers if w.state.value == 'ready'])

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, WorkerPool):
            return self.id == other.id
        return False

    def __init__(
        self,
        workers: Optional[int] = None,
        ready_workers: Optional[int] = None,
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


class CloudCredentialPublic(CloudCredentialBase, PublicFields):
    pass


CloudCredentialsPublic = PaginatedList[CloudCredentialPublic]


class ClusterStateEnum(str, Enum):
    PROVISIONING = 'provisioning'
    PROVISIONED = 'provisioned'
    READY = 'ready'


class ClusterUpdate(SQLModel):
    name: str
    description: Optional[str] = None


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


class Cluster(ClusterBase, BaseModelMixin, table=True):
    __tablename__ = "clusters"
    __table_args__ = (
        sa.Index("idx_clusters_deleted_at_created_at", "deleted_at", "created_at"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_suffix: str = Field(nullable=False, default=secrets.token_hex(6))
    registration_token: str = Field(nullable=False, default=secrets.token_hex(16))
    cluster_worker_pools: List[WorkerPool] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "selectin"},
        back_populates="cluster",
    )
    cluster_models: List["Model"] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"}, back_populates="cluster"
    )
    cluster_model_instances: List["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"}, back_populates="cluster"
    )
    cluster_users: list["User"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "selectin"},
        back_populates="cluster",
    )
    cluster_workers: List["Worker"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "selectin"},
        back_populates="cluster",
    )
    _models: Optional[int] = None
    _workers: Optional[int] = None
    _ready_workers: Optional[int] = None
    _gpus: Optional[int] = None

    @computed_field()
    @property
    def workers(self) -> int:
        if not is_in_session(self):
            return 0 if not self._workers else self._workers
        return len(self.cluster_workers) if self.cluster_workers else 0

    @computed_field()
    @property
    def ready_workers(self) -> int:
        if not is_in_session(self):
            return 0 if not self._ready_workers else self._ready_workers
        if self.cluster_workers is None or len(self.cluster_workers) == 0:
            return 0
        return len([w for w in self.cluster_workers if w.state.value == 'ready'])

    @computed_field(alias="gpus")
    @property
    def gpus(self) -> int:
        if not is_in_session(self):
            return 0 if not self._gpus else self._gpus
        if self.workers == 0:
            return 0
        count = 0
        for worker in self.cluster_workers:
            if worker.status is None or worker.status.gpu_devices is None:
                continue
            count += len(worker.status.gpu_devices)
        return count

    @computed_field(alias="models")
    @property
    def models(self) -> int:
        if not is_in_session(self):
            return 0 if not self._models else self._models
        return len(self.cluster_models) if self.cluster_models else 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if super().__eq__(other) and isinstance(other, Cluster):
            return self.id == other.id
        return False

    def __init__(
        self,
        workers: Optional[int] = None,
        ready_workers: Optional[int] = None,
        gpus: Optional[int] = None,
        models: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._workers = workers
        self._ready_workers = ready_workers
        self._gpus = gpus
        self._models = models


class ClusterPublic(ClusterBase, PublicFields):
    workers: int = Field(default=0)
    ready_workers: int = Field(default=0)
    gpus: int = Field(default=0)
    models: int = Field(default=0)


ClustersPublic = PaginatedList[ClusterPublic]


class ClusterRegistrationTokenPublic(BaseModel):
    token: str
    server_url: str
    image: str


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
