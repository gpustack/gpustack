from typing import Optional, ClassVar, List

from pydantic import ConfigDict, BaseModel
from sqlalchemy import UniqueConstraint, Column, Integer, ForeignKey
from sqlmodel import SQLModel, Field

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    pydantic_column_type,
    ListParams,
    PublicFields,
    PaginatedList,
)


class GPUInstancePersistentVolumeNFS(BaseModel):
    """
    Represents the NFS configuration for a GPU instance persistent volume.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    server: str
    """
    The hostname or IP address of the NFS server.
    """

    share: str
    """
    The exported NFS share path on the server.
    """

    sub_directory: Optional[str] = None
    """
    Optional sub-directory within the NFS share to use for the persistent volume.
    If not specified, the root of the share will be used.
    """

    mount_permissions: Optional[str] = None
    """
    Optional mount permissions for the directory, such as "0777".
    If not specified, it defaults to "0", which means respecting the permissions set on the NFS server side.
    """

    mount_options: Optional[List[str]] = Field(
        default=[
            "hard",
            "vers=4",
            "rsize=1048576",
            "wsize=1048576",
            "noatime",
            "nodiratime",
        ]
    )
    """
    Optional list of mount options for the NFS volume.
    """


class GPUInstancePersistentVolumeS3(BaseModel):
    """
    Represents the S3 configuration for a GPU instance persistent volume, including connection details and mount options.

    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    endpoint: str
    """
    The S3-compatible storage endpoint URL.
    """

    region: Optional[str] = None
    """
    The region of the S3 storage, if applicable.
    """

    insecure: Optional[bool] = False
    """
    Whether to skip TLS verification when connecting to the S3 endpoint.
    """

    access_key: Optional[str] = None
    """
    The access key for authenticating with the S3 storage.

    Write-only input, it is required in create or update operations.
    """

    secret_key: Optional[str] = None
    """
    The secret key for authenticating with the S3 storage.

    Write-only input, it is required in create or update operations.
    """

    bucket: Optional[str] = None
    """
    The S3 bucket name to use for the persistent volume.
    If not specified, a default bucket named with the pattern "gpu-instance-pv-{id}" will be used.
    """

    mount_options: Optional[List[str]] = Field(
        default=[
            "--no-checksum",
            "--memory-limit=4000",
            "--max-flushers=32",
            "--max-parallel-parts=32",
            "--part-sizes=25",
            "--list-type=2",
            "--no-specials",
        ],
    )
    """
    Optional list of mount options for [GeeseFS](https://github.com/yandex-cloud/geesefs).

    Intensive writing for large files:
        disable CPU overhead, reduce freshening frequency, maximize parallelism,
        and reduce part sizes to improve writing performance for large files.
        ["--no-checksum","--memory-limit=4000","--max-flushers=32","--max-parallel-parts=32","--part-sizes=25"]

    Sequential reading for large files:
        increase read-ahead size and parallelism,
        and increase the memory cache limit to improve reading performance for large files.
        ["--read-ahead-large=200000"," --large-read-cutoff=10240","--read-ahead-parallel=40000","--memory-limit=8000"]

    Random reading for small files:
        decrease read-ahead size, extend metadata cache TTL,
        and increase the entry limit to improve reading performance for small files.
        ["--read-ahead-small=64","--small-read-cutoff=64","--read-ahead=1024","--stat-cache-ttl=300s","--entry-limit=200000"]

    High availability for writing:
        increase the number of retries and enable fsync on close to improve data durability for writing.
        ["--sdk-max-retries=10","--read-retry-attempts=5","--fsync-on-close","--cache=/mnt/disk-cache]

    For non-Yandex S3-compatible object storage service:
        ["--list-type=2","--no-specials"]
    """


class GPUInstancePersistentVolumeS3Public(GPUInstancePersistentVolumeS3):
    """Public-view S3 config.

    Identical to :class:`GPUInstancePersistentVolumeS3` except ``secret_key``
    is excluded from serialization, so the credential never leaves the
    server in API responses. Inputs (create/update) continue to bind
    against :class:`GPUInstancePersistentVolumeS3`.
    """

    secret_key: Optional[str] = Field(default=None, exclude=True)


class GPUInstancePersistentVolumeTypeSpec(BaseModel):
    """
    Represents the specification for creating or updating a GPU instance persistent volume type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    nfs: Optional[GPUInstancePersistentVolumeNFS] = None
    """
    Optional NFS configuration for the GPU instance persistent volume type.
    """

    s3: Optional[GPUInstancePersistentVolumeS3] = None
    """
    Optional S3 configuration for the GPU instance persistent volume type.
    """


class GPUInstancePersistentVolumeTypeSpecPublic(GPUInstancePersistentVolumeTypeSpec):
    """Public-view spec.

    Narrows ``s3`` to :class:`GPUInstancePersistentVolumeS3Public` so the
    ``secret_key`` is dropped from API responses.
    """

    s3: Optional[GPUInstancePersistentVolumeS3Public] = None


class GPUInstancePersistentVolumeTypeStatus(BaseModel):
    """
    Represents the status of a GPU instance persistent volume type.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    phase: Optional[str] = None
    """
    The current phase, e.g. "Deleting" during finalizer-driven soft delete;
    None means the type is active.
    """

    phase_message: Optional[str] = None
    """
    Optional message with detail about the current phase (e.g. why finalize is waiting).
    """

    finalizing: Optional[List[int]] = None
    """
    Cluster ids whose downstream object is not yet cleaned up; the row is
    hard-deleted once this is empty.
    """


class GPUInstancePersistentVolumeTypeBase(SQLModel):
    """
    Base model for GPU instance persistent volume types, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    # For tenant scope.
    # Every object belongs to one Org. The route layer fills this with
    # ctx.current_principal_id (or platform_principal_id() for admin).
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    display_name: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=63,
    )
    """
    Display name of the GPU instance persistent volume type, for easier identification by users.
    """

    description: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=1024,
    )
    """
    Description of the GPU instance persistent volume type.
    """


class GPUInstancePersistentVolumeType(
    GPUInstancePersistentVolumeTypeBase, BaseModelMixin, table=True
):
    """
    Represents a GPU Instance persistent volume type.
    """

    __tablename__ = 'gpu_instance_persistent_volume_types'
    __table_args__ = (
        # Enforce unique constraint on (owner_principal_id, name) to ensure
        # each principal can only have one key with a given name.
        # This allows different principals to have keys with the same name,
        # but prevents duplicates for the same principal.
        UniqueConstraint(
            'owner_principal_id',
            'name',
            name='uq_gpu_instance_persistent_volume_type_name_per_principal',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    # Record the creator of the GPU instance persistent volume type for
    # auditing and ownership purposes.
    creator_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    """
    Reference to the principal who created the GPU instance persistent volume type.
    """

    name: str = Field(
        max_length=63,
    )
    """
    Name of the GPU instance persistent volume type.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstancePersistentVolumeTypeSpec = Field(
        sa_type=pydantic_column_type(GPUInstancePersistentVolumeTypeSpec),
    )
    """
    Specification for the GPU instance persistent volume type, including NFS or S3 configuration.
    """

    status: Optional[GPUInstancePersistentVolumeTypeStatus] = Field(
        sa_type=pydantic_column_type(GPUInstancePersistentVolumeTypeStatus),
        default=None,
    )
    """
    Status of the GPU instance persistent volume type, including the soft-delete
    phase and the clusters still being finalized.
    """


class GPUInstancePersistentVolumeTypeUpdate(GPUInstancePersistentVolumeTypeBase):
    """
    Represents the fields that can be updated for a GPU instance persistent volume type.
    """

    pass


class GPUInstancePersistentVolumeTypeCreate(GPUInstancePersistentVolumeTypeUpdate):
    """
    Represents the fields required to create a new GPU instance persistent volume type.
    """

    name: str
    """
    Created name of the GPU instance persistent volume type.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstancePersistentVolumeTypeSpec
    """
    Specification for the GPU instance persistent volume type, including NFS or S3 configuration.
    """


class GPUInstancePersistentVolumeTypePublic(
    GPUInstancePersistentVolumeTypeCreate, PublicFields
):
    """
    Represents the public view of a GPU instance persistent volume type,
    containing only fields that are safe to expose to clients.
    """

    spec: GPUInstancePersistentVolumeTypeSpecPublic

    creator_id: Optional[int] = None
    """
    Reference to the principal who created the GPU instance persistent volume type.
    """

    status: Optional[GPUInstancePersistentVolumeTypeStatus] = None
    """
    Status of the GPU instance persistent volume type (soft-delete phase, finalizing clusters).
    """


class GPUInstancePersistentVolumeTypeListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "created_at",
        "updated_at",
    ]


GPUInstancePersistentVolumeTypesPublic = PaginatedList[
    GPUInstancePersistentVolumeTypePublic
]
