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


class GPUInstancePersistentVolumeSpec(BaseModel):
    """
    Represents the specification for creating or updating a GPU instance persistent volume.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    type_: str
    """
    Specify the name of a GPU instance persistent volume type.
    """

    capacity: str = "20Gi"
    """
    Capacity of the GPU instance persistent volume,
    such as "20Gi".
    """


class GPUInstancePersistentVolumeStatus(BaseModel):
    """
    Represents the status of a GPU instance persistent volume.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    phase: Optional[str] = None
    """
    The current phase, e.g. "Deleting" during finalizer-driven soft delete;
    None means the volume is active.
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


class GPUInstancePersistentVolumeBase(SQLModel):
    """
    Base model for GPU instance persistent volumes, containing common fields.
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
    Display name of the GPU instance persistent volume, for easier identification by users.
    """

    description: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=1024,
    )
    """
    Description of the GPU instance persistent volume.
    """


class GPUInstancePersistentVolume(
    GPUInstancePersistentVolumeBase, BaseModelMixin, table=True
):
    """
    Represents a GPU instance persistent volume.
    """

    __tablename__ = 'gpu_instance_persistent_volumes'
    __table_args__ = (
        # Enforce unique constraint on (owner_principal_id, name) to ensure
        # each principal can only have one key with a given name.
        # This allows different principals to have keys with the same name,
        # but prevents duplicates for the same principal.
        UniqueConstraint(
            'owner_principal_id',
            'name',
            name='uq_gpu_instance_persistent_volume_name_per_principal',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    # Record the creator of the GPU instance persistent volume for auditing and ownership purposes.
    creator_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    """
    Reference to the principal who created the GPU instance persistent volume.
    """

    # Mirror of ``spec.type_`` as a real FK column with ``ON DELETE
    # RESTRICT`` so the DB blocks deleting a GPUInstancePersistentVolumeType
    # while any persistent volume still references it. The route layer
    # resolves the user-facing ``spec.type_`` name to this id at create
    # time; the JSON field stays the source of truth for the on-wire
    # shape and is what the worker-cluster CRD consumes.
    persistent_volume_type_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey(
                "gpu_instance_persistent_volume_types.id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
    )

    name: str = Field(
        max_length=63,
    )
    """
    Name of the GPU instance persistent volume.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstancePersistentVolumeSpec = Field(
        sa_type=pydantic_column_type(GPUInstancePersistentVolumeSpec),
    )
    """
    Specification for the GPU instance persistent volume, including type and capacity.
    """

    status: Optional[GPUInstancePersistentVolumeStatus] = Field(
        sa_type=pydantic_column_type(GPUInstancePersistentVolumeStatus),
        default=None,
    )
    """
    Status of the GPU instance persistent volume, including the soft-delete
    phase and the clusters still being finalized.
    """


class GPUInstancePersistentVolumeUpdate(GPUInstancePersistentVolumeBase):
    """
    Represents the fields that can be updated for a GPU instance persistent volume.
    """

    pass


class GPUInstancePersistentVolumeCreate(GPUInstancePersistentVolumeUpdate):
    """
    Represents the fields required to create a new GPU instance persistent volume.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Created name of the GPU instance persistent volume.
    Must be unique in the scope of the owning principal.
    """

    spec: GPUInstancePersistentVolumeSpec
    """
    Specification for the GPU instance persistent volume, including type and capacity.
    """


class GPUInstancePersistentVolumePublic(
    GPUInstancePersistentVolumeCreate, PublicFields
):
    """
    Represents the public view of a GPU instance persistent volume,
    containing only fields that are safe to expose to clients.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    creator_id: Optional[int] = None
    """
    Reference to the principal who created the GPU instance persistent volume.
    """

    status: Optional[GPUInstancePersistentVolumeStatus] = None
    """
    Status of the GPU instance persistent volume (soft-delete phase, finalizing clusters).
    """


class GPUInstancePersistentVolumeListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "created_at",
        "updated_at",
    ]


GPUInstancePersistentVolumesPublic = PaginatedList[GPUInstancePersistentVolumePublic]
