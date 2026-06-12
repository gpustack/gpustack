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


class GPUInstanceSSHPublicKeySpec(BaseModel):
    """
    Represents the specification for creating or updating a GPU instance SSH public key.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    data: str
    """
    The GPU instance SSH public key data,
    typically in OpenSSH format (e.g., "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC...").
    Multiple keys can be operated by newlines if needed.
    """


class GPUInstanceSSHPublicKeyBase(SQLModel):
    """
    Base model for GPU instance SSH public keys, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    # For tenant scope.
    # Every object belongs to one Org. The route layer fills this with
    # ctx.current_principal_id (or platform_principal_id for admin).
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
    Display name of the GPU instance SSH public key, for easier identification by users.
    """

    description: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=1024,
    )
    """
    Description of the GPU instance SSH public key.
    """

    spec: GPUInstanceSSHPublicKeySpec = Field(
        sa_type=pydantic_column_type(GPUInstanceSSHPublicKeySpec),
    )
    """
    Spec for the GPU instance SSH public key, containing the key data and related information.
    """


class GPUInstanceSSHPublicKey(GPUInstanceSSHPublicKeyBase, BaseModelMixin, table=True):
    """
    Represents a GPU Instance SSH public key.
    """

    __tablename__ = 'gpu_instance_ssh_public_keys'
    __table_args__ = (
        # Enforce unique constraint on (owner_principal_id, name) to ensure
        # each principal can only have one key with a given name.
        # This allows different principals to have keys with the same name,
        # but prevents duplicates for the same principal.
        UniqueConstraint(
            'owner_principal_id',
            'name',
            name='uq_gpu_instance_ssh_public_key_name_per_principal',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    # Record the creator of the GPU instance SSH public key for auditing
    # and ownership purposes.
    creator_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    """
    Reference to the principal who created the GPU instance SSH public key.
    """

    name: str = Field(
        max_length=63,
    )
    """
    Name of the GPU instance SSH public key.
    Must be unique in the scope of the owning principal.
    """


class GPUInstanceSSHPublicKeyUpdate(GPUInstanceSSHPublicKeyBase):
    """
    Represents the fields that can be updated for a GPU instance SSH public key.
    """

    pass


class GPUInstanceSSHPublicKeyCreate(GPUInstanceSSHPublicKeyBase):
    """
    Represents the fields required to create a new GPU instance SSH public key.
    """

    name: str
    """
    Created name of the GPU instance SSH public key.
    Must be unique in the scope of the owning principal.
    """


class GPUInstanceSSHPublicKeyPublic(GPUInstanceSSHPublicKeyCreate, PublicFields):
    """
    Represents the public view of a GPU instance SSH public key,
    containing only fields that are safe to expose to clients.
    """

    creator_id: Optional[int] = None
    """
    Reference to the principal who created the GPU instance SSH public key.
    """

    pass


class GPUInstanceSSHPublicKeyListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "created_at",
        "updated_at",
    ]


GPUInstanceSSHPublicKeysPublic = PaginatedList[GPUInstanceSSHPublicKeyPublic]
