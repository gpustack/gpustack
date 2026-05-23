from typing import Optional, List, ClassVar

from pydantic import ConfigDict
from sqlalchemy import UniqueConstraint, Column, Integer, ForeignKey
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    PublicFields,
    PaginatedList,
    ListParams,
    pydantic_column_type,
)
from gpustack.schemas.gpu_instances import GPUInstanceSpec


class GPUInstanceSpecTemplate(GPUInstanceSpec):
    """
    GPU instance spec template,
    containing fields that define the configuration of a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    type_: Optional[str] = Field(
        default=None,
        exclude=True,
    )
    """
    Hidden this field for template.
    """


class GPUInstanceTemplateBase(SQLModel):
    """
    Base model for GPU instance templates, containing common fields.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    # For tenant scope.
    # - NULL = global (admin-managed).
    # - Non-NULL = belongs to the principal (tenant) that owns it.
    #   The route layer fills this with ctx.current_principal_id.
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    display_name: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=63,
    )
    """
    Display name of the GPU instance template, for easier identification by users.
    """

    description: Optional[str] = Field(
        nullable=True,
        default=None,
        max_length=1024,
    )
    """
    Description of the GPU instance template.
    """

    manufacturer: str = Field(
        index=True,
        unique=False,
        default="cpu",
    )
    """
    Manufacturer of the GPU instance,
    e.g., "nvidia", "amd", "ascend".
    """

    spec: GPUInstanceSpecTemplate = Field(
        sa_type=pydantic_column_type(GPUInstanceSpecTemplate),
    )
    """
    Spec for the GPU instance template, containing details like container image, resources, etc.
    """


class GPUInstanceTemplate(GPUInstanceTemplateBase, BaseModelMixin, table=True):
    """
    Represents a template for creating GPU instances.

    GPU instance is reflected as a Kubernetes Pod in the cluster.
    """

    __tablename__ = 'gpu_instance_templates'
    __table_args__ = (
        # Enforce unique constraint on (owner_principal_id, name) to ensure
        # each principal can only have one key with a given name.
        # This allows different principals to have keys with the same name,
        # but prevents duplicates for the same principal.
        UniqueConstraint(
            'owner_principal_id',
            'name',
            name='uq_gpu_instance_template_name_per_principal',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    name: str = Field(
        max_length=63,
    )
    """
    Name of the GPU instance template.
    Must be unique in the scope of the owning principal.
    """


class GPUInstanceTemplateUpdate(GPUInstanceTemplateBase):
    """
    Represents the fields that can be updated for a GPU instance template.
    """

    pass


class GPUInstanceTemplateCreate(GPUInstanceTemplateUpdate):
    """
    Represents the fields required to create a new GPU instance template.
    """

    name: str
    """
    Created name of the GPU instance template.
    Must be unique in the scope of the owning principal.
    """


class GPUInstanceTemplatePublic(GPUInstanceTemplateCreate, PublicFields):
    """
    Represents the public view of a GPU instance template,
    containing only fields that are safe to expose to clients.
    """

    pass


class GPUInstanceTemplateListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "manufacturer",
        "created_at",
        "updated_at",
    ]


GPUInstanceTemplatesPublic = PaginatedList[GPUInstanceTemplatePublic]
