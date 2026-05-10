from typing import Optional, List, Literal, ClassVar

from pydantic import BaseModel, ConfigDict
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    pydantic_camel_case_generator,
    PublicFields,
    PaginatedList,
    ListParams,
    pydantic_column_type,
)


class GPUInstancePort(BaseModel):
    """
    Represents a port mapping for GPU instances.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    port: int
    """
    The port number inside the container to expose.
    """

    protocol: Literal["TCP", "UDP", "SCTP"] = "TCP"
    """
    The protocol for the port.
    Defaults to "TCP".
    """


class GPUInstanceEnvVar(BaseModel):
    """
    Represents an environment variable for GPU instances.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the environment variable.
    """

    value: str
    """
    Value of the environment variable.
    """


class GPUInstanceResources(BaseModel):
    """
    Represents the resource requirements for a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    cpu: str = "1"
    """
    CPU resource request/limit for the GPU instance,
    e.g., "1" for 1 CPU.
    """

    ram: str = "2Gi"
    """
    RAM resource request/limit for the GPU instance,
    e.g., "2Gi" for 2 gigabyte of memory.
    """

    local_storage: str = "15Gi"
    """
    Local storage resource request/limit for the GPU instance,
    e.g., "15Gi" for 15 gigabytes of local storage.
    """

    accelerator: Optional[str] = None
    """
    Accelerator resource request/limit for the GPU instance,
    e.g., "1" for 1 GPU.
    """


class GPUInstanceImagePullSecretReference(BaseModel):
    """
    Represents a reference to a Kubernetes Secret for pulling container images.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    name: str
    """
    Name of the GPUStack Operator InstanceImagePullSecret to use for pulling container images.
    """


class GPUInstanceSpec(BaseModel):
    """
    Represents the specification for creating a GPU instance.
    """

    model_config = ConfigDict(
        alias_generator=pydantic_camel_case_generator,
        populate_by_name=True,
    )

    image: str
    """
    Container image of the GPU instance to use.
    """

    image_pull_policy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"
    """
    Container image pull policy for the GPU instance.
    Defaults to "IfNotPresent".
    """

    command: Optional[List[str]] = None
    """
    Command to run the GPU instance.
    If not specified, the default command from the image.
    """

    privileged: bool = False
    """
    Whether to run the GPU instance in privileged mode.
    Defaults to False.
    """

    ports: Optional[List[GPUInstancePort]] = None
    """
    List of port mappings for the GPU instance.
    """

    env: Optional[List[GPUInstanceEnvVar]] = None
    """
    List of environment variables for the GPU instance.
    """

    resources: Optional[GPUInstanceResources] = None
    """
    Resource requirements for the GPU instance,
    including CPU, RAM, local storage, and optional accelerator.
    """

    volume_mount: str = "/workspace"
    """
    The path inside the container where the GPU instance's volume will be mounted.
    Defaults to "/workspace".
    """

    image_pull_secret: Optional[GPUInstanceImagePullSecretReference] = None
    """
    Optional reference to a GPUStack Operator InstanceImagePullSecret for pulling container images.
    """


class GPUInstanceTemplateBase(SQLModel):
    """
    Base model for GPU instance templates, containing common fields.
    """

    # Every SSH Public Key belongs to one Org. The route layer fills this with
    # ctx.current_principal_id (or PLATFORM_PRINCIPAL_ID for admin in "All"
    # mode) when callers omit it.
    owner_principal_id: Optional[int] = Field(
        default=None, foreign_key="principals.id", nullable=False
    )

    name: str = Field(
        max_length=255,
    )
    """
    Name of the GPU instance template.
    Must be unique in the scope of the owning principal.
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
    )
    """
    Manufacturer of the GPU instance,
    e.g., "nvidia", "amd", "ascend".
    """

    spec: Optional[GPUInstanceSpec] = Field(
        sa_type=pydantic_column_type(GPUInstanceSpec),
        default=None,
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


class GPUInstanceTemplateListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "id",
        "name",
        "manufacturer",
        "created_at",
        "updated_at",
    ]


class GPUInstanceTemplateCreate(GPUInstanceTemplateBase):
    """
    Represents the fields required to create a new GPU instance template.
    """

    pass


class GPUInstanceTemplateUpdate(GPUInstanceTemplateBase):
    """
    Represents the fields that can be updated for a GPU instance template.
    """

    name: Optional[str] = None
    """
    Updated name of the GPU instance template. Must be unique if provided.
    """

    description: Optional[str] = None
    """
    Updated description of the GPU instance template.
    """

    manufacturer: Optional[str] = None
    """
    Updated manufacturer of the GPU instance.
    """

    spec: Optional[GPUInstanceSpec] = None
    """
    Updated specification for the GPU instance template.
    """


class GPUInstanceTemplatePublic(GPUInstanceTemplateBase, PublicFields):
    """
    Represents the public view of a GPU instance template,
    containing only fields that are safe to expose to clients.
    """

    pass


GPUInstanceTemplatesPublic = PaginatedList[GPUInstanceTemplatePublic]
