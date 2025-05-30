from pydantic import ConfigDict
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
)


class ModelInstanceModelFileLink(SQLModel, table=True):
    model_instance_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())
