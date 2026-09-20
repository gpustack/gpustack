"""The least-load plugin's own table — the external-plugin storage
pattern, same as session-affinity's: plugin-owned, additive-only. A
built-in plugin's tables are part of the central Alembic chain; a
separately shipped plugin manages its own schema and assumes the
tables already exist."""

from typing import ClassVar, Optional

from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    JSON,
    SQLModel,
    UniqueConstraint,
)

from gpustack.mixins import BaseModelMixin


class LeastLoadPolicy(BaseModelMixin, SQLModel, table=True):
    """One route's least-load configuration. Row presence means
    configured; ``config.enabled`` inside the JSON is the switch."""

    __tablename__: ClassVar[str] = "model_route_plugin_least_load"
    __table_args__ = (
        UniqueConstraint("route_id", name="uix_model_route_plugin_least_load_route"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    route_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("model_routes.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    config: dict = Field(sa_column=Column(JSON, nullable=False))
