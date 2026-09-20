"""The session-affinity plugin's own table — the external-plugin
storage pattern: plugin-owned, registered into the global SQLModel
metadata at import, additive-only. A built-in plugin's tables are part
of the central Alembic chain; a separately shipped plugin manages its
own schema and its code assumes the tables already exist (uninstalling
it means dropping the table by hand)."""

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


class SessionAffinityPolicy(BaseModelMixin, SQLModel, table=True):
    """One route's session-affinity configuration. Row presence means
    configured; ``config.enabled`` inside the JSON is the switch."""

    __tablename__: ClassVar[str] = "model_route_plugin_session_affinity"
    __table_args__ = (
        UniqueConstraint(
            "route_id", name="uix_model_route_plugin_session_affinity_route"
        ),
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
