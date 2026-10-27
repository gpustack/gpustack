"""The capability plugins' shared policy storage.

One row per (capability, route): the capability name distinguishes the
plugin, the finisher-weight is a first-class column (it participates in
the cross-plugin weighted sum, so it is queryable rather than buried in
each plugin's JSON), and the ``config`` JSON carries the plugin-specific
fields only (``enabled``, ``sessionKeys``, ...). Built-in capability
tables are part of the central Alembic chain; a separately shipped
plugin manages its own schema and assumes the table exists.
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Float,
    Integer,
    JSON,
    String,
    SQLModel,
    UniqueConstraint,
    col,
)

from gpustack.mixins import BaseModelMixin

logger = logging.getLogger(__name__)


class CapabilityPolicy(BaseModelMixin, SQLModel, table=True):
    """One route's configuration for one LB capability plugin. Row
    presence means configured; ``config.enabled`` inside the JSON is the
    switch; ``weight`` is the finisher's weighted-sum contribution."""

    __tablename__: ClassVar[str] = "model_route_capability_policies"
    __table_args__ = (
        UniqueConstraint(
            "capability",
            "route_id",
            name="uix_capability_policy_capability_route",
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
    # explicit length: VARCHAR without one fails at CREATE TABLE on
    # MySQL-family dialects and openGauss; 64 is plenty for a plugin name
    capability: str = Field(sa_column=Column(String(64), nullable=False))
    weight: Optional[float] = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
        description=(
            "Contribution weight in the finisher's L1-weighted sum; NULL "
            "uses the plugin's compiled-in default."
        ),
    )
    config: dict = Field(sa_column=Column(JSON, nullable=False))


def section_from_policy(policy: CapabilityPolicy) -> Dict[str, Any]:
    """The plugin section a policy row represents: the config JSON with
    the weight column folded back in when set, so responses round-trip
    exactly like the section a client submitted."""
    section = dict(policy.config)
    if policy.weight is not None:
        section["weight"] = policy.weight
    return section


async def policy_for_route(
    session: AsyncSession, capability: str, route_id: int
) -> Optional[CapabilityPolicy]:
    return await CapabilityPolicy.one_by_fields(
        session,
        {"capability": capability, "route_id": route_id, "deleted_at": None},
    )


async def store_capability_policy(
    session: AsyncSession,
    capability: str,
    route_id: int,
    config: Dict[str, Any],
    weight: Optional[float],
) -> None:
    """Upsert one capability's row for a route. Runs inside the caller's
    transaction (no commit) — the policy must live and die with the
    route write that touched it."""
    source = {
        "capability": capability,
        "route_id": route_id,
        "config": config,
        "weight": weight,
    }
    existing = await policy_for_route(session, capability, route_id)
    if existing is None:
        await CapabilityPolicy.create(session=session, source=source, auto_commit=False)
    else:
        await existing.update(session=session, source=source, auto_commit=False)


async def delete_capability_policy(
    session: AsyncSession, capability: str, route_id: int
) -> None:
    """Hard delete, inside the caller's transaction: soft-deleting would
    leave a row that the (capability, route_id) unique constraint counts,
    blocking the policy from being added back after removal."""
    existing = await policy_for_route(session, capability, route_id)
    if existing is not None:
        await existing.delete(session=session, soft=False, auto_commit=False)


async def capability_sections(
    session: AsyncSession, capability: str, route_ids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """The plugin's response sections for a batch of routes."""
    if not route_ids:
        return {}
    policies = await CapabilityPolicy.all_by_fields(
        session,
        {"capability": capability, "deleted_at": None},
        extra_conditions=[col(CapabilityPolicy.route_id).in_(route_ids)],
    )
    return {p.route_id: section_from_policy(p) for p in policies}
