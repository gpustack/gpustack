"""Deleting a decision-service provider routes still pin.

A route's decision-service policy stores the provider id verbatim, so a
delete would leave the UI rendering a bare id over a gone provider. The
delete endpoint refuses while any route's policy pins the provider.
"""

from typing import List, Optional

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import model_provider as route_module
from gpustack.schemas.model_provider import (
    ModelProviderTypeEnum,
    TypesafeConfig,
)


class _Policy:
    def __init__(
        self, route_id: int, provider_id: Optional[int], deleted: bool = False
    ):
        self.route_id = route_id
        self.config = {"providerId": provider_id}
        self.deleted_at = True if deleted else None


class _Route:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name


class _Provider:
    def __init__(self, config, deleted: bool = False):
        self.id = 7
        self.name = "jev"
        self.config = config
        self.deleted_at = True if deleted else None
        self.deleted = False

    async def delete(self, session):
        self.deleted = True


def _decision_config() -> TypesafeConfig:
    return TypesafeConfig.model_validate(
        {
            "type": ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE.value,
            "endpoint": "http://jev.example",
        }
    )


@pytest.fixture
def _stubs(monkeypatch):
    """Wire the endpoint's lookups onto in-memory rows and drop tenancy."""

    policies: List[_Policy] = []
    routes: dict = {}
    provider = _Provider(_decision_config())
    seen: dict = {}
    state = {
        "policies": policies,
        "routes": routes,
        "provider": provider,
        "seen": seen,
    }

    async def provider_by_id(session, id, for_update=False, options=None):
        return state["provider"]

    async def all_policies(session, fields=None):
        rows = list(state["policies"])
        if fields and fields.get("deleted_at", None) is None:
            rows = [p for p in rows if p.deleted_at is None]
        return rows

    async def all_routes(session, fields=None, extra_conditions=None, **kwargs):
        seen["extra_conditions"] = extra_conditions
        # the id-membership condition is evaluated by the real query; the
        # stub returns the rows the policies already pin
        return list(state["routes"].values())

    monkeypatch.setattr(route_module.ModelProvider, "one_by_id", provider_by_id)
    monkeypatch.setattr(route_module.CapabilityPolicy, "all_by_fields", all_policies)
    monkeypatch.setattr(route_module.ModelRoute, "all_by_fields", all_routes)
    monkeypatch.setattr(route_module, "assert_resource_visible", lambda *a, **k: None)
    monkeypatch.setattr(route_module, "tenant_list_conditions", lambda ctx, model: [])
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_kind",
    ["decision", "inference"],
)
async def test_delete_blocked_only_when_decision_routes_reference(_stubs, config_kind):
    state = _stubs
    if config_kind == "inference":
        state["provider"].config = {"type": "openai"}

    state["policies"].append(_Policy(route_id=3, provider_id=7))
    state["routes"][3] = _Route(3, "my-route")

    if config_kind == "decision":
        with pytest.raises(BadRequestException) as exc:
            await route_module.delete_model_provider(session=None, ctx=None, id=7)
        assert "my-route" in str(vars(exc.value).get("message", ""))
        assert not state["provider"].deleted
    else:
        await route_module.delete_model_provider(session=None, ctx=None, id=7)
        assert state["provider"].deleted


@pytest.mark.asyncio
async def test_decision_provider_without_references_deletes(_stubs):
    await route_module.delete_model_provider(session=None, ctx=None, id=7)
    assert _stubs["provider"].deleted


@pytest.mark.asyncio
async def test_soft_deleted_policy_does_not_block_delete(_stubs):
    """A soft-deleted capability policy keeps its stale providerId but no
    longer pins the provider: it must not show up as a live reference."""
    _stubs["policies"].append(_Policy(route_id=3, provider_id=7, deleted=True))
    _stubs["routes"][3] = _Route(3, "my-route")
    await route_module.delete_model_provider(session=None, ctx=None, id=7)
    assert _stubs["provider"].deleted


@pytest.mark.asyncio
async def test_reference_lookup_is_scoped_to_the_caller(_stubs, monkeypatch):
    """CapabilityPolicy rows have no owner column, so the route lookup that
    turns them into names carries the caller's tenant conditions -- a
    foreign policy must not surface another Org's route names."""
    _stubs["policies"].append(_Policy(route_id=3, provider_id=7))
    _stubs["routes"][3] = _Route(3, "my-route")
    monkeypatch.setattr(
        route_module,
        "tenant_list_conditions",
        lambda ctx, model: ["OWNED-BY-CALLER"],
    )
    with pytest.raises(BadRequestException):
        await route_module.delete_model_provider(session=None, ctx=None, id=7)
    conditions = _stubs["seen"]["extra_conditions"]
    assert "OWNED-BY-CALLER" in conditions
    assert _stubs["provider"].deleted is False
