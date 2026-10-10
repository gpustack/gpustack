"""Authorization is mandatory and precedes business filters and projection."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gpustack.api.streaming import tenant_streaming
from gpustack.api.tenant import TenantContext
from gpustack.schemas.clusters import CloudCredential
from gpustack.schemas.model_routes import ModelRouteTarget
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import Event, EventBus, EventType


@pytest.fixture
def ctx():
    return TenantContext(
        user=SimpleNamespace(kind=PrincipalType.USER),
        is_platform_admin=False,
        current_principal_id=10,
        org_role=None,
    )


def test_missing_context_is_rejected():
    with pytest.raises(ValueError, match="tenant context"):
        tenant_streaming(CloudCredential, None, visibility_filter=lambda row: True)


def test_parent_owned_model_requires_an_explicit_policy(ctx):
    with pytest.raises(ValueError, match="explicit visibility filter"):
        tenant_streaming(ModelRouteTarget, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_visibility", [False, True])
@pytest.mark.parametrize("async_business", [False, True])
async def test_visibility_precedes_business_and_projection(
    monkeypatch, ctx, async_visibility, async_business
):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        CloudCredential(
            id=i,
            name=f"credential-{i}",
            owner_principal_id=owner,
            created_at=now,
            updated_at=now,
        )
        for i, owner in [(1, 20), (2, 10), (3, 10)]
    ]

    async def subscribe(**kwargs):
        for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED):
            for row in rows:
                yield Event(type=event_type, data=row)

    monkeypatch.setattr(CloudCredential, "subscribe", subscribe)
    mock_visibility = AsyncMock if async_visibility else Mock
    visible = mock_visibility(side_effect=lambda row: row.owner_principal_id == 10)
    mock_business = AsyncMock if async_business else Mock
    business = mock_business(side_effect=lambda row: row.id == 2)
    transform = AsyncMock()
    frames = [
        json.loads(frame)
        async for frame in tenant_streaming(
            CloudCredential,
            ctx,
            visibility_filter=visible,
            filter_func=business,
            event_transform=transform,
        )
    ]
    assert [frame["data"]["id"] for frame in frames] == [2, 2, 2]
    assert [call.args[0].id for call in business.call_args_list] == [2, 3] * 3
    assert transform.await_count == 3


@pytest.mark.asyncio
async def test_business_filter_cannot_override_default_visibility(monkeypatch, ctx):
    async def subscribe(**kwargs):
        yield Event(
            type=EventType.DELETED,
            data=SimpleNamespace(id=1, owner_principal_id=20),
        )

    monkeypatch.setattr(CloudCredential, "subscribe", subscribe)
    business = Mock(return_value=True)
    assert [
        frame
        async for frame in tenant_streaming(CloudCredential, ctx, filter_func=business)
    ] == []
    business.assert_not_called()


@pytest.mark.asyncio
async def test_visibility_lookup_failure_emits_no_event(monkeypatch, ctx, caplog):
    async def subscribe(**kwargs):
        yield Event(type=EventType.DELETED, data={"id": 1})

    monkeypatch.setattr(CloudCredential, "subscribe", subscribe)
    assert [
        frame
        async for frame in tenant_streaming(
            CloudCredential,
            ctx,
            visibility_filter=AsyncMock(side_effect=RuntimeError("lookup unavailable")),
        )
    ] == []
    assert any(
        record.levelno == logging.ERROR
        and "Error in streaming CloudCredential: lookup unavailable" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_business", [False, True])
async def test_replay_applies_visibility_and_business_filters_and_cleans_up(
    monkeypatch, ctx, async_business
):
    bus = EventBus()
    monkeypatch.setattr("gpustack.mixins.active_record.event_bus", bus)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        CloudCredential(
            id=i,
            name=name,
            owner_principal_id=owner,
            created_at=now,
            updated_at=now,
        )
        for i, name, owner in [
            (3, "other", 10),
            (4, "match-four", 20),
            (1, "match-one", 10),
            (2, "match-two", 10),
        ]
    ]
    loader = AsyncMock(return_value=rows)
    monkeypatch.setattr(CloudCredential, "cached_all", loader)
    visible = AsyncMock(side_effect=lambda row: row.owner_principal_id == 10)
    business = (AsyncMock if async_business else Mock)(
        side_effect=lambda row: row.id == 2
    )
    transform = AsyncMock()
    stream = tenant_streaming(
        CloudCredential,
        ctx,
        fields={"deleted_at": None},
        fuzzy_fields={"name": "match"},
        visibility_filter=visible,
        filter_func=business,
        event_transform=transform,
    )
    try:
        frame = json.loads(await asyncio.wait_for(anext(stream), timeout=1))
        assert frame["data"]["id"] == 2
        assert frame["type"] == EventType.CREATED.value
        assert type(transform.call_args.args[0]) is Event
    finally:
        await stream.aclose()
    loader.assert_awaited_once()
    assert [call.args[0].id for call in visible.call_args_list] == [4, 1, 2]
    assert [call.args[0].id for call in business.call_args_list] == [1, 2]
    transform.assert_awaited_once()
    assert not bus.subscribers


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_replay_failure_cleans_up(monkeypatch, ctx, cancelled):
    bus = EventBus()
    monkeypatch.setattr("gpustack.mixins.active_record.event_bus", bus)

    async def load(**kwargs):
        assert bus.subscribers["cloudcredential"]
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("snapshot unavailable")

    cached = AsyncMock(side_effect=load)
    monkeypatch.setattr(CloudCredential, "cached_all", cached)
    stream = tenant_streaming(CloudCredential, ctx)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await anext(stream)
    else:
        assert [frame async for frame in stream] == []
    cached.assert_awaited_once()
    assert not bus.subscribers
