"""Watch replay and bus events obey the caller's tenant visibility."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.tenant import TenantContext
from gpustack.mixins import active_record
from gpustack.routes import cloud_credentials, model_provider, worker_pools
from gpustack.schemas.clusters import (
    CloudCredential,
    CloudCredentialListParams,
    WorkerPool,
)
from gpustack.schemas.common import ListParams
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderListParams,
    OpenAIConfig,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.bus import Event, EventType


def _context(scope):
    return TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM if "system" in scope else PrincipalType.USER,
        ),
        is_platform_admin=scope.startswith("admin"),
        current_principal_id=(
            None
            if scope in ("admin-all", "system", "scoped-system")
            else 7 if scope == "personal" else 101
        ),
        org_role=OrgRole.OWNER if scope == "org" else None,
        current_is_personal_scope=scope == "personal",
        scoped_cluster_id=11 if scope == "scoped-system" else None,
    )


def _row(model, row_id, owner_id, cluster_id, name="shared"):
    fields = dict(
        id=row_id,
        name=name,
        owner_principal_id=owner_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    if model is CloudCredential:
        fields.update(key="credential-identifier", secret="private-cloud-secret")
    elif model is WorkerPool:
        fields.update(
            cluster_id=cluster_id,
            instance_type="test-instance",
            os_image="test-os",
            image_name="test-image",
        )
    else:
        fields.update(
            config=OpenAIConfig(type="openai"), api_tokens=["private-provider-token"]
        )
    return model(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,route,params_class",
    [
        (CloudCredential, cloud_credentials.list, CloudCredentialListParams),
        (WorkerPool, worker_pools.list, ListParams),
        (ModelProvider, model_provider.get_model_providers, ModelProviderListParams),
    ],
)
@pytest.mark.parametrize(
    "scope", ["org", "personal", "admin-all", "admin-org", "system", "scoped-system"]
)
@pytest.mark.parametrize("filters", [{}, {"name": "shared"}, {"search": "share"}])
async def test_watch_filters_replay_and_bus_events(
    monkeypatch, model, route, params_class, scope, filters
):
    # Owners represent two organizations and platform-owned infrastructure.
    rows = [
        _row(model, 1, 101, 11),
        _row(model, 2, 202, 22),
        _row(model, 3, 303, 33),
        _row(model, 4, 101, 11, name="other"),
    ]
    if scope in ("org", "personal", "admin-org"):
        rows.append(_row(model, 5, None, 11))
    cached_all = AsyncMock(return_value=rows)
    monkeypatch.setattr(model, "cached_all", cached_all)
    events = []
    for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED):
        for row in rows:
            data = row.model_copy()
            if event_type == EventType.DELETED:
                data.deleted_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
            events.append(Event(type=event_type, data=data))
    # ID-only deletes cannot establish ownership for a tenant-scoped caller.
    if scope in ("org", "personal", "admin-org") or (
        scope == "scoped-system" and model is WorkerPool
    ):
        events.append(Event(type=EventType.DELETED, data={"id": 99}))
    events.append(Event(type=EventType.HEARTBEAT, data=None))
    subscriber = SimpleNamespace(
        receive=AsyncMock(side_effect=[*events, asyncio.CancelledError()])
    )
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)

    response = await route(
        ctx=_context(scope), params=params_class(watch=True), **filters
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(frame)

    expected_ids = [1, 4]
    if scope == "personal":
        expected_ids = []
    elif scope in ("admin-all", "system") or (
        scope == "scoped-system" and model is not WorkerPool
    ):
        expected_ids = [1, 2, 3, 4]
    if filters:
        expected_ids = [row_id for row_id in expected_ids if row_id != 4]

    assert frames[-1] == "\n\n"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert [(event["type"], event["data"]["id"]) for event in payloads] == [
        (event_type.value, row_id)
        for event_type in (
            EventType.CREATED,  # Cached replay.
            EventType.CREATED,
            EventType.UPDATED,
            EventType.DELETED,
        )
        for row_id in expected_ids
    ]
    assert "private-cloud-secret" not in "".join(frames)
    assert "private-provider-token" not in "".join(frames)
    cached_all.assert_awaited_once()
    bus.unsubscribe.assert_called_once_with(model.__name__.lower(), subscriber)
