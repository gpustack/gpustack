"""``APIKeyService.get_by_access_key`` is the query behind ``/token-auth``'s
credential path, which every request falls back to when the gateway's local key
table cannot answer. It must agree with that table on what a live key is, so it
runs against a real database here rather than a stubbed session."""

from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.security import new_secret_key_digest
from gpustack.server.cache import delete_cache_by_key
from gpustack.server.services import APIKeyService

NOW = datetime(2026, 5, 26, 12, 0, 0)

LIVE_ACCESS_KEY = "3192253c1f4a9b7e"
DELETED_ACCESS_KEY = "7e4a9b3192253c1f"
SECRET_KEY = "c11c75ed6334ea9505da4ad9c11c75ed"


def _key(id_, access_key, deleted_at=None):
    return ApiKey(
        id=id_,
        name=f"key{id_}",
        access_key=access_key,
        hashed_secret_key=f"argon2-hash-{id_}",
        secret_key_digest=new_secret_key_digest(
            secret_key=SECRET_KEY, is_custom=False, access_key=access_key
        ),
        scope=[PermissionScope.ALL],
        user_id=7,
        is_custom=False,
        created_at=NOW,
        updated_at=NOW,
        deleted_at=deleted_at,
    )


async def _drop_cache(*access_keys):
    for access_key in access_keys:
        await delete_cache_by_key(APIKeyService.get_by_access_key, access_key)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(ApiKey.__table__.create)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        s.add_all(
            [
                _key(1, LIVE_ACCESS_KEY),
                _key(2, DELETED_ACCESS_KEY, deleted_at=NOW),
            ]
        )
        await s.commit()
        # The read-through cache is process-global, so entries another test
        # left behind would answer before the query ever runs.
        await _drop_cache(LIVE_ACCESS_KEY, DELETED_ACCESS_KEY)
        yield s
        await _drop_cache(LIVE_ACCESS_KEY, DELETED_ACCESS_KEY)
    await engine.dispose()


@pytest.mark.asyncio
async def test_get_by_access_key_returns_a_live_key(session):
    found = await APIKeyService(session).get_by_access_key(LIVE_ACCESS_KEY)

    assert found is not None
    assert found.id == 1


@pytest.mark.asyncio
async def test_get_by_access_key_ignores_a_soft_deleted_key(session):
    """A soft-deleted row is gone as far as authentication is concerned. The
    gateway's local table filters it out, so a fallback path that still
    resolved it would authenticate a key nothing else honours."""
    assert await APIKeyService(session).get_by_access_key(DELETED_ACCESS_KEY) is None


@pytest.mark.asyncio
async def test_soft_deleting_a_key_stops_it_authenticating(session, monkeypatch):
    """End to end through the credential path: the same token authenticates
    before the row is soft-deleted and is refused after, so the refusal is the
    deletion and not a malformed credential."""
    from gpustack.api.auth import get_user_from_api_token

    principal = type("User", (), {"is_active": True, "id": 7})()

    async def fake_get_by_id(self, user_id):
        assert user_id == 7
        return principal

    monkeypatch.setattr("gpustack.api.auth.UserService.get_by_id", fake_get_by_id)

    token = f"gpustack_{LIVE_ACCESS_KEY}_{SECRET_KEY}"
    user, api_key = await get_user_from_api_token(session, token)
    assert user is principal and api_key is not None

    live = await ApiKey.one_by_id(session, 1)
    await live.delete(session, soft=True)
    await _drop_cache(LIVE_ACCESS_KEY)

    user, api_key = await get_user_from_api_token(session, token)
    assert user is None and api_key is None


@pytest.mark.asyncio
async def test_soft_deleted_key_stays_refused_after_the_cache_is_dropped(session):
    """The cache is not what does the rejecting: drop the entry and the query
    itself still refuses to hand the row back."""
    row = await ApiKey.one_by_id(session, 1)
    row.deleted_at = NOW
    session.add(row)
    await session.commit()
    await _drop_cache(LIVE_ACCESS_KEY)

    assert await APIKeyService(session).get_by_access_key(LIVE_ACCESS_KEY) is None
