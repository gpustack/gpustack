from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.login_captcha import LoginCaptchaNonce
from gpustack.server import login_captcha_nonce_cleaner as cleaner_module
from gpustack.server.login_captcha_nonce_cleaner import LoginCaptchaNonceCleaner


def test_cleaner_rejects_invalid_interval():
    with pytest.raises(ValueError, match="positive"):
        LoginCaptchaNonceCleaner(interval=0)


@pytest.mark.asyncio
async def test_cleaner_deletes_only_expired_nonces(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(LoginCaptchaNonce.__table__.create)

    now = datetime.now(timezone.utc)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add_all(
            [
                LoginCaptchaNonce(
                    nonce_hash="a" * 64,
                    expires_at=now - timedelta(seconds=1),
                ),
                LoginCaptchaNonce(
                    nonce_hash="b" * 64,
                    expires_at=now + timedelta(minutes=5),
                ),
            ]
        )
        await session.commit()

    @asynccontextmanager
    async def session_factory():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setattr(cleaner_module, "async_session", session_factory)
    try:
        await LoginCaptchaNonceCleaner().cleanup_once()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            rows = (await session.exec(select(LoginCaptchaNonce))).all()
        assert [row.nonce_hash for row in rows] == ["b" * 64]
    finally:
        await engine.dispose()
