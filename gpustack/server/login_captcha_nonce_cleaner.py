"""Periodic cleanup for the shared login CAPTCHA nonce ledger."""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import delete

from gpustack.schemas.login_captcha import LoginCaptchaNonce
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)
DEFAULT_CLEANUP_INTERVAL_SECONDS = 5 * 60


class LoginCaptchaNonceCleaner:
    """Remove expired nonce hashes without adding writes to the login path."""

    def __init__(self, interval: int = DEFAULT_CLEANUP_INTERVAL_SECONDS):
        if interval < 1:
            raise ValueError("Cleanup interval must be positive")
        self._interval = interval

    async def start(self) -> None:
        while True:
            try:
                await self.cleanup_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to clean expired login CAPTCHA nonces")
            await asyncio.sleep(self._interval)

    async def cleanup_once(self) -> None:
        async with async_session() as session:
            await session.exec(
                delete(LoginCaptchaNonce).where(
                    LoginCaptchaNonce.expires_at <= datetime.now(timezone.utc)
                )
            )
            await session.commit()
