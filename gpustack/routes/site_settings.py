"""Public branding reads and platform-admin configuration writes."""

from datetime import datetime, timezone
from hashlib import sha256

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlmodel import select

from gpustack.api.auth import get_admin_user
from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.schemas.site_settings import (
    SiteSettings,
    SiteSettingsPublic,
    SiteSettingsUpdate,
)
from gpustack.server.deps import SessionDep

router = APIRouter(dependencies=[Depends(get_admin_user)])
public_router = APIRouter()


async def _read_settings(session: SessionDep) -> SiteSettingsPublic:
    row = await session.get(SiteSettings, 1)
    return SiteSettingsPublic.model_validate(row) if row else SiteSettingsPublic()


@router.get("", response_model=SiteSettingsPublic)
async def get_site_settings(
    session: SessionDep, response: Response
) -> SiteSettingsPublic:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await _read_settings(session)
    except SQLAlchemyError as exc:
        raise ServiceUnavailableException(
            message="Site settings are temporarily unavailable."
        ) from exc


@public_router.get("/public", response_model=SiteSettingsPublic)
async def get_public_site_settings(
    session: SessionDep, request: Request, response: Response
) -> SiteSettingsPublic | Response:
    # Browsers must revalidate on reload so saved branding does not stay stale.
    response.headers["Cache-Control"] = "no-cache"
    try:
        settings = await _read_settings(session)
        digest = sha256(settings.model_dump_json().encode()).hexdigest()
        etag = f'W/"{digest}"'
        response.headers['ETag'] = etag
        candidates = request.headers.get('If-None-Match', '').split(',')
        if any(
            tag.strip() == '*' or tag.strip().removeprefix('W/') == etag[2:]
            for tag in candidates
        ):
            return Response(
                status_code=304,
                headers={'Cache-Control': 'no-cache', 'ETag': etag},
            )
        return settings
    except SQLAlchemyError as exc:
        raise ServiceUnavailableException(
            message="Site settings are temporarily unavailable."
        ) from exc


@router.put("", response_model=SiteSettingsPublic)
async def update_site_settings(
    values: SiteSettingsUpdate, session: SessionDep, response: Response
) -> SiteSettingsPublic:
    response.headers["Cache-Control"] = "no-store"
    statement = (
        select(SiteSettings)
        .where(SiteSettings.id == 1)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    try:
        # A non-locking existence check avoids InnoDB gap locks on a missing
        # singleton. Insert contenders are serialized by the primary key.
        row = await session.get(SiteSettings, 1)
        if row is None:
            # The primary key arbitrates simultaneous first saves. A savepoint
            # keeps the outer transaction usable after the losing insert.
            try:
                async with session.begin_nested():
                    row = SiteSettings()
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                row = (await session.exec(statement)).one()
        else:
            # Refresh the instance loaded before the lock: another writer may
            # have committed while this transaction waited to acquire it.
            row = (await session.exec(statement)).one()
        changes = values.model_dump()
        if any(getattr(row, key) != value for key, value in changes.items()):
            for key, value in changes.items():
                setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
        await session.flush()
        result = SiteSettingsPublic.model_validate(row)
        await session.commit()
        return result
    except SQLAlchemyError as exc:
        await session.rollback()
        raise ServiceUnavailableException(
            message="Site settings could not be saved. Please try again later."
        ) from exc
