"""Report and trigger the source refresher.

Per-kind fields live on the source rows, so any server reports them; round-level
state only exists where the refresher runs, so this API says which server that is
rather than reporting silence as success (as ``routes/update.py`` does).
"""

from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.config.config import get_global_config
from gpustack.server.deps import SessionDep
from gpustack.schemas.source import SourceTypeEnum
from gpustack.server.sources.core import OFFICIAL_SOURCE_NAME
from gpustack.server.sources.probe import (
    OFFICIAL_KINDS,
    OTA_SERVER_URL,
    RefreshRound,
    SourceRefresher,
    applied_official_ref,
    official_filename,
    running_refresher,
)

router = APIRouter()


class SourceKindStatus(BaseModel):
    """One kind's *active* remote source (the custom source when it masks
    OFFICIAL, else the OFFICIAL slot) plus the tag its stored content came from
    and the last round's error (both only where the refresher runs)."""

    source_type: Optional[SourceTypeEnum] = None
    # None for OFFICIAL, which reads the OTA server rather than a URL of the user's;
    # a failing round reports the OTA server URL in ``error``.
    url: Optional[str] = None
    official_masked: bool = False
    # Whether any remote layer serves this kind. False is the fall-back state:
    # the packaged baseline alone, with the documents parked for the way back.
    remote_enabled: bool = True
    auto_update_hours: int = 0
    # Join onto ``ota_server_url`` to link straight at the published document — the
    # starting point for a custom source, which replaces it.
    filename: str
    # Both set only when OFFICIAL is the active source (leader-only, best-effort).
    # Any ref string, not just a release tag: ``effective_ref_kind`` says which
    # line it is on.
    effective_tag: Optional[str] = None
    effective_ref_kind: Optional[str] = None
    remote_hash: Optional[str] = None
    content_hash: Optional[str] = None
    updated_at: Optional[datetime] = None
    error: Optional[str] = None


class SourceRefreshStatus(BaseModel):
    """Where the refresher runs, and each kind's slot. Whether official refresh
    is on is per kind, reported as that kind's ``auto_update_hours``."""

    # The directory the official documents are published under, after any
    # ``ota_server_url`` override — so a client builds a download link
    # without knowing which OTA server this cluster reads.
    ota_server_url: str
    # Whether this server runs the refresher — else a standby's empty round
    # fields look like a refresher that found nothing.
    refreshing_on_this_server: bool
    kinds: Dict[str, SourceKindStatus] = Field(default_factory=dict)
    refreshed_at: Optional[datetime] = None


async def source_probe_status(
    session: AsyncSession, refresher: Optional[SourceRefresher]
) -> SourceRefreshStatus:
    last_round = refresher.last_round if refresher else None
    kinds = {}
    for kind in OFFICIAL_KINDS:
        rows = await kind.source_cls.all(session)
        official = next((row for row in rows if row.name == OFFICIAL_SOURCE_NAME), None)
        # A custom source masks OFFICIAL, so it is the active remote source.
        custom = next(
            (
                row
                for row in rows
                if row.enabled
                and row.source_type in (SourceTypeEnum.FILE, SourceTypeEnum.URL)
            ),
            None,
        )
        active = custom or official
        # Errors key by kind name for OFFICIAL, ``table:name`` for a user source.
        error_key = (
            f"{kind.source_cls.__tablename__}:{custom.name}"
            if custom is not None
            else kind.name
        )
        applied = applied_official_ref(kind)
        # Serving, not merely present: a document an admin fell back from is
        # parked rather than discarded, so a row alone says nothing about what
        # runs. ``custom`` above is already the *enabled* one, so a parked
        # document leaves both of these false — the fall-back state.
        official_serving = custom is None and official is not None and official.enabled
        remote_enabled = custom is not None or official_serving
        # The filename is reported whichever source is active — the published
        # document is exactly what an admin downloads to edit *while* their own is
        # configured — hence resolving it rather than reading the last round, which
        # a masked kind never enters. The ref describes stored OFFICIAL content, so
        # it is reported only while OFFICIAL is serving.
        kinds[kind.name] = SourceKindStatus(
            source_type=active.source_type if active else None,
            url=active.url if active else None,
            official_masked=custom is not None,
            remote_enabled=remote_enabled,
            auto_update_hours=active.auto_update_hours if active else 0,
            filename=await official_filename(kind),
            effective_tag=applied.ref if applied and official_serving else None,
            effective_ref_kind=(
                applied.ref_kind if applied and official_serving else None
            ),
            remote_hash=active.remote_hash if active else None,
            content_hash=active.content_hash if active else None,
            updated_at=active.updated_at if active else None,
            error=last_round.errors.get(error_key) if last_round else None,
        )

    return SourceRefreshStatus(
        ota_server_url=(get_global_config().ota_server_url or OTA_SERVER_URL).rstrip(
            "/"
        ),
        refreshing_on_this_server=refresher is not None,
        kinds=kinds,
        refreshed_at=last_round.refreshed_at if last_round else None,
    )


async def run_source_probe(refresher: Optional[SourceRefresher]) -> RefreshRound:
    """Run one round now, through the schedule's own code but forced: every
    enabled source counts as due, so the trigger really does check instead of
    no-op'ing inside its cadence. It skips the cadence only — an unmoved document
    still costs no download, and a kind that opted out stays out.

    Every kind at once, which is what makes this the operator's verb; refreshing
    one kind is that kind's own ``POST /v2/ota-sources/{kind}/reload``."""
    if refresher is None:
        raise ServiceUnavailableException(
            message="The source refresher runs on the leader; this server is a standby"
        )
    return await refresher.refresh_once(force=True)


@router.get("", response_model=SourceRefreshStatus)
async def get_status(request: Request, session: SessionDep) -> SourceRefreshStatus:
    return await source_probe_status(session, running_refresher(request.app.state))


@router.post("", response_model=RefreshRound)
async def probe_now(request: Request) -> RefreshRound:
    return await run_source_probe(running_refresher(request.app.state))
