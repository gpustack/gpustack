"""Refresh the remote source slots on the leader.

Each kind of official content (catalog / community backend / built-in backend)
has a single OFFICIAL row served from the official OTA server. A round reads
the OTA server's index for every published document's ref and checksum, fetches the
ones that moved, and writes them for the controllers to reconcile. A user's own
URL source (opted in with ``auto_update_hours``) is refreshed the same way, but
by plain fetch.

Every stage below has a fast return, so the cost of a round scales with what
actually moved: with all three slots fallen back, one makes no request at all —
not even for the index.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type

import httpx
import yaml
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.datastructures import State

from gpustack import __version__
from gpustack.config.config import get_global_config
from gpustack.schemas.catalog_source import CatalogSource, normalize_catalog_yaml
from gpustack.schemas.inference_backend_source import (
    InferenceBackendSource,
    normalize_backend_yaml,
)
from gpustack.schemas.runner_source import InferenceRunnerSource, normalize_runner_json
from gpustack.server.catalog import get_builtin_model_catalog_file
from gpustack.server.db import async_session

from gpustack.schemas.source import SourceMixin, SourceTypeEnum

from .core import (
    MAX_REDIRECTS,
    MAX_SOURCE_BYTES,
    OFFICIAL_SOURCE_NAME,
    fetch_source_text,
    reject_a_forbidden_address,
    sha256_of,
)

logger = logging.getLogger(__name__)

# How often the thread wakes; each source still refreshes on its own
# ``auto_update_hours`` (tracked in ``_last_refresh_attempt``).
REFRESH_TICK_SECONDS = 60 * 60

# The cadence an OFFICIAL row is created with. A user URL source defaults to 0.
OFFICIAL_DEFAULT_HOURS = 12

# Per-operation (connect, each read), not whole-download.
_REFRESH_TIMEOUT_SECONDS = 30

_HEADERS = {"User-Agent": f"gpustack/{__version__}"}

# The directory the index and the published documents sit in — the only remote
# the OFFICIAL slots read. There is no fallback to the source repos, so an
# unreachable OTA server leaves the stored content alone and retries next tick.
# ``ota_server_url`` (config / CLI / ``GPUSTACK_OTA_SERVER_URL``)
# replaces the whole URL, so an OTA server of your own can live at any path.
OTA_SERVER_URL = "https://ota.gpustack.ai/latest"

# A ref and a sha256 per file, so one small fetch tells every kind whether its
# document moved.
_INDEX_FILENAME = "index.yaml"

# Documents track their source repo's latest release by default; one can be
# pinned to ``main`` to publish ahead of a release.
REF_KIND_RELEASE = "release"


class OfficialKind(NamedTuple):
    """One kind of official content: which table it lands in, how to validate it,
    and where it lives on the OTA server. Adding one needs an entry here, a
    ``SourceKind`` value, and its ``_SPECS`` binding.

    - ``repo``: the source repo it is published from
    - ``filename``: the file beside the index (catalog swaps in its ModelScope
      variant — see ``official_filename``)
    """

    name: str
    source_cls: Type[SourceMixin]
    normalize: Callable[[str], str]
    repo: str
    filename: str


class OfficialRef(BaseModel):
    """Where a published document came from: any ref string (a release tag, or a
    commit on ``main`` when pinned), which line it is on, and the file it was
    read from — catalog resolves its variant per round, so the name is only
    known afterwards."""

    ref: str
    ref_kind: str = REF_KIND_RELEASE
    filename: str = ""


OFFICIAL_KINDS = (
    OfficialKind(
        "catalog",
        CatalogSource,
        normalize_catalog_yaml,
        "gpustack/gpustack",
        "model-catalog.yaml",
    ),
    OfficialKind(
        "community-backend",
        InferenceBackendSource,
        normalize_backend_yaml,
        "gpustack/gpustack",
        "community-inference-backends.yaml",
    ),
    OfficialKind(
        "built-in-backend",
        InferenceRunnerSource,
        normalize_runner_json,
        "gpustack/runner",
        "runner.py.json",
    ),
)


class RefreshRound(BaseModel):
    """What one refresh round did. In memory on the leader only — a lost round
    costs one repeated round. Keys are an OFFICIAL kind's ``name``, or a user
    source's ``table:name``."""

    refreshed_at: Optional[datetime] = None
    # Whether the source row was written this round.
    changed: Dict[str, bool] = Field(default_factory=dict)
    # Why a source was rejected, its stored content left untouched.
    errors: Dict[str, str] = Field(default_factory=dict)
    # The ref the index reports, keyed by kind and not by repo: two documents from
    # one repo can sit on different lines once one is pinned to main.
    refs: Dict[str, OfficialRef] = Field(default_factory=dict)

    def absorb(self, scoped: "RefreshRound") -> None:
        """Fold a per-kind refresh into this round, so the status API keeps what
        the last full round found for the kinds it did not touch — and stops
        reporting an error the refreshed kind just cleared. Each key lands on one
        map only (a kind is either written or in error), hence the cross-removal.
        ``refreshed_at`` is left alone: it dates the last full round.
        """
        for key, changed in scoped.changed.items():
            self.changed[key] = changed
            self.errors.pop(key, None)
        for key, error in scoped.errors.items():
            self.errors[key] = error
            self.changed.pop(key, None)
        self.refs.update(scoped.refs)


# (table, name) → the ref its OFFICIAL row's content is at. Per kind, not per
# repo, so a masked sibling can't make an unmasked kind skip its fetch.
_applied_ref: Dict[Tuple[str, str], OfficialRef] = {}

# (table, name) → when a source was last attempted; gates ``auto_update_hours``
# in place of a persisted timestamp (a restart empties it → each source due once).
_last_refresh_attempt: Dict[Tuple[str, str], datetime] = {}

# First round after a start re-reads every auto-refreshing source (a new process
# may normalize differently); BUILTIN rows re-normalize on every leader start.
_revalidated_since_start = False


def _source_key(source_cls: Type[SourceMixin], name: str) -> Tuple[str, str]:
    return (source_cls.__tablename__, name)


def record_refresh_attempt(source_cls: Type[SourceMixin], name: str) -> None:
    """Note that a source's document was just fetched, so the schedule counts its
    cadence from here. The configuration API fetches too (PUT and reload); without
    this, the next tick finds no attempt on record and refetches at once.
    """
    _last_refresh_attempt[_source_key(source_cls, name)] = datetime.now(timezone.utc)


def applied_official_ref(kind: OfficialKind) -> Optional[OfficialRef]:
    """The ref this kind's stored OFFICIAL content came from. Process state, not
    round state: a skipped kind resolves no ref, yet its content is still at the
    last one."""
    return _applied_ref.get(_source_key(kind.source_cls, OFFICIAL_SOURCE_NAME))


def _is_due(key: Tuple[str, str], hours: int, now: datetime, revalidate: bool) -> bool:
    """Whether a source's cadence says to refresh it now. The first round after a
    start refreshes everything; otherwise a source is due ``hours`` after its
    last attempt (a never-attempted source is due immediately)."""
    if revalidate:
        return True
    last = _last_refresh_attempt.get(key)
    if last is None:
        return True
    return now - last >= timedelta(hours=hours)


async def _has_enabled_custom(
    session: AsyncSession, source_cls: Type[SourceMixin]
) -> bool:
    """Whether a user source (FILE/URL) is enabled for this table — in which case
    it masks OFFICIAL, so a freshly created OFFICIAL row must start disabled."""
    return any(
        source.enabled
        and source.source_type in (SourceTypeEnum.FILE, SourceTypeEnum.URL)
        for source in await source_cls.all(session)
    )


async def _ensure_official_row(
    session: AsyncSession, kind: OfficialKind, masked: bool
) -> SourceMixin:
    """The kind's OFFICIAL row, created if missing. This task is its only creator;
    a new row starts disabled when a user source masks it. ``masked`` is passed by
    the caller, which already computed it to re-assert the mask each round.
    """
    source = await kind.source_cls.one_by_field(session, "name", OFFICIAL_SOURCE_NAME)
    if source is not None:
        return source
    return await kind.source_cls.create(
        session,
        kind.source_cls(
            name=OFFICIAL_SOURCE_NAME,
            source_type=SourceTypeEnum.OFFICIAL,
            enabled=not masked,
            auto_update_hours=OFFICIAL_DEFAULT_HOURS,
        ),
    )


def _ota_url(filename: str, ota_server_url: Optional[str] = None) -> str:
    """Where a published file lives — the only place the OTA URL is assembled.
    ``ota_server_url`` is the configured override; ``None`` means the default OTA server.
    A trailing slash is tolerated, since a configured URL commonly carries one."""
    return f"{(ota_server_url or OTA_SERVER_URL).rstrip('/')}/{filename}"


@lru_cache(maxsize=1)
def _packaged_catalog_filename() -> str:
    """Which catalog variant this cluster resolved (Hugging Face or ModelScope).

    Memoized because the resolution costs two uncached HTTP probes and the answer
    is a property of the deployment, not of a round — recomputing it hourly
    burns six seconds forever on a cluster that can reach neither.
    """
    return os.path.basename(get_builtin_model_catalog_file())


async def official_filename(kind: OfficialKind) -> str:
    """The file this kind is published as on the OTA server — a basename is the whole
    path. Catalog swaps in its ModelScope variant; the rest are ``kind.filename``.

    Resolved here rather than read off the last round, so it is right for a masked
    kind too: that kind never enters a round, yet its published document is
    exactly what an admin downloads to edit.
    """
    if kind.name != "catalog":
        return kind.filename
    # Blocking on the first call only (the probe), off the loop either way.
    return await asyncio.to_thread(_packaged_catalog_filename)


async def official_document_url(kind: OfficialKind) -> str:
    """Where this kind's official document is published: what the OFFICIAL slot
    reads, and what an admin typing that address into a URL source of their own
    means to follow.
    """
    return _ota_url(await official_filename(kind), get_global_config().ota_server_url)


def _refresh_client() -> httpx.AsyncClient:
    """The client every refresh fetch goes through, so no path can quietly skip
    the redirect cap or the forbidden-address hook."""
    return httpx.AsyncClient(
        timeout=_REFRESH_TIMEOUT_SECONDS,
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
        event_hooks={"request": [reject_a_forbidden_address]},
    )


async def _fetch_raw(client: httpx.AsyncClient, url: str) -> str:
    """GET ``url`` and return its text, capped at ``MAX_SOURCE_BYTES``. Streamed so
    the cap applies while reading. ``ValueError`` on any transport/decode error."""
    chunks, total = [], 0
    try:
        async with client.stream("GET", url, headers=_HEADERS) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError(
                        f"document exceeds the "
                        f"{MAX_SOURCE_BYTES // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        raise ValueError(f"failed to fetch {url}: {e}")
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{url} is not valid UTF-8: {e}")


async def _fetch_ota_index(
    client: httpx.AsyncClient, ota_server_url: Optional[str] = None
) -> Dict[str, Any]:
    """The OTA server's index: a ref and a sha256 per published file. One small fetch
    serves every kind. ``ValueError`` when it cannot be fetched, is not YAML, or
    is not a mapping."""
    url = _ota_url(_INDEX_FILENAME, ota_server_url)
    raw = await _fetch_raw(client, url)
    try:
        index = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"{url} is not valid YAML: {e}")
    if not isinstance(index, dict):
        raise ValueError(f"{url} must be a YAML mapping")
    return index


async def _apply_raw(
    session: AsyncSession,
    kind: OfficialKind,
    source: SourceMixin,
    raw: str,
    revalidate: bool,
) -> bool:
    """Normalize and store ``raw`` on ``source``; True if the row was written.
    Two fast returns: the raw document hasn't moved (skip normalizing), or it
    normalizes to the stored text (writing would cost a pointless reconcile).
    """
    remote_hash = sha256_of(raw)
    if not revalidate and source.remote_hash == remote_hash:
        return False
    content = await asyncio.to_thread(kind.normalize, raw)
    content_hash = sha256_of(content)
    if source.content_hash == content_hash:
        return False
    await source.update(
        session,
        {"content": content, "content_hash": content_hash, "remote_hash": remote_hash},
    )
    return True


async def _due_official_rows(
    session: AsyncSession, now: datetime, ignore_cadence: bool
) -> List[Tuple[OfficialKind, SourceMixin]]:
    """The OFFICIAL rows to refresh this round, each created if missing.

    The mask is re-asserted every round, so a concurrent config write cannot leave
    OFFICIAL and a custom source both enabled. A slot that is not serving — masked,
    or turned off by the admin — is left out.
    """
    due: List[Tuple[OfficialKind, SourceMixin]] = []
    for kind in OFFICIAL_KINDS:
        masked = await _has_enabled_custom(session, kind.source_cls)
        source = await _ensure_official_row(session, kind, masked)
        # Only ever disables: ``enabled`` is also the admin's fall-back switch, so
        # a round must not turn a slot back on that someone turned off. This
        # direction is the one protecting an invariant — OFFICIAL and a custom
        # source must never both reach the merge.
        if masked and source.enabled:
            await source.update(session, {"enabled": False})
        if masked or not source.enabled:
            continue
        key = _source_key(kind.source_cls, OFFICIAL_SOURCE_NAME)
        # ``auto_update_hours`` gates whether this refreshes at all, so it is
        # honored even when the cadence itself is ignored.
        if source.auto_update_hours > 0 and _is_due(
            key, source.auto_update_hours, now, ignore_cadence
        ):
            due.append((kind, source))
    return due


async def _apply_official_kind(
    session: AsyncSession,
    client: httpx.AsyncClient,
    kind: OfficialKind,
    source: SourceMixin,
    index: Dict[str, Any],
    revalidate: bool,
    ota_server_url: Optional[str] = None,
) -> Tuple[bool, OfficialRef]:
    """Bring one kind up to what the OTA server publishes: whether the row was written,
    and the ref its content is now at. ``ValueError`` when the OTA server does not
    publish this kind's file, or serves one its index does not describe."""
    filename = await official_filename(kind)
    published = (index.get("files") or {}).get(filename) or {}
    ref, remote_hash = published.get("ref"), published.get("sha256")
    if not ref or not remote_hash:
        raise ValueError(f"the OTA server does not publish {filename}")
    # ``ref_kind`` is display metadata, so an index that omits it still applies;
    # the checksum below is what this round is actually trusted on.
    official_ref = OfficialRef(
        ref=str(ref),
        ref_kind=published.get("ref_kind") or REF_KIND_RELEASE,
        filename=filename,
    )
    # Unchanged checksum → this kind is current, no download.
    if not revalidate and source.remote_hash == remote_hash:
        return False, official_ref
    raw = await _fetch_raw(client, _ota_url(filename, ota_server_url))
    # A mismatch means the index and the document disagree, so the ref would
    # misreport what is stored; wait for a consistent round instead.
    if sha256_of(raw) != remote_hash:
        raise ValueError(
            f"{filename} does not match the sha256 the OTA server publishes"
        )
    return await _apply_raw(session, kind, source, raw, revalidate), official_ref


async def _refresh_official(
    session: AsyncSession,
    client: httpx.AsyncClient,
    now: datetime,
    revalidate: bool,
    result: RefreshRound,
    force: bool = False,
    ota_server_url: Optional[str] = None,
) -> None:
    """Refresh every enabled, due OFFICIAL row from the OTA server: one index fetch
    serves all of them, and a kind downloads only when the index says its document
    moved. ``force`` takes every enabled row as due — it skips the cadence, not
    the checksum.
    """
    due = await _due_official_rows(session, now, revalidate or force)
    if not due:
        return

    try:
        index = await _fetch_ota_index(client, ota_server_url)
    except ValueError as e:
        # No attempt recorded → the next tick retries, not a full-cadence backoff.
        for kind, _ in due:
            result.errors[kind.name] = str(e)
        # Louder than the per-kind failures below: this one took out every kind.
        logger.warning(f"Source refresh could not read the OTA server index: {e}")
        return

    for kind, source in due:
        key = _source_key(kind.source_cls, OFFICIAL_SOURCE_NAME)
        try:
            result.changed[kind.name], official_ref = await _apply_official_kind(
                session, client, kind, source, index, revalidate, ota_server_url
            )
            result.refs[kind.name] = official_ref
            _applied_ref[key] = official_ref
            _last_refresh_attempt[key] = now
        except Exception as e:
            # Any exception (not just ValueError): _apply_raw writes, so a
            # failure costs this kind alone; no attempt recorded → retry next tick.
            result.errors[kind.name] = str(e) or type(e).__name__
            logger.debug(f"Source refresh rejected official {kind.name}: {e}")


async def _refresh_user_urls(
    session: AsyncSession,
    now: datetime,
    revalidate: bool,
    result: RefreshRound,
    force: bool = False,
) -> None:
    """Refresh every enabled URL source that opted into auto-refresh and is due —
    a plain fetch, no index logic. ``force`` skips the cadence but not the opt-in:
    a source left at ``auto_update_hours=0`` is never fetched behind the user's
    back.
    """
    for kind in OFFICIAL_KINDS:
        for source in await kind.source_cls.all(session):
            if source.source_type != SourceTypeEnum.URL or not source.enabled:
                continue
            if source.auto_update_hours <= 0:
                continue
            key = _source_key(kind.source_cls, source.name)
            if not _is_due(key, source.auto_update_hours, now, revalidate or force):
                continue
            label = f"{kind.source_cls.__tablename__}:{source.name}"
            try:
                raw = await fetch_source_text(source.url)
                result.changed[label] = await _apply_raw(
                    session, kind, source, raw, revalidate
                )
                # Record on success only, so a failed fetch retries next tick.
                _last_refresh_attempt[key] = now
            except Exception as e:
                result.errors[label] = str(e) or type(e).__name__
                logger.debug(f"Source refresh rejected URL source {label}: {e}")


async def refresh_sources(
    session: AsyncSession,
    now: Optional[datetime] = None,
    force: bool = False,
    ota_server_url: Optional[str] = None,
) -> RefreshRound:
    """Run one refresh round: the OFFICIAL slots and the opted-in user URL sources.

    Whether a kind refreshes at all is its own ``auto_update_hours`` (0 = off) —
    there is no global switch. Failures are recorded, never raised: a round must
    never block start-up or a request path.

    - ``now``: injectable for tests
    - ``force``: a manual trigger; every enabled source counts as due. Not
      ``revalidate`` — the checksum fast returns stay, so an unmoved document
      still costs no download and no write.
    - ``ota_server_url``: overrides the OTA server base (``None`` = default)
    """
    global _revalidated_since_start
    revalidate = not _revalidated_since_start
    if now is None:
        now = datetime.now(timezone.utc)

    result = RefreshRound()
    async with _refresh_client() as client:
        await _refresh_official(
            session, client, now, revalidate, result, force, ota_server_url
        )
    await _refresh_user_urls(session, now, revalidate, result, force)

    result.refreshed_at = now
    _revalidated_since_start = True
    return result


async def refresh_official_kind(
    session: AsyncSession,
    kind: OfficialKind,
    ota_server_url: Optional[str] = None,
) -> Tuple[bool, OfficialRef]:
    """Bring one kind's OFFICIAL slot up to what the OTA server publishes, now:
    whether the row was written, and the ref its content is at.

    The manual per-kind refresh, behind that kind's own reload endpoint — which
    is why both cadence gates are skipped, ``_is_due`` and ``auto_update_hours``
    itself: that setting withholds consent from a *scheduled* round, and an admin
    pressing this kind's own button has given it. The checksum fast returns stay,
    so an unmoved document still costs no download and no write.

    Deliberately not routed through ``refresh_sources``: that would spend the
    first-round revalidation the other two kinds have not had yet.

    ``ValueError`` when the slot is not serving, or the document cannot be read —
    a manual refresh reports its failure, where "unchanged" would read as current.
    """
    # Only reached with no custom source configured, so a slot created here
    # starts in service.
    source = await _ensure_official_row(session, kind, masked=False)
    if not source.enabled:
        raise ValueError(
            "official content is out of service for this kind; "
            "put it back in service before refreshing it"
        )

    async with _refresh_client() as client:
        index = await _fetch_ota_index(client, ota_server_url)
        changed, official_ref = await _apply_official_kind(
            session,
            client,
            kind,
            source,
            index,
            revalidate=False,
            ota_server_url=ota_server_url,
        )

    key = _source_key(kind.source_cls, OFFICIAL_SOURCE_NAME)
    _applied_ref[key] = official_ref
    _last_refresh_attempt[key] = datetime.now(timezone.utc)
    return changed, official_ref


class SourceRefresher:
    """Runs the refresh on the leader: one round at start-up, then every
    ``REFRESH_TICK_SECONDS``. ``last_round`` is what the status API reports, and
    the manual trigger runs the same ``refresh_once``, with ``force``.

    There is nothing per-refresher to switch off — that lives on each OFFICIAL
    row: ``auto_update_hours`` for its cadence, ``enabled`` for whether the slot
    serves at all.
    """

    def __init__(self, ota_server_url: Optional[str] = None):
        self.ota_server_url = ota_server_url
        self.last_round: Optional[RefreshRound] = None
        self._round_lock = asyncio.Lock()

    async def refresh_once(self, force: bool = False) -> RefreshRound:
        # Manual trigger and schedule can overlap; two rounds would each insert a
        # missing OFFICIAL row (name uniqueness is ORM-only, not a DB index).
        async with self._round_lock:
            async with async_session() as session:
                self.last_round = await refresh_sources(
                    session,
                    force=force,
                    ota_server_url=self.ota_server_url,
                )
        return self.last_round

    async def refresh_kind_now(self, session: AsyncSession, kind: OfficialKind) -> bool:
        """Refresh one kind's OFFICIAL slot now, on the caller's session — the
        manual per-kind trigger, which serves a request and answers with what
        that kind's row ended up holding.

        Under the round lock, because a scheduled round writes the same row and
        both create it when missing. The outcome is folded into ``last_round``
        rather than replacing it: refreshing one kind must not erase what that
        round reported for the other two.
        """
        scoped = RefreshRound()
        try:
            async with self._round_lock:
                changed, ref = await refresh_official_kind(
                    session, kind, self.ota_server_url
                )
            scoped.changed[kind.name] = changed
            scoped.refs[kind.name] = ref
            return changed
        except ValueError as e:
            scoped.errors[kind.name] = str(e)
            raise
        finally:
            if self.last_round is None:
                self.last_round = RefreshRound()
            self.last_round.absorb(scoped)

    async def start(self) -> None:
        while True:
            try:
                await self.refresh_once()
            except Exception as e:
                # A round records its own failures; this catches only the
                # unexpected and must never end the loop.
                logger.error(f"Source refresh round failed: {e}")
            await asyncio.sleep(REFRESH_TICK_SECONDS)


def running_refresher(app_state: State) -> Optional[SourceRefresher]:
    """The refresher this server runs, if it is the leader. ``Server.
    _start_source_probe`` writes the key; the status route and the per-kind
    reload read it here rather than each knowing its name."""
    return getattr(app_state, "source_probe", None)
