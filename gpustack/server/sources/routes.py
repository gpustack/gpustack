import asyncio
from datetime import datetime
from typing import Awaitable, Callable, List, NamedTuple, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException, ServiceUnavailableException

from gpustack.schemas.source import SourceContent, SourceMixin, SourceTypeEnum

from .core import (
    OFFICIAL_SOURCE_NAME,
    fetch_source_text,
    order_source_contents,
    sha256_of,
)
from .probe import (
    OFFICIAL_DEFAULT_HOURS,
    OFFICIAL_KINDS,
    SourceRefresher,
    official_document_url,
    record_refresh_attempt,
)

# Called with the merge input a write would produce (already ordered), before
# that write lands. Raises to reject it; a spec without one skips the check.
PreWriteCheck = Callable[[AsyncSession, List[SourceContent]], Awaitable[None]]


class SourceConfigSpec(NamedTuple):
    """One source table's binding for the shared operations.

    - ``custom_name``: the admin's row
    - ``builtin_name``: the packaged baseline (``None`` for runner, whose
      baseline is in-code)
    - ``official_name``: the platform's remote row
    - ``allowed_types``: what a client may configure

    Both baselines are masked while a custom source is set — a custom document
    replaces them outright.
    """

    source_cls: Type[SourceMixin]
    normalize: Callable[[str], str]
    custom_name: str
    builtin_name: Optional[str] = None
    official_name: str = OFFICIAL_SOURCE_NAME
    allowed_types: Tuple[SourceTypeEnum, ...] = (
        SourceTypeEnum.FILE,
        SourceTypeEnum.URL,
    )
    pre_write_check: Optional[PreWriteCheck] = None


class CustomSourceState(BaseModel):
    """The admin's own document — the whole content for this kind once set."""

    source_type: SourceTypeEnum
    url: Optional[str] = None
    # Returned so the UI can edit a saved FILE source — its only copy lives here.
    content: Optional[str] = None
    # Refresh cadence in hours (0 = off); only a URL source can opt in.
    auto_update_hours: int = 0
    # When that content was taken (a URL is fetched at PUT / reload time).
    updated_at: Optional[datetime] = None
    content_hash: Optional[str] = None


class OfficialSourceState(BaseModel):
    """The platform's remote slot. Its document is written by the refresh task,
    so the cadence is the only part an admin sets; the rest is state to show
    beside it."""

    auto_update_hours: int
    # Derived, not a setting: false while a custom document replaces the slot, or
    # while remote content is off altogether.
    enabled: bool
    updated_at: Optional[datetime] = None
    content_hash: Optional[str] = None


class SourceConfig(BaseModel):
    """A kind's whole source configuration — exactly what the screen shows.

    - ``custom`` set: the admin's document replaces both the packaged baseline
      and the official slot
    - ``custom`` null: the official slot serves
    - ``remote_enabled`` false: neither does, and the packaged baseline is on
      its own
    """

    # Whether remote content serves at all. False is the fall-back state: the
    # packaged baseline alone, with both remote documents kept for the way back.
    remote_enabled: bool = True
    custom: Optional[CustomSourceState] = None
    official: OfficialSourceState


class SourceWriteResult(SourceConfig):
    """A write's outcome plus ``changed``: false when the new text hashed the
    same as the stored one (nothing written).
    """

    changed: bool = False


class CustomSourceUpsert(BaseModel):
    """The custom half of a write."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceTypeEnum = SourceTypeEnum.FILE
    content: Optional[str] = None
    url: Optional[str] = None
    # Refresh cadence in hours, 0 = off. Only a URL source can opt in.
    auto_update_hours: int = Field(default=0, ge=0)


class OfficialSourceUpsert(BaseModel):
    """The official half of a write: the cadence and nothing else. ``extra=
    "forbid"`` is the write protection — a request carrying ``url`` or ``content``
    is refused rather than silently ignored."""

    model_config = ConfigDict(extra="forbid")

    # 0 turns this kind's official refresh off.
    auto_update_hours: int = Field(ge=0)


class SourceConfigUpsert(BaseModel):
    """Request body: a full replacement of this kind's source configuration.

    Both halves are required: the screen this serves saves both at once, and a
    partial body would make "switch back to official" a second request.
    ``custom: null`` *is* that switch — it drops the configured source and lets
    the platform layers serve again.

    Every model here forbids extra keys, so a misspelled field is named rather
    than dropped, which would read as a write that took.
    """

    model_config = ConfigDict(extra="forbid")

    # False falls this kind back to the packaged baseline — the escape hatch for
    # remote content that turns out to be wrong. Nothing is discarded: both remote
    # documents are parked (kept, out of the merge), so switching back needs no
    # fetch. Distinct from ``official.auto_update_hours: 0``, which stops the
    # updates and keeps serving what is stored.
    remote_enabled: bool = True
    custom: Optional[CustomSourceUpsert]
    official: OfficialSourceUpsert


def _remote_enabled(
    custom: Optional[SourceMixin], official: Optional[SourceMixin]
) -> bool:
    """Whether remote content is in service, read off whichever row carries it:
    the configured document while there is one, else the official slot.
    ``_read_remote_enabled`` applies the same rule to rows read without documents.
    """
    if custom is not None:
        return custom.enabled
    return official.enabled if official is not None else True


async def _read_remote_enabled(session: AsyncSession, spec: SourceConfigSpec) -> bool:
    """``_remote_enabled`` straight from the database, reading the flags alone.

    The change detection needs two booleans, while the ``content`` beside them is
    a merged catalog — hundreds of KB the request would carry twice for nothing.
    """
    rows = await session.exec(
        select(spec.source_cls.name, spec.source_cls.enabled).where(
            spec.source_cls.name.in_((spec.custom_name, spec.official_name))
        )
    )
    flags = dict(rows.all())
    if spec.custom_name in flags:
        return flags[spec.custom_name]
    return flags.get(spec.official_name, True)


def _config(
    custom: Optional[SourceMixin], official: Optional[SourceMixin]
) -> SourceConfig:
    return SourceConfig(
        remote_enabled=_remote_enabled(custom, official),
        custom=(
            None
            if custom is None
            else CustomSourceState(
                source_type=custom.source_type,
                url=custom.url,
                content=custom.content,
                auto_update_hours=custom.auto_update_hours,
                updated_at=custom.updated_at,
                content_hash=custom.content_hash,
            )
        ),
        official=OfficialSourceState(
            # No row yet: the refresh task will create one at the default
            # cadence, which is what to report until then.
            auto_update_hours=(
                official.auto_update_hours
                if official is not None
                else OFFICIAL_DEFAULT_HOURS
            ),
            enabled=(
                official.enabled
                if official is not None
                else custom is None or not custom.enabled
            ),
            updated_at=official.updated_at if official is not None else None,
            content_hash=official.content_hash if official is not None else None,
        ),
    )


async def _set_builtin_enabled(
    session: AsyncSession,
    spec: SourceConfigSpec,
    enabled: bool,
    auto_commit: bool = True,
) -> None:
    """Drive the baseline through the BUILTIN row's ``enabled`` flag. Creates a
    placeholder row when the leader has not seeded yet — otherwise the seed's
    create branch would overwrite the admin's choice. No-op for runner.
    """
    if spec.builtin_name is None:
        return
    builtin = await spec.source_cls.one_by_field(session, "name", spec.builtin_name)
    if builtin is not None:
        if builtin.enabled != enabled:
            await builtin.update(session, {"enabled": enabled}, auto_commit=auto_commit)
        return
    await spec.source_cls.create(
        session,
        spec.source_cls(
            name=spec.builtin_name,
            source_type=SourceTypeEnum.BUILTIN,
            content=None,
            enabled=enabled,
        ),
        auto_commit=auto_commit,
    )


async def _set_row_enabled(
    session: AsyncSession,
    spec: SourceConfigSpec,
    name: str,
    enabled: bool,
    auto_commit: bool = True,
) -> None:
    """Toggle whether a row takes part in the merge. Existing rows only — the
    refresh task owns creating OFFICIAL, and a custom row is created by the write
    that configures it.
    """
    source = await spec.source_cls.one_by_field(session, "name", name)
    if source is not None and source.enabled != enabled:
        await source.update(session, {"enabled": enabled}, auto_commit=auto_commit)


async def _apply_layer_states(
    session: AsyncSession,
    spec: SourceConfigSpec,
    has_custom: bool,
    remote_enabled: bool,
    auto_commit: bool = True,
) -> None:
    """Resolve which layers serve, from the two facts a write carries: whether a
    document is configured, and whether remote content is in service at all.

    A configured document replaces both platform layers — but only while it is
    serving, so the packaged baseline takes over the moment it is parked. Runner
    has no BUILTIN row (its baseline is in-code), where the setter no-ops.

    Grow before shrink: everything that ends up on is switched on first, so no
    intermediate commit holds less than the final state. The leader may reconcile
    any of them, and shrinking first would flash an empty merge. ``auto_commit``
    false leaves the flags to a caller landing its own write in the same
    transaction, where that ordering stops mattering.
    """
    custom_serving = has_custom and remote_enabled
    builtin_on = not custom_serving
    official_on = remote_enabled and not has_custom

    if builtin_on:
        await _set_builtin_enabled(session, spec, True, auto_commit)
    if official_on:
        await _set_row_enabled(session, spec, spec.official_name, True, auto_commit)
    if custom_serving:
        await _set_row_enabled(session, spec, spec.custom_name, True, auto_commit)

    if not builtin_on:
        await _set_builtin_enabled(session, spec, False, auto_commit)
    if not official_on:
        await _set_row_enabled(session, spec, spec.official_name, False, auto_commit)
    if has_custom and not remote_enabled:
        await _set_row_enabled(session, spec, spec.custom_name, False, auto_commit)


async def _check_before_write(
    session: AsyncSession,
    spec: SourceConfigSpec,
    custom_content: Optional[str] = None,
    custom_source_type: SourceTypeEnum = SourceTypeEnum.FILE,
    remote_enabled: bool = True,
) -> None:
    """Let the table's owner reject a write by the merge input it would produce,
    ordered exactly as the real merge orders it. What that input holds:

    - ``custom_content`` set and serving: that document alone
    - otherwise (dropped, or parked out of the merge): the packaged baseline,
      plus the official slot unless ``remote_enabled`` is false
    """
    if spec.pre_write_check is None:
        return

    # Dropped here and added back below per the outcome, so the check sees exactly
    # the post-write merge rather than today's rows.
    managed = {spec.builtin_name, spec.custom_name, spec.official_name}
    proposed = [
        source
        for source in await spec.source_cls.all(session)
        if source.name not in managed and source.enabled
    ]
    if custom_content is not None and remote_enabled:
        # A transient row (never added to the session) carrying only what the
        # merge reads: identity, type, content.
        proposed.append(
            spec.source_cls(
                name=spec.custom_name,
                source_type=custom_source_type,
                content=custom_content,
            )
        )
    else:
        returning = [spec.builtin_name]
        if remote_enabled:
            returning.append(spec.official_name)
        for name in returning:
            if name is None:
                continue
            # Included even if currently masked — this write unmasks it.
            baseline = await spec.source_cls.one_by_field(session, "name", name)
            if baseline is not None:
                proposed.append(baseline)
    await spec.pre_write_check(session, order_source_contents(proposed))


async def _read_rows(
    session: AsyncSession, spec: SourceConfigSpec
) -> Tuple[Optional[SourceMixin], Optional[SourceMixin]]:
    """The custom and official rows, either of which may not exist yet."""
    return (
        await spec.source_cls.one_by_field(session, "name", spec.custom_name),
        await spec.source_cls.one_by_field(session, "name", spec.official_name),
    )


async def get_source_config(
    session: AsyncSession, spec: SourceConfigSpec
) -> SourceConfig:
    return _config(*await _read_rows(session, spec))


async def _apply_official_settings(
    session: AsyncSession,
    spec: SourceConfigSpec,
    source_in: OfficialSourceUpsert,
    remote_enabled: bool,
) -> None:
    """Set how often this kind's official slot refreshes — 0 to turn it off.

    Creates the row when the refresh task has not yet, so an air-gapped cluster can
    opt out before the first round rather than after it. An existing row's
    ``enabled`` belongs to ``_apply_layer_states``; only a row created here carries
    both.
    """
    official = await spec.source_cls.one_by_field(session, "name", spec.official_name)
    if official is None:
        custom = await spec.source_cls.one_by_field(session, "name", spec.custom_name)
        await spec.source_cls.create(
            session,
            spec.source_cls(
                name=spec.official_name,
                source_type=SourceTypeEnum.OFFICIAL,
                enabled=remote_enabled and custom is None,
                auto_update_hours=source_in.auto_update_hours,
            ),
        )
    elif official.auto_update_hours != source_in.auto_update_hours:
        await official.update(
            session, {"auto_update_hours": source_in.auto_update_hours}
        )


async def _park_custom_source(
    session: AsyncSession,
    spec: SourceConfigSpec,
    custom: Optional[SourceMixin],
    custom_in: CustomSourceUpsert,
) -> bool:
    """Store a configured document out of service, without reading it. Answers
    whether anything moved.

    - FILE: carries its own text, so it is normalized as in any other write
    - URL, unchanged: keeps the document it was parked with
    - URL, pointed elsewhere: stored with its document cleared — nothing it could
      serve while parked, and the write that puts it back in service reads it
    """
    if custom_in.source_type == SourceTypeEnum.FILE:
        try:
            content = await asyncio.to_thread(spec.normalize, custom_in.content or "")
        except ValueError as e:
            raise BadRequestException(message=f"Invalid source content: {e}")
        document = {
            "content": content,
            "content_hash": sha256_of(content),
            "remote_hash": None,
        }
    elif (
        custom is not None
        and custom.source_type == SourceTypeEnum.URL
        and custom.url == custom_in.url
    ):
        # Still the URL it was parked with, so keep that document.
        document = {
            "content": custom.content,
            "content_hash": custom.content_hash,
            "remote_hash": custom.remote_hash,
        }
    else:
        document = {"content": None, "content_hash": None, "remote_hash": None}

    update = {
        "source_type": custom_in.source_type,
        "url": custom_in.url,
        "enabled": False,
        "auto_update_hours": custom_in.auto_update_hours,
        **document,
    }
    if custom is None:
        await spec.source_cls.create(
            session, spec.source_cls(name=spec.custom_name, **update)
        )
        return True
    if all(getattr(custom, field) == value for field, value in update.items()):
        return False
    await custom.update(session, update)
    return True


async def _drop_custom_source(
    session: AsyncSession, spec: SourceConfigSpec, remote_enabled: bool = True
) -> Optional[SourceMixin]:
    """Restore the platform layers and drop the configured document, in one
    transaction.

    Atomic because restoring the layers does not disable the custom row (the drop
    is what removes it), so a commit in between would hold the custom source and
    OFFICIAL both enabled — which the refresh task reads as "OFFICIAL is masked"
    and disables, and it only ever disables.

    ``remote_enabled`` false leaves the official slot out of the restoration — the
    packaged baseline alone is what comes back.

    The check runs even with no document to drop: taking remote content out of
    service shrinks the merge on its own.
    """
    custom = await spec.source_cls.one_by_field(session, "name", spec.custom_name)
    await _check_before_write(session, spec, remote_enabled=remote_enabled)
    await _apply_layer_states(
        session,
        spec,
        has_custom=False,
        remote_enabled=remote_enabled,
        auto_commit=False,
    )
    if custom is not None:
        await custom.delete(session, auto_commit=False)
    await session.commit()
    return custom


async def _means_the_official_source(
    spec: SourceConfigSpec, custom: Optional[CustomSourceUpsert]
) -> bool:
    """Whether a configured URL is just the official document's own address.

    Typing it in is how an admin says "follow the official source", but storing it
    would mask that slot and serve the same file the weaker way: a plain GET, with
    no index checksum, no sha256 verification and no release to track.

    Decided at write time, so what lands in the table is the *state* — a mirror
    reconfigured later moves this kind with it, instead of a stored string quietly
    ceasing to match. Only this cluster's own catalog variant counts: naming the
    other one means that file.
    """
    if custom is None or custom.source_type != SourceTypeEnum.URL or not custom.url:
        return False
    kind = next((k for k in OFFICIAL_KINDS if k.source_cls is spec.source_cls), None)
    if kind is None:
        return False
    # Only whitespace and a trailing slash are normalized away: a missed match
    # still behaves correctly, a false one would serve the wrong document.
    official = await official_document_url(kind)
    return custom.url.strip().rstrip("/") == official.rstrip("/")


async def update_source_config(  # noqa: C901
    session: AsyncSession, spec: SourceConfigSpec, source_in: SourceConfigUpsert
) -> SourceWriteResult:
    """Replace this kind's whole source configuration in one write.

    - ``custom`` null: switch back to the platform layers
    - ``custom`` set: fetch and validate the document once here (``POST /reload``
      reuses the stored URL), and let it replace them
    - ``remote_enabled`` false: store the document without reading it, leaving
      the packaged baseline serving on its own

    The official settings are applied in every case, so the screen saves it all in
    one request — but *last*, so a request whose document turns out to be
    unfetchable applies nothing at all rather than only the cadence.
    """
    # A URL naming the official document takes that path, not a masking copy.
    if await _means_the_official_source(spec, source_in.custom):
        source_in = source_in.model_copy(update={"custom": None})

    remote_enabled = source_in.remote_enabled
    # Moves what is served without any document changing, so it counts as a
    # change of its own.
    toggled = (await _read_remote_enabled(session, spec)) != remote_enabled

    if source_in.custom is None:
        dropped = await _drop_custom_source(
            session, spec, remote_enabled=remote_enabled
        )
        await _apply_official_settings(
            session, spec, source_in.official, remote_enabled
        )
        return await _write_result(
            session, spec, changed=dropped is not None or toggled
        )

    custom_in = source_in.custom
    if custom_in.source_type not in spec.allowed_types:
        allowed = " or ".join(
            f"'{source_type.value}'" for source_type in spec.allowed_types
        )
        raise BadRequestException(message=f"source_type must be {allowed}")

    if not remote_enabled:
        # Parking never reaches the network: fetching here would refuse the very
        # request meant to escape unreadable content, and this same write is how a
        # parked kind is reconfigured while that URL is still down.
        custom = await spec.source_cls.one_by_field(session, "name", spec.custom_name)
        await _check_before_write(session, spec, remote_enabled=False)
        # Layers first, so the content only moves while the row is already out
        # of the merge.
        await _apply_layer_states(
            session, spec, has_custom=custom is not None, remote_enabled=False
        )
        parked = await _park_custom_source(session, spec, custom, custom_in)
        await _apply_official_settings(session, spec, source_in.official, False)
        return await _write_result(session, spec, changed=toggled or parked)

    try:
        if custom_in.source_type == SourceTypeEnum.URL:
            raw = await fetch_source_text(custom_in.url)
        else:
            raw = custom_in.content or ""
        # CPU-bound, and alias expansion makes the byte cap no work cap.
        content = await asyncio.to_thread(spec.normalize, raw)
    except ValueError as e:
        raise BadRequestException(message=f"Invalid source content: {e}")

    content_hash = sha256_of(content)
    custom = await spec.source_cls.one_by_field(session, "name", spec.custom_name)
    changed = custom is None or custom.content_hash != content_hash
    # The configuration can move without the content moving (FILE→equivalent URL).
    reconfigured = custom is not None and (
        custom.source_type != custom_in.source_type
        or custom.url != custom_in.url
        or custom.auto_update_hours != custom_in.auto_update_hours
    )
    # Both cases that shrink the merge: a first custom document masking the
    # baselines (always ``changed`` — no stored row to hash against), and one
    # coming back out of the fall-back, which the baseline steps aside for.
    if changed or toggled:
        await _check_before_write(
            session,
            spec,
            custom_content=content,
            custom_source_type=custom_in.source_type,
        )

    if changed or reconfigured:
        update = {
            "source_type": custom_in.source_type,
            "content": content,
            "content_hash": content_hash,
            # Only a URL source has a remote document to compare against.
            "remote_hash": (
                sha256_of(raw) if custom_in.source_type == SourceTypeEnum.URL else None
            ),
            "url": custom_in.url,
            "enabled": True,
            "auto_update_hours": custom_in.auto_update_hours,
        }
        if custom is not None:
            await custom.update(session, update)
        else:
            custom = await spec.source_cls.create(
                session, spec.source_cls(name=spec.custom_name, **update)
            )

    # Mask the baselines last: each is its own commit, and disabling them shrinks
    # the merge, so the custom row must already be in place.
    await _apply_layer_states(session, spec, has_custom=True, remote_enabled=True)
    await _apply_official_settings(session, spec, source_in.official, True)
    if custom_in.source_type == SourceTypeEnum.URL:
        # This call fetched it, so the cadence runs from here.
        record_refresh_attempt(spec.source_cls, spec.custom_name)
    return await _write_result(session, spec, changed=changed or toggled)


async def delete_source_config(
    session: AsyncSession, spec: SourceConfigSpec
) -> SourceConfig:
    """Restore the factory state: drop the configured document and put every
    platform layer back in service, including one the admin had fallen back from —
    this is the "everything as shipped" verb. The cadence is left as configured.
    To fall back instead, `PUT` ``remote_enabled: false``, which keeps the
    document."""
    await _drop_custom_source(session, spec)
    return _config(*await _read_rows(session, spec))


async def _write_result(
    session: AsyncSession, spec: SourceConfigSpec, changed: bool
) -> SourceWriteResult:
    """A write's outcome, with both rows re-read so the response reflects what
    this call just applied — the cadence, and which layers ended up serving."""
    return SourceWriteResult(
        **_config(*await _read_rows(session, spec)).model_dump(), changed=changed
    )


async def _reload_official(
    session: AsyncSession,
    spec: SourceConfigSpec,
    refresher: Optional[SourceRefresher],
) -> SourceWriteResult:
    """Refresh this kind's OFFICIAL slot — what a reload means while no document
    of the admin's own is configured, so one endpoint per kind refreshes whatever
    that kind serves and cannot reach the other two.

    Leader-only: the ref a slot's content came from and its cadence are process
    state on the server running the refresher, so refreshing anywhere else would
    leave the reported ref describing content that has already moved.
    """
    kind = next((k for k in OFFICIAL_KINDS if k.source_cls is spec.source_cls), None)
    if kind is None:
        raise BadRequestException(message="This kind has no official source")
    if refresher is None:
        raise ServiceUnavailableException(
            message="The source refresher runs on the leader; this server is a standby"
        )
    try:
        changed = await refresher.refresh_kind_now(session, kind)
    except ValueError as e:
        raise BadRequestException(message=f"Failed to refresh official content: {e}")
    return await _write_result(session, spec, changed=changed)


async def reload_source_config(
    session: AsyncSession,
    spec: SourceConfigSpec,
    refresher: Optional[SourceRefresher],
) -> SourceWriteResult:
    """Refresh this kind now, from whichever layer serves it: the configured URL
    source while there is one, else the OFFICIAL slot.

    Both paths skip the cadence and keep the same two fast returns:
    ``remote_hash`` match → nothing to normalize; ``content_hash`` match →
    nothing to write.
    """
    custom = await spec.source_cls.one_by_field(session, "name", spec.custom_name)
    if custom is None:
        return await _reload_official(session, spec, refresher)
    if custom.source_type != SourceTypeEnum.URL:
        raise BadRequestException(
            message="Only a URL source can be reloaded; "
            "update a file source by PUTting new content"
        )

    try:
        raw = await fetch_source_text(custom.url)
        # In hand, so the cadence runs from here whatever the rest of this call
        # does with it — a fast return still means current.
        record_refresh_attempt(spec.source_cls, spec.custom_name)
        remote_hash = sha256_of(raw)
        if remote_hash == custom.remote_hash:
            return await _write_result(session, spec, changed=False)
        content = await asyncio.to_thread(spec.normalize, raw)
    except ValueError as e:
        raise BadRequestException(message=f"Invalid source content: {e}")

    content_hash = sha256_of(content)
    if content_hash == custom.content_hash:
        # Moved, but normalizes to the same text: writing would cost a pointless
        # reconcile.
        return await _write_result(session, spec, changed=False)

    await _check_before_write(
        session,
        spec,
        custom_content=content,
        custom_source_type=custom.source_type,
        # A parked document is not in the merge, so refreshing it takes nothing
        # away.
        remote_enabled=custom.enabled,
    )
    await custom.update(
        session,
        {
            "content": content,
            "content_hash": content_hash,
            "remote_hash": remote_hash,
        },
    )
    return await _write_result(session, spec, changed=True)
