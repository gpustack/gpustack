"""Streaming writers for tabular exports (CSV, zip-of-CSV, xlsx).

Shared by the usage and resource-usage export routes so both produce
byte-identical file shapes. Nothing here knows about usage rows: callers hand
over columns plus an async row iterator and get an async byte iterator back,
ready for ``StreamingResponse``.

Why CSV is the primary format and xlsx the compatibility one:

* CSV appends row by row, so it streams with constant memory and has no row
  ceiling.
* xlsx is a zip of XML parts whose central directory can only be finalized
  once every part is written, and a worksheet holds at most 1,048,576 rows.
  ``xlsxwriter``'s ``constant_memory`` mode keeps row data out of RAM, but the
  archive itself is still assembled before the first byte can be sent — so the
  xlsx path buffers where the CSV path streams. That is acceptable precisely
  because the format caps out around a million rows anyway.
"""

import asyncio
import contextlib
import csv
import io
import time
import zipfile
from datetime import date, datetime
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

CSV_UTF8_BOM = b"\xef\xbb\xbf"

# Columns are plain strings, and whatever the caller passes is written
# verbatim. The usage exports pass readable-but-never-localized titles
# ("User ID"), keeping the machine keys behind them on a separate channel
# (``export_column_keys``) for the preview — a header that changed with the
# viewer's language would break exactly the reconciliation scripts these files
# exist for. That stability is the caller's contract to keep, not something
# this module enforces; the same reasoning keeps zip member names in one
# vocabulary.


class ExportStageTimer:
    """Wall-clock accounting for one streaming export.

    The row generator spends most of its life SUSPENDED at ``yield``, while
    CSV encoding, zip deflate and the client's download speed run, so one
    end-to-end number charges all of that to the query. Stages are timed
    individually instead, and whatever they don't claim is reported as the
    derived ``downstream`` remainder — so a forgotten cost shows up as a gap
    rather than hiding.
    """

    def __init__(self):
        self._stages: Dict[str, float] = {}
        self._cpu = 0.0
        self._started = time.monotonic()

    @contextlib.contextmanager
    def stage(self, name: str):
        """Charge the wrapped block to ``name`` (repeat calls accumulate)."""
        started = time.monotonic()
        cpu_started = time.thread_time()
        try:
            yield
        finally:
            self._stages[name] = self._stages.get(name, 0.0) + (
                time.monotonic() - started
            )
            self._cpu += time.thread_time() - cpu_started

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def breakdown(self) -> Dict[str, float]:
        """Every stage, plus CPU, the ``downstream`` remainder, and the total.

        ``cpu`` is the event-loop THREAD's CPU time across the timed stages —
        an upper bound on this export's own, since the loop runs other
        coroutines while a stage is suspended. Read as a bound it still
        answers the question that matters: cpu ≪ wall means the stage spent
        its time waiting, not executing, so there is nothing here to optimize.
        """
        elapsed = self.elapsed
        measured = sum(self._stages.values())
        return {
            **self._stages,
            "cpu": self._cpu,
            "downstream": max(0.0, elapsed - measured),
            "total": elapsed,
        }

    def summary(self) -> str:
        """One decimal is deliberate: these are seconds-scale questions."""
        return " ".join(f"{name}={s:.1f}s" for name, s in self.breakdown().items())


class _ChunkBuffer(io.RawIOBase):
    """File-like sink that accumulates writes for a generator to drain.

    ``csv.writer`` and ``zipfile.ZipFile`` both want a writable file object,
    but a ``StreamingResponse`` wants an iterator. This adapter bridges the
    two: library code writes into it, the generator pops what accumulated and
    yields it, keeping peak memory at roughly one chunk.
    """

    def __init__(self):
        self._parts: List[bytes] = []

    def writable(self) -> bool:  # pragma: no cover - trivial
        return True

    def write(self, data) -> int:
        payload = bytes(data)
        self._parts.append(payload)
        return len(payload)

    def drain(self) -> bytes:
        if not self._parts:
            return b""
        payload = b"".join(self._parts)
        self._parts.clear()
        return payload


# A leading one of these makes Excel / LibreOffice parse the field as a
# formula when the CSV is opened by double-click, quoted or not.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_cell(value: Any) -> Any:
    """Render one cell.

    ``None`` becomes an empty field rather than the string ``"None"``.
    Numbers, dates and booleans are written as-is — no thousands separators,
    no reformatting — so they stay machine-readable.

    A STRING that opens with a formula trigger gets a leading apostrophe:
    entity names reach these cells verbatim and are not character-restricted
    (a volume may be named ``=HYPERLINK("http://…")``), so otherwise a tenant
    could plant a formula that runs when an operator double-clicks the file —
    the threat the xlsx path answers with ``write_string``. Only ``str``
    values are touched, so a negative number is still a number.
    """
    if value is None:
        return ""
    if isinstance(value, str) and value[:1] in _CSV_FORMULA_PREFIXES:
        return f"'{value}"
    return value


async def stream_csv(
    columns: Sequence[str],
    rows: AsyncIterator[Sequence[Any]],
    *,
    trailer: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Yield a UTF-8 CSV with a BOM, streaming row by row.

    The BOM is what makes Excel recognize UTF-8 on a double-click; without it
    non-ASCII headers arrive mojibake on Windows.

    ``trailer`` is the only defense against the silent-truncation failure mode
    of streaming: the status code is committed before the first row, so a
    mid-stream error yields a well-formed file that is missing rows. It goes
    out THROUGH the csv writer, padded to the header width — CSV has no
    comment syntax, and a one-field last line is a parse error for the strict
    readers these files exist for.

    ``rows`` is closed when this generator finishes or is closed early, so an
    abandoned download doesn't leave the row source (and its cursor) open.
    """
    buffer = _ChunkBuffer()
    # ``write_through`` stops TextIOWrapper from holding rows back until an
    # arbitrary flush boundary — every written row must reach the chunk buffer
    # so the generator can yield it.
    text = io.TextIOWrapper(buffer, encoding="utf-8", newline="", write_through=True)
    writer = csv.writer(text)

    yield CSV_UTF8_BOM
    columns = list(columns)
    writer.writerow(columns)
    yield buffer.drain()

    async with _closing(rows):
        async for row in rows:
            writer.writerow([_csv_cell(value) for value in row])
            chunk = buffer.drain()
            if chunk:
                yield chunk

    if trailer:
        writer.writerow([f"# {trailer}"] + [""] * (len(columns) - 1))
    text.flush()
    tail = buffer.drain()
    if tail:
        yield tail


async def take_rows(
    rows: AsyncIterator[Sequence[Any]], count: int
) -> AsyncIterator[Sequence[Any]]:
    """Pull at most ``count`` rows off a shared iterator, then stop.

    Splitting a large export into files is done by chopping the ROW STREAM,
    not by re-querying narrower date ranges. One cursor feeds every part in a
    single pass, which is what makes the parts cheap (one aggregate instead of
    one per part) and consistent (one snapshot, so a row written between two
    parts cannot be duplicated or lost).

    ``stream_zip`` consumes its members one at a time, so handing each member
    a ``take_rows`` view of the same iterator hands out consecutive slices.

    Deliberately does NOT close ``rows`` on the way out: the iterator is shared
    with the parts that follow, so closing it here would end the export at the
    first part. Whoever owns the shared iterator closes it once every part has
    been written — see the split export responses.
    """
    taken = 0
    async for row in rows:
        yield row
        taken += 1
        if taken >= count:
            return


@contextlib.asynccontextmanager
async def _closing(iterator):
    """Close ``iterator`` on the way out, if it is closeable.

    ``async for`` does not close what it iterates, so an abandoned consumer
    leaves a generator suspended at its ``yield`` — holding whatever it opened
    (here: database cursors and sessions) until the event loop finalizes it.
    Plain iterables have nothing to close and pass through untouched.
    """
    try:
        yield iterator
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def _as_async_iterator(members) -> AsyncIterator[Any]:
    """View a plain iterable of members as an async one."""
    for member in members:
        yield member


async def stream_zip(
    members: Union[
        Iterable[Tuple[str, AsyncIterator[bytes]]],
        AsyncIterator[Tuple[str, AsyncIterator[bytes]]],
    ],
) -> AsyncIterator[bytes]:
    """Yield a zip archive built from named byte streams, without buffering it.

    ``zipfile`` detects that the sink is not seekable and switches to data
    descriptors, so the archive can be produced in one forward pass.

    ``members`` may be an async iterator, which is how a caller keeps a
    resource alive across several members (one cursor feeding every part of a
    split export). Both it and each member's byte stream are closed when this
    generator ends — including when the client disconnects mid-archive, which
    is the case that would otherwise strand an open cursor until the event
    loop got around to finalizing an abandoned generator.
    """
    buffer = _ChunkBuffer()
    if not hasattr(members, "__aiter__"):
        members = _as_async_iterator(members)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        async with _closing(members) as member_stream:
            async for name, chunks in member_stream:
                async with _closing(chunks):
                    with archive.open(name, mode="w") as member:
                        async for chunk in chunks:
                            member.write(chunk)
                            payload = buffer.drain()
                            if payload:
                                yield payload
                payload = buffer.drain()
                if payload:
                    yield payload
    tail = buffer.drain()
    if tail:
        yield tail


# Enough for "2026-08-12" and "2026-08-12 10:30:00" plus Excel's cell padding.
# A too-narrow date column renders as "########" — Excel refuses to truncate a
# number the way it truncates text, so the value becomes unreadable rather
# than merely clipped.
_DATE_WIDTH = 12
_DATETIME_WIDTH = 21
# Text and numbers fall back to their header, which is the longest thing in
# most columns ("Input Tokens Cached"). Clamped so a stray long name can't
# push a column off the screen.
_MIN_WIDTH = 10
_MAX_WIDTH = 40


def _column_widths(
    columns: Sequence[str], sample: Optional[Sequence[Any]]
) -> List[int]:
    """Pick a width per column, using one data row to spot the date columns."""
    widths = []
    for index, column in enumerate(columns):
        value = sample[index] if sample and index < len(sample) else None
        if isinstance(value, datetime):
            widths.append(_DATETIME_WIDTH)
        elif isinstance(value, date):
            widths.append(_DATE_WIDTH)
        else:
            widths.append(min(max(len(column) + 2, _MIN_WIDTH), _MAX_WIDTH))
    return widths


async def _chain_first(
    first: Optional[Sequence[Any]], rest: AsyncIterator[Sequence[Any]]
) -> AsyncIterator[Sequence[Any]]:
    """Put a peeked-at row back in front of the iterator it came from."""
    if first is not None:
        yield first
    async for row in rest:
        yield row


def _write_row(worksheet, row_index, row, date_format, datetime_format) -> None:
    """Write one row, each cell with the writer its type calls for.

    Every branch is explicit because ``worksheet.write()`` guesses from the
    value, and one of its guesses is dangerous: a string starting with ``=``
    is written as a FORMULA. Entity names reach these cells verbatim and are
    not character-restricted (an API key or volume may be named
    ``=HYPERLINK("http://…")``), so a tenant could plant a formula that runs
    when an operator opens the workbook. ``write_string`` stores the name as
    text — no value changes, the injection vector simply doesn't exist. The
    same convention as the enterprise billing export.

    ``bool`` is checked before ``int`` because it subclasses it, and
    ``datetime`` before ``date`` for the same reason — otherwise every instant
    would lose its time of day.
    """
    for index, value in enumerate(row):
        if value is None:
            continue
        if isinstance(value, datetime):
            worksheet.write_datetime(row_index, index, value, datetime_format)
        elif isinstance(value, date):
            worksheet.write_datetime(row_index, index, value, date_format)
        elif isinstance(value, bool):
            worksheet.write_boolean(row_index, index, value)
        elif isinstance(value, (int, float)):
            worksheet.write_number(row_index, index, value)
        else:
            worksheet.write_string(row_index, index, str(value))


async def build_xlsx(
    sheets: Iterable[Tuple[str, Sequence[str], AsyncIterator[Sequence[Any]]]],
) -> bytes:
    """Assemble a workbook, one worksheet per sheet.

    Returns the finished bytes rather than an iterator: the xlsx container has
    to be complete before any of it is valid. ``constant_memory`` keeps row
    data off the heap while rows are written.
    """
    # Imported inside the function so a server that never exports an xlsx
    # doesn't pay for the module at startup.
    import xlsxwriter

    buffer = io.BytesIO()
    # ``constant_memory`` streams rows out of RAM as they're written, but only
    # to xlsxwriter's temporary files — ``in_memory=True`` puts those back on
    # the heap and undoes it. At 100k rows that is the difference between
    # ~5 MB and ~240 MB peak, so the temp files stay on disk.
    workbook = xlsxwriter.Workbook(
        buffer,
        {
            "constant_memory": True,
            "in_memory": False,
            # The spreadsheet format has no concept of an offset, and
            # xlsxwriter raises rather than guess. Usage exports now coerce
            # their dates to calendar days before they get here, so nothing
            # should arrive tz-aware — this is a backstop that keeps a future
            # aware column from failing the whole export.
            "remove_timezone": True,
        },
    )
    # A date written without a number format lands in Excel as its raw serial
    # ("46246"), because that is genuinely what the file stores — the display
    # format is a property of the CELL, not of the value. So every date cell
    # has to carry one. Dates get day precision (the token rollup is a calendar
    # day; a time-of-day would be invented) and instants keep seconds.
    date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
    datetime_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
    taken: set = set()
    try:
        for name, columns, rows in sheets:
            worksheet = workbook.add_worksheet(_unique_sheet_name(name, taken))
            # Column widths have to be set before ANY cell is written — in
            # constant_memory mode a row is flushed as soon as the next one
            # starts, so the sheet's column metadata is already gone by then.
            # Hence the peek: the first data row is what tells us which
            # columns hold dates.
            try:
                first_row = await rows.__anext__()
            except StopAsyncIteration:
                first_row = None
            for index, width in enumerate(_column_widths(columns, first_row)):
                worksheet.set_column(index, index, width)

            for index, column in enumerate(columns):
                worksheet.write_string(0, index, column)
            row_index = 1
            async with _closing(rows):
                async for row in _chain_first(first_row, rows):
                    _write_row(worksheet, row_index, row, date_format, datetime_format)
                    row_index += 1
    finally:
        # ``close()`` is where the workbook is actually built: it reads back
        # every temporary file, deflates it and assembles the zip, all
        # synchronously. On a 100k-row export that is seconds during which the
        # event loop cannot run anything else — every other request on this
        # process would wait — so it goes to a worker thread. Nothing else
        # touches the workbook at this point, so the handoff is safe.
        await asyncio.to_thread(workbook.close)
    return buffer.getvalue()


_XLSX_FORBIDDEN_SHEET_CHARS = set(r"[]:*?/\\")


def _safe_sheet_name(name: str) -> str:
    """Coerce a name into what Excel accepts for a worksheet title.

    Excel rejects a handful of characters outright and truncates past 31
    chars; an invalid name aborts the whole workbook, so sanitize rather than
    trust the caller.
    """
    cleaned = "".join(
        "_" if ch in _XLSX_FORBIDDEN_SHEET_CHARS else ch for ch in (name or "")
    ).strip("'")
    cleaned = cleaned[:31]
    return cleaned or "Sheet"


def _unique_sheet_name(name: str, taken: set) -> str:
    """A sanitized name no other worksheet in this workbook is using.

    Sheet names are the caller's LOCALIZED display strings, so two tables can
    legitimately arrive with the same one (and truncation to 31 chars can
    collide names that started out different). xlsxwriter raises on a
    duplicate — case-insensitively — which would turn a request the server
    accepted into a 500 halfway through building the workbook. The data is
    fine; only the label needs to differ, so disambiguate with a suffix.
    """
    base = _safe_sheet_name(name)
    candidate = base
    suffix = 2
    while candidate.casefold() in taken:
        # Keep the suffix inside Excel's 31-char ceiling.
        tail = f"_{suffix}"
        candidate = base[: 31 - len(tail)] + tail
        suffix += 1
    taken.add(candidate.casefold())
    return candidate
