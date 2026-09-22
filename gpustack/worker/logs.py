import asyncio
import bisect
import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import os
import logging
from pathlib import Path
from typing import (
    Annotated,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
from fastapi import Depends, Query

from gpustack.worker.log_sources import (
    MarkerCounts,
    is_marker_log_path,
    marker_log_path,
    next_segment_log_path,
    read_marker_counts,
    split_segment_log_path,
    surviving_segment_log_path,
)


logger = logging.getLogger(__name__)


@dataclass
class LogOptions:
    tail: int = -1  # -1 by default means read all logs
    follow: bool = False
    stop_event: Optional[asyncio.Event] = None
    previous: bool = False
    offset: Optional[int] = None  # first line to return, 0-based
    limit: int = 1000

    def url_encode(self):
        params = f"tail={self.tail}&follow={self.follow}"
        if self.previous:
            params += "&previous=true"
        if self.offset is not None:
            params += f"&offset={self.offset}&limit={self.limit}"
        return params


default_tail = Query(
    default=-1, description="Number of lines to read from the end of the log"
)
default_follow = Query(default=False, description="Whether to follow the log output")
default_previous = Query(
    default=False, description="Whether to fetch logs from the previous restart"
)
default_offset = Query(
    default=None,
    ge=0,
    description=(
        "First line to return, counted from the start of the log. "
        "Cannot be combined with tail or follow."
    ),
)
default_limit = Query(
    default=1000, gt=0, description="How many lines to return from offset"
)


def get_log_options(
    tail: int = default_tail,
    follow: bool = default_follow,
    previous: bool = default_previous,
    offset: Optional[int] = default_offset,
    limit: int = default_limit,
) -> LogOptions:
    return LogOptions(
        tail=tail, follow=follow, previous=previous, offset=offset, limit=limit
    )


LogOptionsDep = Annotated[LogOptions, Depends(get_log_options)]


# A checkpoint every this many bytes. Counting newlines is C-speed, but every
# checkpoint costs a Python step, so the index of a 64 MiB log is a thousand
# entries rather than one per line. A window read seeks to the checkpoint
# before it and scans forward over at most this much of the file.
_INDEX_STRIDE_BYTES = 1 << 16
_INDEX_READ_BLOCK = 1 << 20
# Enough for every log of every restart a worker keeps, several times over.
_INDEX_CAPACITY = 512
# Enough of the indexed bytes to recognise them again on the next read.
_INDEX_FINGERPRINT_BYTES = 64
# How much of a line range one read hands over at a time.
_SLICE_CHUNK_BYTES = 1 << 18


@dataclass
class LogFileIndex:
    """Where the lines of one log file start, sampled every stride bytes.

    Attributes:
        size: Bytes the index covers; the file has grown if it is now larger.
        newlines: Complete lines, i.e. how many newline bytes were counted.
        last_newline_end: Offset just past the final newline, so a file ending
            mid-line can be told from one that does not.
        checkpoints: (byte offset of a line start, that line's number), the
            first entry always (0, 0).
        file_id: (st_dev, st_ino) of the file these counts describe.
        tail: The last bytes of what was counted, to recognise it again.
        generation: Which count of the file this is: a new one each time it
            is counted from the start, kept as it is extended.
    """

    size: int = 0
    newlines: int = 0
    last_newline_end: int = 0
    checkpoints: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
    file_id: Tuple[int, int] = (0, 0)
    tail: bytes = b''
    generation: int = 0

    @property
    def line_count(self) -> int:
        """Lines in the file, counting a trailing fragment as one."""
        return self.newlines + (1 if self.size > self.last_newline_end else 0)


_index_cache: "OrderedDict[str, LogFileIndex]" = OrderedDict()
# Guards the table only, never a scan: an uncapped log takes seconds to scan,
# and every other log would wait it out. An index is extended on a copy and
# never changed once published, so whoever holds one reads it unlocked.
_index_lock = threading.Lock()
_index_generations = itertools.count(1)


def _index_describes(path: Path, index: LogFileIndex) -> bool:
    """Whether the bytes the index counted are still the ones on disk.

    A log truncated and written again behind the same name is a different log:
    extending the old counts over it reports lines that never existed.
    """
    if index.size == 0:
        return True
    with open(path, 'rb') as f:
        f.seek(index.size - len(index.tail))
        return f.read(len(index.tail)) == index.tail


def _extend_index(path: Path, index: LogFileIndex):
    with open(path, 'rb') as f:
        f.seek(index.size)
        position = index.size
        next_checkpoint = index.size + _INDEX_STRIDE_BYTES
        while True:
            block = f.read(_INDEX_READ_BLOCK)
            if not block:
                break
            counted_to = 0
            while next_checkpoint < position + len(block):
                found = block.find(b'\n', max(0, next_checkpoint - position))
                if found < 0:
                    break
                line_start = found + 1
                index.newlines += block.count(b'\n', counted_to, line_start)
                counted_to = line_start
                index.checkpoints.append((position + line_start, index.newlines))
                next_checkpoint = position + line_start + _INDEX_STRIDE_BYTES
            index.newlines += block.count(b'\n', counted_to)
            last = block.rfind(b'\n')
            if last >= 0:
                index.last_newline_end = position + last + 1
            position += len(block)
        index.size = position
        # Only what was counted: the log may have grown since the last read,
        # and a fingerprint reaching past the counted bytes matches nothing.
        start = max(0, position - _INDEX_FINGERPRINT_BYTES)
        f.seek(start)
        index.tail = f.read(position - start)


def index_log_file(path: Path) -> LogFileIndex:
    """The line index of one log file, built once and extended as it grows.

    Args:
        path: Path to the log file.

    Returns:
        The index, empty when the file cannot be read.
    """
    key = str(path)
    try:
        status = path.stat()
    except OSError:
        return LogFileIndex()
    size = status.st_size
    file_id = (status.st_dev, status.st_ino)
    with _index_lock:
        cached = _index_cache.get(key)
    try:
        if (
            cached is None
            or cached.file_id != file_id
            or not _index_describes(path, cached)
        ):
            index = LogFileIndex(file_id=file_id, generation=next(_index_generations))
        elif size <= cached.size:
            index = cached
        else:
            index = replace(cached, checkpoints=list(cached.checkpoints))
        if size > index.size:
            _extend_index(path, index)
    except OSError as e:
        logger.warning(f"Failed to index {path}: {e}")
        return LogFileIndex()
    with _index_lock:
        current = _index_cache.get(key)
        # A count from the start sees the file as it is now, so it beats one
        # extended from an earlier look: that one may describe a log since
        # rewritten shorter. Two extensions of one count are both valid, and
        # the longer wins.
        if current is not None and current.file_id == file_id:
            if (current.generation, current.size) > (index.generation, index.size):
                index = current
        _index_cache[key] = index
        _index_cache.move_to_end(key)
        while len(_index_cache) > _INDEX_CAPACITY:
            _index_cache.popitem(last=False)
    return index


@dataclass(frozen=True)
class LineWindow:
    """A slice of a concatenated log stream, resolved against its files.

    Attributes:
        offset: The first line number the window holds.
        line_count: How many line numbers it spans; fewer than asked for at the
            end. A marker spans the lines it stands in for but reads as one.
        total_lines: Line numbers in the whole stream, as the index sees it now.
        reads: (path, first line within that file, how many lines, the file's
            size when planned) in order. No read goes past that size, so what
            the window returns is the stream as it was planned. The marker's
            size is None: it is read whole, as it stands when its read opens it.
        includes_marker: Whether the size cap has already dropped part of it.
    """

    offset: int
    line_count: int
    total_lines: int
    reads: Tuple[Tuple[Path, int, int, Optional[int]], ...]
    includes_marker: bool = False

    @property
    def byte_count(self) -> Optional[int]:
        """The exact length of the window, when it covers the whole stream.

        Unknown once the cap has dropped part of it: the marker may be
        rewritten, longer, before its read opens it.
        """
        if (
            self.includes_marker
            or self.offset != 0
            or self.line_count != self.total_lines
        ):
            return None
        return sum(end for _path, _first, _count, end in self.reads)


def plan_line_window(paths: Iterable[Path], offset: int, limit: int) -> LineWindow:
    """Work out which files hold a line range, and where in them.

    Blocking: indexes each file, so call it off the event loop.

    Args:
        paths: The stream's files, in reading order.
        offset: First line to return, counted from the start of the stream.
        limit: How many lines to return.

    Returns:
        The resolved window. An offset past the end resolves to no reads and
        the real total, rather than to an error.
    """
    return _plan_spans(_line_spans(paths), offset, limit)


def plan_tail_window(paths: Iterable[Path], tail: int) -> LineWindow:
    """The window holding a stream's last `tail` lines, across all its files.

    Blocking: indexes each file, so call it off the event loop.

    Args:
        paths: The stream's files, in reading order.
        tail: How many lines from the end.

    Returns:
        The resolved window, planned on one look at the files, so its reads end
        exactly where a follower has to pick up.
    """
    spans = _line_spans(paths)
    total = sum(span.lines for span in spans)
    return _plan_spans(spans, max(0, total - tail), tail)


class _Span(NamedTuple):
    path: Path
    # None for the marker, whose line count is the lines it holds.
    index: Optional[LogFileIndex]
    lines: int


# A plan that keeps losing shards to the cap settles for what it has.
_PLAN_ATTEMPTS = 3


def _line_spans(paths: Iterable[Path]) -> List[_Span]:
    paths = list(paths)
    for _ in range(_PLAN_ATTEMPTS):
        spans, settled = _try_line_spans(paths)
        if settled:
            break
    return spans


def _try_line_spans(paths: List[Path]) -> Tuple[List[_Span], bool]:
    """Line spans of the files, and whether the cap left them alone meanwhile.

    A marker holds the numbers of the lines it replaced, so dropping a shard
    renumbers nothing after it. It counts a shard in before the shard goes, so
    a shard it covers is skipped whether or not it is still on disk, and one
    it does not cover going missing means the marker has moved on since it
    was read.
    """
    spans: List[_Span] = []
    markers: Dict[Path, Optional[MarkerCounts]] = {}
    placed = set()
    settled = True

    def place_marker(head: Path):
        if head in placed:
            return
        placed.add(head)
        marker = marker_log_path(head)
        counts = markers[head]
        lines = counts.omitted_lines if counts else index_log_file(marker).line_count
        spans.append(_Span(marker, None, lines))

    for path in paths:
        head, shard = split_segment_log_path(path)
        if head not in markers:
            markers[head] = read_marker_counts(marker_log_path(head))
        if is_marker_log_path(path):
            place_marker(head)
            continue
        counts = markers[head]
        if shard > 0 and counts is not None and shard <= counts.dropped_shards:
            # Listed before the marker covered it, or before the marker was.
            place_marker(head)
            continue
        index = index_log_file(path)
        if shard > 0 and index.size == 0 and not path.exists():
            settled = False
        spans.append(_Span(path, index, index.line_count))
    return spans, settled


def _plan_spans(spans: List[_Span], offset: int, limit: int) -> LineWindow:
    reads = []
    total = 0
    wanted = limit
    includes_marker = False
    for path, index, lines in spans:
        includes_marker = includes_marker or index is None
        start = offset - total
        if wanted > 0 and start < lines:
            first = max(0, start)
            take = min(wanted, lines - first)
            if take > 0:
                if index is None:
                    reads.append((path, 0, 1, None))
                else:
                    reads.append((path, first, take, index.size))
                wanted -= take
        total += lines
    return LineWindow(
        offset=offset,
        line_count=limit - wanted,
        total_lines=total,
        reads=tuple(reads),
        includes_marker=includes_marker,
    )


def checkpoint_at_or_before(index: LogFileIndex, line: int) -> Tuple[int, int]:
    """The nearest place a read for `line` may start from.

    This is the whole point of the index: without it a page deep in the log
    costs a scan of everything before it.

    Args:
        index: The file's index.
        line: The line the read is after.

    Returns:
        (byte offset of a line start, that line's number), never past `line`.
    """
    # Checkpoints rise in both fields, so bisecting on the line number finds
    # the last one at or before it.
    position = bisect.bisect_right(index.checkpoints, line, key=lambda c: c[1])
    return index.checkpoints[max(0, position - 1)]


def read_line_slice(
    path: Path,
    first: int,
    count: int,
    end: Optional[int] = None,
    file: Optional[BinaryIO] = None,
) -> Iterator[bytes]:
    """Read `count` lines from one file, starting at its line `first`.

    Blocking: seeks to the nearest checkpoint and scans forward from there.
    A download asks for every line of a log in one range, so the lines leave
    in pieces rather than as one string the size of the file.

    The bytes are handed over as they are on disk. Decoding them would turn
    every byte no encoding claims into a three-byte replacement character, and
    a download's Content-Length is measured on the file, not on that.

    Args:
        path: Path to the log file.
        first: First line to read within this file, 0-based.
        count: How many lines to read.
        end: Read nothing at or past this byte, so a line completed after the
            range was planned still ends where it did then.
        file: The file already open for reading, closed once read; opened
            from `path` when not given.

    Yields:
        The lines in order, newlines included, in chunks of about
        `_SLICE_CHUNK_BYTES`. Nothing at all when the file cannot be read.
    """
    index = index_log_file(path)
    at_offset, at_line = checkpoint_at_or_before(index, first)
    try:
        with file or open(path, 'rb') as f:
            f.seek(at_offset)
            position = at_offset

            def next_line() -> bytes:
                nonlocal position
                line = f.readline(-1 if end is None else end - position)
                position += len(line)
                return line

            for _ in range(first - at_line):
                if not next_line():
                    return
            taken: List[bytes] = []
            held = 0
            for _ in range(count):
                line = next_line()
                if not line:
                    break
                taken.append(line)
                held += len(line)
                if held >= _SLICE_CHUNK_BYTES:
                    yield b"".join(taken)
                    taken, held = [], 0
            if taken:
                yield b"".join(taken)
    except OSError as e:
        logger.warning(f"Failed to read a line range of {path}: {e}")


def _open_reads(
    reads: Iterable[Tuple[Path, int, int, Optional[int]]],
) -> List[Optional[BinaryIO]]:
    files = []
    for path, *_ in reads:
        try:
            files.append(open(path, 'rb'))
        except OSError as e:
            logger.warning(f"Failed to read a line range of {path}: {e}")
            files.append(None)
    return files


async def line_window_generator(window: LineWindow):
    """Yield a resolved window's lines, a chunk of one file's slice at a time."""
    # All opened before the first line leaves: an open file outlives its name,
    # so a slow download still gets the shards the cap removes meanwhile.
    files = await asyncio.to_thread(_open_reads, window.reads)
    reading = None
    try:
        for read, file in zip(window.reads, files):
            if file is None:
                continue
            chunks = read_line_slice(*read, file=file)
            try:
                while True:
                    chunk = await asyncio.to_thread(next, chunks, None)
                    if chunk is None:
                        break
                    yield chunk
            finally:
                # Cancelling the await abandons the read, it does not stop it,
                # so the reader may still be inside the generator here -- and
                # closing one mid-read raises, which would replace the
                # cancellation with an error. The abandoned read ends on its own
                # a chunk later and closes its file then: closing it here would
                # wait on that read, on the event loop.
                try:
                    chunks.close()
                except ValueError:
                    reading = file
    finally:
        for file in files:
            if file is not None and file is not reading:
                file.close()


async def tail_log_generator(paths: List[Path], options: LogOptions):
    """A stream's last `options.tail` lines, then what it gains if following.

    The tail is counted across the stream's files: right after a roll the
    newest one holds a line or two, and the rest of the tail is behind it.

    Args:
        paths: The stream's files, in reading order.
        options: tail, follow and the stop event.
    """
    window = await asyncio.to_thread(plan_tail_window, paths, options.tail)
    async for chunk in line_window_generator(window):
        yield chunk
    if not options.follow or not paths:
        return
    last = paths[-1]
    # Picked up exactly where the tail's read of it stopped.
    start_at = next((end for path, _f, _c, end in window.reads if path == last), 0)
    async for line in log_generator(
        str(last), replace(options, tail=-1), start_at=start_at
    ):
        yield line


async def log_generator(path: str, options: LogOptions, start_at: Optional[int] = None):
    """Read one log file, and when following, the shards it rolls into.

    Args:
        path: The file to start from.
        options: tail and follow; offset and limit do not apply here.
        start_at: A byte offset to read on from, in place of the whole file or
            its tail.
    """
    logger.debug(f"Reading logs from {path} with options {options}")

    while True:
        try:
            # By default, universal newline mode is used, which means that all of
            # \n, \r, or \r\n are recognized as end-of-line characters.
            # We use os.linesep to ensure that \r is reserved. It's useful for showing progress bars.
            async with aiofiles.open(
                path, "r", encoding="utf-8", errors="ignore", newline=os.linesep
            ) as file:
                if start_at is not None:
                    await file.seek(start_at)
                    async for line in read_all_lines(file):
                        yield line
                elif options.tail > 0:
                    # Move to the end of the file and read the last 'tail' lines
                    await file.seek(0, os.SEEK_END)
                    file_size = await file.tell()
                    buffer = []
                    BLOCK_SIZE = 2**16  # 64KB
                    while file_size > 0 and len(buffer) <= options.tail:
                        await file.seek(max(0, file_size - BLOCK_SIZE), os.SEEK_SET)
                        buffer = await file.readlines()
                        file_size -= BLOCK_SIZE
                    for line in buffer[-options.tail :]:
                        yield line
                else:
                    async for line in read_all_lines(file):
                        yield line

                if options.follow:
                    async for line in follow_file(file, options.stop_event, path):
                        yield line
        except FileNotFoundError:
            # The cap dropped this shard while it was being read. The log goes
            # on in the successor, so carry on there rather than stop here.
            logger.debug(f"Serve log {path} went away while it was being read")
        except Exception as e:
            logger.error(f"Failed to read logs from {path}. {e}")
            return

        # A capped log continues in the next shard, so the end of this file
        # is not the end of the log -- and a reader that fell a whole budget
        # behind continues in whichever shard the cap has not dropped yet.
        successor = (
            await asyncio.to_thread(surviving_segment_log_path, Path(path))
            if options.follow
            else None
        )
        if successor is None:
            return
        path = str(successor)
        options = replace(options, tail=-1)
        start_at = None


async def read_all_lines(file: AsyncTextIOWrapper):
    """Read all lines from the file."""
    while True:
        line = await file.readline()
        if not line:
            break
        yield line


async def follow_file(
    file: AsyncTextIOWrapper,
    stop_event: Optional[asyncio.Event] = None,
    path: Optional[str] = None,
):
    """Follow the file and yield new lines as they are written.

    Returns once the file turns out to be a finished shard of a capped log,
    leaving the caller to continue in the next one: its successor exists, or
    the cap already dropped it. The writer flushes a shard before creating its
    successor, so one more read takes the rest of it.
    """
    while True:
        if stop_event and stop_event.is_set():
            return
        line = await file.readline()
        if line:
            yield line
            continue
        if path is not None and (
            next_segment_log_path(Path(path)) is not None
            or os.fstat(file.fileno()).st_nlink == 0
        ):
            async for remaining in read_all_lines(file):
                yield remaining
            return
        await asyncio.sleep(0.1)  # wait before retrying
