"""Bounded container log archives shared by workload managers."""

from datetime import datetime, timezone
import io
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Optional, Tuple

from gpustack_runtime.deployer import (
    get_workload,
    logs_workload,
    WorkloadStatusStateEnum,
)

from gpustack import envs
from gpustack.worker.log_sources import CappedLogWriter, tail_shard_log_paths

logger = logging.getLogger(__name__)

# One health-check cycle (+2s margin) to let a container return after a stream
# EOF; beyond that gpustack marks it ERROR and takes over recovery.
LOG_RECONNECT_GRACE_SECONDS = envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL + 2

# Read behind the end of a log when trimming a fragment a killed worker left.
_LOG_TAIL_CHUNK_SIZE = 8192

# Written into the archive where the runtime could not replay from the cursor.
_LOG_GAP_MARKER = (
    "... reconnected here; the container runtime could no longer replay from "
    "{timestamp}, so some lines may be missing ...\n"
)

# The runtime prefixes every streamed line with an RFC3339Nano timestamp and a
# space. Go trims the fraction's trailing zeros, so the prefix has no fixed
# width and can only be split on the first space.
_LOG_TIMESTAMP_RE = re.compile(
    r'^(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
    r'(?:\.\d+)?(?P<offset>Z|[+-]\d{2}:\d{2})$'
)


def _split_log_timestamp(text: str) -> Tuple[Optional[int], str]:
    """Take the runtime's timestamp prefix off one streamed chunk.

    Args:
        text: One chunk as the runtime yielded it.

    Returns:
        (epoch second, the chunk without its prefix). The epoch is None when
        the chunk carries no prefix to read.
    """
    prefix, separator, rest = text.partition(' ')
    if not separator:
        return None, text
    match = _LOG_TIMESTAMP_RE.match(prefix)
    if not match:
        return None, text
    moment = datetime.strptime(match.group("stamp"), "%Y-%m-%dT%H:%M:%S")
    epoch = int(moment.replace(tzinfo=timezone.utc).timestamp())
    offset = match.group("offset")
    if offset != 'Z':
        minutes = int(offset[1:3]) * 60 + int(offset[4:6])
        epoch += minutes * 60 if offset[0] == '-' else -minutes * 60
    return epoch, rest


def _streamed_lines(log_stream, stop_event: threading.Event):
    """Strip the runtime's timestamp prefix off each chunk as it arrives.

    One chunk is usually one line, but a line past 16 KiB arrives in several
    and a progress bar redrawing in place may never end one at all. Text is
    passed straight on so nothing waits on a newline that may be minutes away;
    the flag is what tells a caller a line just finished.

    Args:
        log_stream: The runtime's chunk iterator.
        stop_event: Event to signal the thread to stop.

    Yields:
        (epoch second, text, whether the text ends a line) triples.
    """
    for chunk in log_stream:
        if stop_event.is_set():
            return
        text = (
            chunk.decode('utf-8', errors='replace')
            if isinstance(chunk, bytes)
            else str(chunk)
        )
        epoch, content = _split_log_timestamp(text)
        yield epoch, content, text.endswith('\n')


def _log_archive_is_empty(log_path: str) -> bool:
    """Whether an archive holds nothing yet -- the only case that may open 'w'.

    Args:
        log_path: Path to the archive's head.

    Returns:
        True when the file is missing or zero length.
    """
    try:
        return Path(log_path).stat().st_size == 0
    except OSError:
        return True


# Fixed-width, so an update overwrites the previous record whole.
_CURSOR_RECORD = "{epoch:020d} {count:012d}\n"
_CURSOR_RE = re.compile(r'^(?P<epoch>\d{20}) (?P<count>\d{12})\n$')

# How often the cursor reaches disk. A worker killed between writes resumes at
# most this far back, which costs duplicated lines -- never lost ones.
_CURSOR_STORE_INTERVAL = 1.0


class _LogCursor:
    """Where the copier left off in a container's log stream.

    An epoch second plus how many of that second's lines the archive already
    holds. A runtime can only be asked to resume at a whole second, so the
    count is what keeps the rest of that second from arriving twice. Counting
    is enough because a stream and its replay are the same append-only file
    read in the same order.

    Stored beside the archive so a worker restart resumes the way a reconnect
    does, and written through a temporary file so a kill cannot leave half a
    record behind.
    """

    def __init__(self, log_path: str):
        self.path = Path(f"{log_path}.cursor")
        self.epoch: Optional[int] = None
        self.count = 0
        # Whether the position was invented rather than recorded: no line the
        # archive holds sits behind it, so nothing a replay skips proves a seam.
        self.synthesized = False
        self._stored_at = 0.0

    def load(self):
        """Read the position a previous worker process left behind."""
        try:
            match = _CURSOR_RE.match(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if match:
            self.epoch = int(match.group("epoch"))
            self.count = int(match.group("count"))
            # Only an invented position is stored with nothing counted behind it.
            self.synthesized = self.count == 0

    def note(self, epoch: Optional[int]):
        """Record that one more line has reached the archive."""
        if epoch is None:
            return
        if epoch != self.epoch:
            self.epoch, self.count = epoch, 0
        self.count += 1
        self.synthesized = False
        if time.monotonic() - self._stored_at >= _CURSOR_STORE_INTERVAL:
            self.store()

    def clear(self):
        """Forget the position, and the file holding it."""
        self.epoch, self.count = None, 0
        self.synthesized = False
        try:
            self.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to remove the log cursor {self.path}: {e}")

    def store(self):
        """Put the current position on disk."""
        if self.epoch is None:
            return
        self._stored_at = time.monotonic()
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        try:
            temporary.write_text(
                _CURSOR_RECORD.format(epoch=self.epoch, count=self.count),
                encoding='utf-8',
            )
            os.replace(temporary, self.path)
        except OSError as e:
            logger.warning(f"Failed to record the log cursor {self.path}: {e}")


class _ReplaySkipper:
    """Tells the lines a reconnect replays from the ones the archive lacks.

    Everything before the cursor's second is already there, and so are the
    first lines of that second -- the runtime replays them in the order it
    streamed them, so counting them off is enough to find where to resume.
    """

    def __init__(self, cursor: _LogCursor):
        self._epoch = cursor.epoch
        self._remaining = cursor.count

    def skip(self, epoch: Optional[int]) -> bool:
        if self._epoch is None:
            return False
        # A line with no timestamp says nothing about where the replay is.
        if epoch is None:
            return False
        if epoch > self._epoch:
            self._replaying_done()
            return False
        if epoch < self._epoch:
            return True
        if self._remaining == 0:
            self._replaying_done()
            return False
        self._remaining -= 1
        return True

    def _replaying_done(self):
        # The replay is a prefix: once one line is new, so is every later one,
        # whatever the clock stamps on it after stepping back.
        self._epoch = None


def _drop_partial_last_line(log_path: str):
    """Truncate an archive's trailing line when it carries no newline.

    A worker killed mid-write leaves a fragment; the runtime replays that line
    whole, so appending after the fragment would join the two. A runtime sends
    a long line in 16 KiB pieces, so the fragment can be any length: a segment
    holding nothing but a fragment is emptied, and the fragment's start looked
    for in the one before, as a line that never ends rolls over mid-line.

    Args:
        log_path: Path to the archive's head.
    """
    head = Path(log_path)
    for segment in reversed([head, *tail_shard_log_paths(head)]):
        if _ends_at_a_line_end(segment):
            return


def _ends_at_a_line_end(log_path: Path) -> bool:
    """Truncate one segment back to its last newline.

    Returns:
        Whether it now ends at one; False when no newline was left to keep.
    """
    try:
        with open(log_path, 'rb') as f:
            end = f.seek(0, os.SEEK_END)
            if end == 0:
                return False
            f.seek(end - 1)
            if f.read(1) == b'\n':
                return True
            keep = 0
            position = end
            while position > 0:
                start = max(0, position - _LOG_TAIL_CHUNK_SIZE)
                f.seek(start)
                last_newline = f.read(position - start).rfind(b'\n')
                if last_newline >= 0:
                    keep = start + last_newline + 1
                    break
                position = start
        os.truncate(log_path, keep)
        return keep > 0
    except FileNotFoundError:
        # A fresh archive: nothing written yet, so nothing to trim.
        return False
    except OSError as e:
        logger.warning(f"Failed to trim the partial last line of {log_path}: {e}")
        # Trimming the segments before this one would not help.
        return True


class ContainerLogPersister:
    """Persist runtime output with bounded files and a durable reconnect cursor.

    The workload manager owns the thread, stop event and archive lifetime.
    """

    def _persist_container_logs(
        self,
        workload_name: str,
        log_path: str,
        stop_event: threading.Event,
        token: Optional[str] = None,
        resume: bool = False,
    ):
        """Persist container logs to local file (runs in a separate thread).

        Reconnects on stream EOF while the workload is still alive, resuming
        from a timestamp cursor rather than from the archive's contents. A
        manual/runtime restart briefly looks terminated at EOF, so it waits a
        grace window for the container to return before giving up. Exits only
        if the container stays terminated for that whole window or the thread
        is asked to stop.

        Args:
            workload_name: Name of the container workload
            log_path: Path to save container logs
            stop_event: Event to signal thread to stop
            token: Operation token identifying a specific container in the workload.
                If None, logs from the default (index=0) container are fetched.
            resume: Adopt a log file a previous worker process left behind,
                appending to it instead of rewriting it from the runtime's replay.
        """
        retry_count = 0
        cursor = _LogCursor(log_path)
        if resume:
            cursor.load()

        while not stop_event.is_set():
            try:
                # A connection cut mid-line left the archive ending mid-line, and
                # the runtime replays that line whole, so the fragment goes
                # first -- before the cursor, as an archive of nothing but a
                # fragment is an empty one.
                _drop_partial_last_line(log_path)
                self._locate_cursor(cursor, log_path)
                log_stream = logs_workload(
                    name=workload_name,
                    token=token,
                    tail=-1,
                    timestamps=True,
                    since=cursor.epoch,
                    follow=True,
                )
                # Stopped while the runtime was connecting: the instance may be
                # gone, and copying now would recreate its purged directory.
                if stop_event.is_set():
                    break

                if hasattr(log_stream, '__iter__'):
                    self._copy_container_log_stream(
                        log_stream, log_path, cursor, stop_event
                    )
                    retry_count = 0

                # A restart briefly looks terminated at EOF; wait for the
                # container to return before giving up, so logs aren't dropped.
                if stop_event.is_set() or not self._wait_for_container_recovery(
                    workload_name, stop_event
                ):
                    break
                logger.debug(
                    f"Log stream for {workload_name} ended while workload still "
                    f"running; reconnecting"
                )
                stop_event.wait(timeout=1)

            except Exception as e:
                if stop_event.is_set():
                    break
                # TODO: Unify retry termination for missing workloads across
                # model and cache log persistence during workload unification.
                retry_count += 1
                logger.debug(
                    f"Container not ready for {workload_name}, retrying "
                    f"(attempt {retry_count}): {e}"
                )
                stop_event.wait(timeout=2)

        logger.debug(f"Log persistence thread for {workload_name} exiting")

    def _archive_container_logs(self, workload_name: str, log_path: str) -> None:
        """Drain existing runtime output before teardown, resuming the archive cursor."""
        cursor = _LogCursor(log_path)
        cursor.load()
        _drop_partial_last_line(log_path)
        self._locate_cursor(cursor, log_path)
        stream = logs_workload(
            name=workload_name,
            tail=-1,
            timestamps=True,
            since=cursor.epoch,
            follow=False,
        )
        try:
            chunks = (
                io.BytesIO(stream)
                if isinstance(stream, bytes)
                else io.StringIO(stream) if isinstance(stream, str) else stream
            )
            self._copy_container_log_stream(chunks, log_path, cursor, threading.Event())
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def _locate_cursor(self, cursor: _LogCursor, log_path: str):
        """Point the cursor at where the next connection should resume.

        Args:
            cursor: The copier's position, adjusted in place.
            log_path: Path to the archive's head.
        """
        if _log_archive_is_empty(log_path):
            # A cursor into an archive that is gone would relocate into a log
            # that no longer exists.
            cursor.clear()
        elif cursor.epoch is None:
            # An archive written before the cursor existed says nothing about
            # where it ends. Resuming here keeps it whole, at the cost of
            # whatever was logged meanwhile -- which the marker announces.
            cursor.epoch, cursor.count = int(time.time()), 0
            cursor.synthesized = True

    def _copy_container_log_stream(
        self,
        log_stream,
        log_path: str,
        cursor: _LogCursor,
        stop_event: threading.Event,
    ):
        """Write one connection's lines into the archive.

        The archive is opened for append unless it holds nothing at all, so no
        reconnect can ever shorten it. When the runtime cannot replay the
        cursor's own second, what is appended does not continue what is already
        there, and a marker says so.

        Args:
            log_stream: The runtime's chunk iterator.
            log_path: Path to save container logs.
            cursor: The copier's position, advanced as lines are written.
            stop_event: Event to signal thread to stop.
        """
        skipper = _ReplaySkipper(cursor)
        relocating = cursor.epoch is not None
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with CappedLogWriter(
            log_path, append=not _log_archive_is_empty(log_path)
        ) as archive:
            # A line arrives in one chunk or in several. Whether to skip it is
            # decided once, on the chunk that starts it, and holds until it ends.
            starting, replayed, line_epoch = True, False, None
            wrote, stamped = False, False
            for epoch, text, ends_line in _streamed_lines(log_stream, stop_event):
                stamped = stamped or epoch is not None
                if starting:
                    line_epoch = epoch
                    replayed = skipper.skip(epoch)
                    if relocating:
                        relocating = False
                        # An invented position skips lines the archive never
                        # held, so the skip says nothing about the seam.
                        if not replayed or cursor.synthesized:
                            moment = datetime.fromtimestamp(
                                cursor.epoch, tz=timezone.utc
                            )
                            archive.write(
                                _LOG_GAP_MARKER.format(timestamp=moment.isoformat())
                            )
                            logger.warning(
                                f"Container log {log_path} could not resume at "
                                f"{moment.isoformat()}; some lines may be missing"
                            )
                starting = ends_line
                if replayed:
                    continue
                archive.write(text)
                archive.flush()
                wrote = True
                # A line the stream cut short is on disk but not behind the
                # cursor: the next connection trims it and takes it whole.
                if ends_line:
                    cursor.note(line_epoch)
        if wrote and not stamped:
            # Nothing carried a timestamp, so the cursor did not move, and the
            # next connection would replay from where this one began.
            cursor.epoch, cursor.count = int(time.time()), 0
            cursor.synthesized = True
            logger.warning(
                f"Container log {log_path} carries no timestamps; the next "
                f"connection resumes from now"
            )
        cursor.store()

    def _container_still_running(self, workload_name: str) -> bool:
        """Whether the workload is still alive (a dead stream should reconnect
        rather than exit)."""
        try:
            workload = get_workload(workload_name)
        except Exception:
            return True  # transient query failure: reconnect, don't drop logs
        return bool(workload) and workload.state in (
            WorkloadStatusStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING,
            WorkloadStatusStateEnum.RUNNING,
        )

    def _wait_for_container_recovery(
        self,
        workload_name: str,
        stop_event: threading.Event,
        grace_seconds: float = LOG_RECONNECT_GRACE_SECONDS,
        poll_interval: float = 1.0,
    ) -> bool:
        """Poll until the workload is alive again (True -> reconnect) or the
        grace window elapses / stop_event fires (False -> give up). A restart
        momentarily looks terminated at EOF, which a single check can't tell
        apart from a real termination.
        """
        attempts = max(1, int(grace_seconds / poll_interval))
        for _ in range(attempts):
            if stop_event.is_set():
                return False
            if self._container_still_running(workload_name):
                return True
            stop_event.wait(timeout=poll_interval)
        return False
