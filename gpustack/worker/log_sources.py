"""
Log source strategies for unified log streaming.

This module provides a strategy pattern approach
for handling different log sources (download logs, main logs, container logs),
plus the serve log layout the read and write sides share and the writer that
keeps a serve log inside its size budget.
"""

import asyncio
import codecs
import contextlib
import io
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from gpustack import envs
from gpustack.utils import file

logger = logging.getLogger(__name__)


def legacy_main_log_path(log_dir: Path, model_instance_id: int) -> Path:
    """Path of the main serve log written before v2.2.0, named {id}.log.

    It carries no restart count and counts as restart 0.
    """
    return log_dir / f"{model_instance_id}.log"


# The path the name lands in has a length limit even though the name does not.
_MAX_INSTANCE_NAME_LENGTH = 96


def sanitize_instance_name(model_instance_name: str) -> str:
    """Reduce an instance name to what a directory name can carry.

    Everything outside [A-Za-z0-9_-] becomes a dash, the dot included, so that
    {name}.{id} stays two segments even for a model name holding a dot.

    Args:
        model_instance_name: The instance name as stored.

    Returns:
        The sanitized name, empty when nothing usable survives.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '-', model_instance_name)
    return sanitized[:_MAX_INSTANCE_NAME_LENGTH]


def instance_log_dir(
    log_dir: Path, model_instance_name: str, model_instance_id: int
) -> Path:
    """Directory holding every serve log of one model instance.

    Identity lives in the directory rather than in each file name, so the names
    inside stay short and `ls` alone says which deployment a log belongs to.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        model_instance_id: Model instance ID.

    Returns:
        Path named {name}.{id}, or {id} when the name sanitizes to nothing.
    """
    sanitized = sanitize_instance_name(model_instance_name)
    name = f"{sanitized}.{model_instance_id}" if sanitized else str(model_instance_id)
    return log_dir / name


def restart_log_dir(
    log_dir: Path,
    model_instance_name: str,
    model_instance_id: int,
    restart_count: int,
) -> Path:
    """Directory holding one restart's serve logs.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        model_instance_id: Model instance ID.
        restart_count: The restart these logs belong to.

    Returns:
        Path named {name}.{id}/{restart_count}
    """
    return instance_log_dir(log_dir, model_instance_name, model_instance_id) / str(
        restart_count
    )


MAIN_LOG_NAME = "main.log"
CONTAINER_LOG_NAME = "container.log"


def main_log_path(
    log_dir: Path,
    model_instance_name: str,
    model_instance_id: int,
    restart_count: int,
) -> Path:
    """Path of the main serve log for one restart of an instance.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        model_instance_id: Model instance ID.
        restart_count: The restart this log belongs to.

    Returns:
        Path named {name}.{id}/{restart_count}/main.log
    """
    return (
        restart_log_dir(log_dir, model_instance_name, model_instance_id, restart_count)
        / MAIN_LOG_NAME
    )


def container_log_path(
    log_dir: Path,
    model_instance_name: str,
    model_instance_id: int,
    restart_count: int,
) -> Path:
    """Path of the workload container's log for one restart of an instance.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        model_instance_id: Model instance ID.
        restart_count: The restart this log belongs to.

    Returns:
        Path named {name}.{id}/{restart_count}/container.log
    """
    return (
        restart_log_dir(log_dir, model_instance_name, model_instance_id, restart_count)
        / CONTAINER_LOG_NAME
    )


def sidecar_container_log_path(
    log_dir: Path,
    model_instance_name: str,
    model_instance_id: int,
    container_name: str,
    restart_count: int,
) -> Path:
    """Path of a sidecar container's log for one restart of an instance.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        model_instance_id: Model instance ID.
        container_name: Sidecar container name, e.g. "ray-head". The runtime
            names it, and it is sanitized like an instance name so nothing it
            carries can lead out of the restart's directory.
        restart_count: The restart this log belongs to.

    Returns:
        Path named {name}.{id}/{restart_count}/container.{container_name}.log
    """
    return (
        restart_log_dir(log_dir, model_instance_name, model_instance_id, restart_count)
        / f"container.{sanitize_instance_name(container_name)}.log"
    )


# A capped log's other pieces append one suffix to its head: the shard number,
# or "truncated" for the marker. After ".log", so container.log.1 (a shard) and
# container.1.log (a sidecar named "1") cannot collide.
_LOG_SUFFIX = ".log"
_MARKER_SUFFIX = "truncated"
_SHARD_RE = re.compile(r'^[0-9]+$')


def marker_log_path(head_log_path: Path) -> Path:
    """Path of the marker recording what the size cap dropped from a log.

    Args:
        head_log_path: The log's head.

    Returns:
        Path named like the head with ".truncated" appended.
    """
    return head_log_path.with_name(f"{head_log_path.name}.{_MARKER_SUFFIX}")


def tail_shard_log_path(head_log_path: Path, shard: int) -> Path:
    """Path of one tail shard of a size-capped log.

    Shards are numbered from 1 and a number is never reused, so the names alone
    give the order they were written in.

    Args:
        head_log_path: The log's head.
        shard: The shard's number.

    Returns:
        Path named like the head with ".{shard}" appended.
    """
    return head_log_path.with_name(f"{head_log_path.name}.{shard}")


class ServeLogKind(str, Enum):
    """Which stream a serve log file holds."""

    MAIN = "main"
    CONTAINER = "container"
    SIDECAR = "sidecar"


class ServeLogSegment(str, Enum):
    """Which part of a size-capped serve log a file holds.

    Head, marker, tail shards: read back in that order they reproduce the log.
    An uncapped log is a head and nothing else.
    """

    HEAD = "head"
    MARKER = "marker"
    TAIL = "tail"


@dataclass(frozen=True)
class ServeLogName:
    """What a serve log's location says about the file.

    Attributes:
        model_instance_id: The instance the file belongs to.
        restart_count: The restart the file belongs to.
        kind: Which stream it holds.
        container_name: The sidecar's container name; empty for the other kinds.
        instance_name: The instance name its directory carries; empty for a
            flat file, which has no directory to carry one.
        flat: True for a file left in the serve directory itself by a release
            before the per-instance directory.
        legacy: True for the pre-v2.2.0 {id}.log, which carries no restart.
        segment: Which part of a capped log it holds.
        shard: The tail shard's number; 0 for the other segments.
    """

    model_instance_id: int
    restart_count: int
    kind: ServeLogKind
    container_name: str = ""
    instance_name: str = ""
    flat: bool = False
    legacy: bool = False
    segment: ServeLogSegment = ServeLogSegment.HEAD
    shard: int = 0


# \d is avoided throughout: it also matches digits no int() round-trip survives.
_RESTART_DIR_RE = re.compile(r'^[0-9]+$')
# A sanitized instance name cannot hold a dot, so {name}.{id} is two segments.
_INSTANCE_DIR_RE = re.compile(r'^(?:(?P<name>[A-Za-z0-9_-]+)\.)?(?P<id>[0-9]+)$')
# The whole vocabulary inside a restart directory, so a sidecar's container
# name needs no escaping to stay unambiguous.
_MAIN_FILE_RE = re.compile(rf'^{re.escape(MAIN_LOG_NAME)}$')
_CONTAINER_FILE_RE = re.compile(rf'^{re.escape(CONTAINER_LOG_NAME)}$')
_SIDECAR_FILE_RE = re.compile(r'^container\.(?P<cname>.+)\.log$')

# The flat names earlier releases wrote, frozen: read only so they can be
# migrated. Anchored at both ends, or 1.container.2.3.log reads as instance 2's
# main log and 1.5.logXYZ reads as a log at all.
_FLAT_MAIN_RE = re.compile(r'^(?P<id>\d+)\.(?P<rc>\d+)\.log$')
_FLAT_CONTAINER_RE = re.compile(r'^(?P<id>\d+)\.container\.(?P<rc>\d+)\.log$')
_FLAT_SIDECAR_RE = re.compile(
    r'^(?P<id>\d+)\.container\.(?P<cname>[^.]+)\.(?P<rc>\d+)\.log$'
)
_FLAT_LEGACY_MAIN_RE = re.compile(r'^(?P<id>\d+)\.log$')


def parse_instance_dir_name(dirname: str) -> Optional[Tuple[str, int]]:
    """Read an instance log directory's name.

    Args:
        dirname: Bare directory name, without a parent.

    Returns:
        (instance name, instance id), the name empty when the directory carries
        only an id, or None if this is not an instance log directory.
    """
    match = _INSTANCE_DIR_RE.match(dirname)
    if not match:
        return None
    return match.group("name") or "", int(match.group("id"))


def _split_segment_suffix(filename: str) -> Optional[Tuple[str, ServeLogSegment, int]]:
    """Take a capped log's segment suffix off a file name.

    Returns:
        (head file name, segment, shard number), or None when what follows
        ".log" is neither a shard number nor the marker.
    """
    if filename.endswith(_LOG_SUFFIX):
        return filename, ServeLogSegment.HEAD, 0
    head_name, _, suffix = filename.rpartition('.')
    if not head_name.endswith(_LOG_SUFFIX):
        return None
    if suffix == _MARKER_SUFFIX:
        return head_name, ServeLogSegment.MARKER, 0
    if _SHARD_RE.match(suffix):
        return head_name, ServeLogSegment.TAIL, int(suffix)
    return None


def _parse_log_file_name(
    filename: str,
) -> Optional[Tuple[ServeLogKind, str, ServeLogSegment, int]]:
    split = _split_segment_suffix(filename)
    if split is None:
        return None
    head_name, segment, shard = split
    if _MAIN_FILE_RE.match(head_name):
        return ServeLogKind.MAIN, "", segment, shard
    if _CONTAINER_FILE_RE.match(head_name):
        return ServeLogKind.CONTAINER, "", segment, shard
    match = _SIDECAR_FILE_RE.match(head_name)
    if match:
        return ServeLogKind.SIDECAR, match.group("cname"), segment, shard
    return None


def _parse_flat_log_name(filename: str) -> Optional[ServeLogName]:
    # A sidecar carries two segments after "container" and a workload container
    # one, so the two patterns cannot both match.
    match = _FLAT_SIDECAR_RE.match(filename)
    if match:
        return ServeLogName(
            model_instance_id=int(match.group("id")),
            restart_count=int(match.group("rc")),
            kind=ServeLogKind.SIDECAR,
            container_name=match.group("cname"),
            flat=True,
        )

    match = _FLAT_CONTAINER_RE.match(filename)
    if match:
        return ServeLogName(
            model_instance_id=int(match.group("id")),
            restart_count=int(match.group("rc")),
            kind=ServeLogKind.CONTAINER,
            flat=True,
        )

    match = _FLAT_MAIN_RE.match(filename)
    if match:
        return ServeLogName(
            model_instance_id=int(match.group("id")),
            restart_count=int(match.group("rc")),
            kind=ServeLogKind.MAIN,
            flat=True,
        )

    match = _FLAT_LEGACY_MAIN_RE.match(filename)
    if match:
        return ServeLogName(
            model_instance_id=int(match.group("id")),
            restart_count=0,
            kind=ServeLogKind.MAIN,
            flat=True,
            legacy=True,
        )

    return None


def parse_serve_log_path(path: Path) -> Optional[ServeLogName]:
    """Read what a serve log's location says about it, or report it is not one.

    The single place the serve log layout lives: the per-instance directory a
    current worker writes, plus the flat names earlier releases left behind.
    A path this cannot read returns None rather than a default -- read as
    restart 0 it would be retired by the retention window.

    Args:
        path: Full path to the file.

    Returns:
        What the location says, or None if the path is not a serve log.
    """
    restart_dir = path.parent
    if _RESTART_DIR_RE.match(restart_dir.name):
        instance = parse_instance_dir_name(restart_dir.parent.name)
        located = _parse_log_file_name(path.name)
        if instance is not None and located is not None:
            instance_name, model_instance_id = instance
            kind, container_name, segment, shard = located
            return ServeLogName(
                model_instance_id=model_instance_id,
                restart_count=int(restart_dir.name),
                kind=kind,
                container_name=container_name,
                instance_name=instance_name,
                segment=segment,
                shard=shard,
            )
        return None

    return _parse_flat_log_name(path.name)


def canonical_log_path(
    log_dir: Path, model_instance_name: str, parsed: ServeLogName
) -> Path:
    """Where a parsed log file would be written today.

    Adoption compares this against the file's current path to decide whether it
    predates the per-instance directory.

    Args:
        log_dir: Directory holding serve logs.
        model_instance_name: Model instance name, sanitized on the way in.
        parsed: What parse_serve_log_path read off the file.

    Returns:
        The path the current writer would use for that same file.
    """
    if parsed.kind is ServeLogKind.SIDECAR:
        return sidecar_container_log_path(
            log_dir,
            model_instance_name,
            parsed.model_instance_id,
            parsed.container_name,
            parsed.restart_count,
        )
    if parsed.kind is ServeLogKind.CONTAINER:
        return container_log_path(
            log_dir,
            model_instance_name,
            parsed.model_instance_id,
            parsed.restart_count,
        )
    return main_log_path(
        log_dir,
        model_instance_name,
        parsed.model_instance_id,
        parsed.restart_count,
    )


def extract_restart_count(path: Path) -> Optional[int]:
    """Restart count of a main serve log.

    Args:
        path: Full path to the log file.

    Returns:
        The restart count, or None if the path is not a main serve log.
    """
    parsed = parse_serve_log_path(path)
    if parsed is None or parsed.kind is not ServeLogKind.MAIN:
        return None
    return parsed.restart_count


async def has_log_content(log_file: Path) -> bool:
    """Check if log file has any actual content.

    Args:
        log_file: Path to log file

    Returns:
        True if file exists and has size > 0
    """
    return await asyncio.to_thread(
        lambda: log_file.exists() and log_file.stat().st_size > 0
    )


def find_instance_log_dir(log_dir: Path, model_instance_id: int) -> Optional[Path]:
    """The directory currently holding an instance's logs, whatever it is named.

    Membership is decided by the parsed id, never by a glob: 2945* also matches
    12945, and the instance name sits in front of the id.

    Args:
        log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.

    Returns:
        The instance's directory, or None if it has none yet.
    """
    if not log_dir.is_dir():
        return None
    matches = []
    for child in log_dir.iterdir():
        if not child.is_dir():
            continue
        instance = parse_instance_dir_name(child.name)
        if instance is not None and instance[1] == model_instance_id:
            matches.append(child)
    if not matches:
        return None
    if len(matches) > 1:
        # One of them is then invisible to every caller, retention included.
        logger.warning(
            f"Model instance {model_instance_id} has more than one log "
            f"directory: {', '.join(sorted(child.name for child in matches))}"
        )
    return min(matches)


def restart_log_dirs(log_dir: Path, model_instance_id: int) -> Dict[int, Path]:
    """Each restart directory an instance has on disk, by restart count.

    Args:
        log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.

    Returns:
        Mapping from restart count to directory. A directory whose name is not
        a restart count is left out, and so out of every retention decision.
    """
    found: Dict[int, Path] = {}
    instance_dir = find_instance_log_dir(log_dir, model_instance_id)
    if instance_dir is None:
        return found
    try:
        children = list(instance_dir.iterdir())
    except FileNotFoundError:
        # Purged or renamed since it was found.
        return found
    for child in children:
        if child.is_dir() and _RESTART_DIR_RE.match(child.name):
            found[int(child.name)] = child
    return found


def flat_instance_logs(
    log_dir: Path, model_instance_id: int
) -> List[Tuple[Path, ServeLogName]]:
    """An instance's logs that an earlier release left in the serve directory.

    These are what adoption migrates into the instance's directory. A file the
    frozen flat grammar cannot read is left out, so nothing downstream can
    retire a name it does not understand.

    Args:
        log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.

    Returns:
        (path, parsed name) pairs, in no particular order.
    """
    if not log_dir.is_dir():
        return []
    found = []
    for path in log_dir.glob("*.log"):
        parsed = _parse_flat_log_name(path.name)
        if parsed and parsed.model_instance_id == model_instance_id:
            found.append((path, parsed))
    return found


def instance_log_files(
    log_dir: Path, model_instance_id: int
) -> List[Tuple[Path, ServeLogName]]:
    """Every serve log belonging to one instance, wherever it lives.

    The instance's own directory plus whatever an earlier release left flat
    beside it, so an upgraded but not yet restarted node loses nothing.

    Args:
        log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.

    Returns:
        (path, parsed name) pairs, in no particular order.
    """
    found = []
    for restart_dir in restart_log_dirs(log_dir, model_instance_id).values():
        try:
            children = list(restart_dir.iterdir())
        except FileNotFoundError:
            # Retention removed this restart since the listing above; a restart
            # is exactly when someone has the logs open.
            continue
        for path in children:
            parsed = parse_serve_log_path(path)
            if parsed:
                found.append((path, parsed))
    found.extend(flat_instance_logs(log_dir, model_instance_id))
    return _without_covered_shards(found)


def _without_covered_shards(
    entries: List[Tuple[Path, "ServeLogName"]],
) -> List[Tuple[Path, "ServeLogName"]]:
    # A shard the marker counts in is no longer part of its log, though it may
    # still be on disk: the marker is written before the shard is removed.
    dropped = {}
    for path, parsed in entries:
        if parsed.segment is ServeLogSegment.MARKER:
            counts = read_marker_counts(path)
            if counts:
                dropped[split_segment_log_path(path)[0]] = counts.dropped_shards
    return [
        (path, parsed)
        for path, parsed in entries
        if parsed.segment is not ServeLogSegment.TAIL
        or parsed.shard > dropped.get(split_segment_log_path(path)[0], 0)
    ]


_SEGMENT_ORDER = {
    ServeLogSegment.HEAD: 0,
    ServeLogSegment.MARKER: 1,
    ServeLogSegment.TAIL: 2,
}


def _sort_key(entry: Tuple[Path, ServeLogName]) -> Tuple[int, int, int, int, int, str]:
    # Writing order: oldest restart, then flat before the directory it migrates
    # into, legacy {id}.log before the numbered one, then head, marker and tail
    # shards. The name breaks any remaining tie so repeated calls agree.
    path, parsed = entry
    return (
        parsed.restart_count,
        0 if parsed.flat else 1,
        0 if parsed.legacy else 1,
        _SEGMENT_ORDER[parsed.segment],
        parsed.shard,
        path.name,
    )


def stream_log_files(
    entries: List[Tuple[Path, ServeLogName]],
    restart_count: Optional[int],
    container_name: Optional[str] = None,
) -> List[Path]:
    """One restart's stream as an ordered file list, out of one walk.

    The order a line number is counted against: the restart's main logs then
    its container logs, or a named sidecar's logs on their own. Listing every
    stream of every restart this way costs one directory walk rather than two
    per stream.

    Args:
        entries: Everything `instance_log_files` found for the instance.
        restart_count: The restart to read, None for every one of them.
        container_name: A sidecar's name to read only its logs; "default" or
            None for the main and workload container logs.

    Returns:
        The files, in reading order.
    """
    if container_name and container_name != "default":
        wanted = [(ServeLogKind.SIDECAR, container_name)]
    else:
        wanted = [(ServeLogKind.MAIN, None), (ServeLogKind.CONTAINER, None)]

    paths: List[Path] = []
    for kind, name in wanted:
        chosen = [
            entry
            for entry in entries
            if entry[1].kind is kind
            and (name is None or entry[1].container_name == name)
            and (restart_count is None or entry[1].restart_count == restart_count)
        ]
        paths.extend(path for path, _ in sorted(chosen, key=_sort_key))
    return paths


def select_log_files(
    entries: List[Tuple[Path, ServeLogName]],
    container: bool = False,
    restart_count: Optional[int] = None,
    container_name: Optional[str] = None,
) -> List[Path]:
    """Pick one kind of log out of one walk, sorted by restart count.

    Args:
        entries: Everything `instance_log_files` found for the instance.
        container: If True, get container logs; if False, get main logs
        restart_count: If specified, only return logs for this restart count
        container_name: If specified with container=True, get sidecar container
            logs for this container name (e.g., "ray-head").

    Returns:
        List of log file paths sorted by restart count (oldest first)
    """
    if container and container_name:
        wanted = ServeLogKind.SIDECAR
    elif container:
        wanted = ServeLogKind.CONTAINER
    else:
        wanted = ServeLogKind.MAIN

    entries = [(p, n) for p, n in entries if n.kind is wanted]

    if container_name:
        entries = [(p, n) for p, n in entries if n.container_name == container_name]
    if restart_count is not None:
        entries = [(p, n) for p, n in entries if n.restart_count == restart_count]

    return [path for path, _ in sorted(entries, key=_sort_key)]


async def get_all_log_files(
    log_dir: Path,
    model_instance_id: int,
    container: bool = False,
    restart_count: Optional[int] = None,
    container_name: Optional[str] = None,
) -> List[Path]:
    """Walk an instance's logs and pick one kind, as `select_log_files` does.

    Args:
        log_dir: Directory containing log files
        model_instance_id: Model instance ID
        container: If True, get container logs; if False, get main logs
        restart_count: If specified, only return logs for this restart count
        container_name: If specified with container=True, get sidecar container
            logs for this container name (e.g., "ray-head").

    Returns:
        List of log file paths sorted by restart count (oldest first)
    """
    entries = await asyncio.to_thread(instance_log_files, log_dir, model_instance_id)
    return select_log_files(entries, container, restart_count, container_name)


async def resolve_restart_count(
    log_dir: Path, model_instance_id: int, previous: bool
) -> Optional[int]:
    """Resolve ``previous`` flag to an actual restart_count from disk files.

    Returns:
        The restart_count integer for the target log set, or ``None`` when
        no log files exist on disk yet.
    """
    files = await get_all_log_files(log_dir, model_instance_id, container=False)
    if not files:
        return None
    counts = sorted(set(extract_restart_count(f) for f in files))
    if previous and len(counts) >= 2:
        return counts[-2]
    return counts[-1]


def group_container_names_by_restart(
    entries: List[Tuple[Path, ServeLogName]],
) -> Dict[int, List[str]]:
    """Map each restart to the container names that have logs on disk for it.

    A restart with any container log carries "default" for the workload's own
    container, followed by the sidecar names in sorted order.

    Args:
        entries: Everything `instance_log_files` found for the instance.

    Returns:
        Mapping from restart_count to container names.
    """
    sidecar_names_by_restart: Dict[int, set] = defaultdict(set)
    default_container_rcs = set()
    for _path, parsed in entries:
        if parsed.kind is ServeLogKind.SIDECAR:
            sidecar_names_by_restart[parsed.restart_count].add(parsed.container_name)
        elif parsed.kind is ServeLogKind.CONTAINER:
            default_container_rcs.add(parsed.restart_count)

    container_names_by_restart: Dict[int, List[str]] = defaultdict(list)
    for rc, names in sidecar_names_by_restart.items():
        container_names_by_restart[rc] = ["default"] + sorted(names)
    # Also add "default" for restarts that have container logs but no sidecars.
    for rc in default_container_rcs:
        if rc not in container_names_by_restart:
            container_names_by_restart[rc] = ["default"]

    return container_names_by_restart


def expected_container_log_path(main_log_path_on_disk: Path) -> Path:
    """Where the workload container's log sits beside a main log.

    The reader knows an instance by id and never by name, so it cannot build
    this path on its own -- it reads it off a main log already there.

    Args:
        main_log_path_on_disk: A main log of the restart in question.

    Returns:
        The container log path for that same restart.
    """
    parsed = parse_serve_log_path(main_log_path_on_disk)
    if parsed is not None and parsed.flat:
        return main_log_path_on_disk.with_name(
            f"{parsed.model_instance_id}.container.{parsed.restart_count}.log"
        )
    return main_log_path_on_disk.with_name(CONTAINER_LOG_NAME)


def _numbered_tail_shards(head_log_path: Path) -> List[Tuple[int, Path]]:
    prefix = f"{head_log_path.name}."
    try:
        paths = list(head_log_path.parent.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    found = []
    for path in paths:
        if not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix) :]
        if _SHARD_RE.match(suffix):
            found.append((int(suffix), path))
    return sorted(found)


def tail_shard_log_paths(head_log_path: Path) -> List[Path]:
    """Tail shards that exist on disk for one capped log, oldest first.

    Args:
        head_log_path: The log's head.

    Returns:
        Shard paths ordered by shard number, empty when the log has no tail.
    """
    return [path for _, path in _numbered_tail_shards(head_log_path)]


def split_segment_log_path(segment_log_path: Path) -> Tuple[Path, int]:
    """Which log a file is part of, and where in it.

    Args:
        segment_log_path: Any part of a log -- its head, marker or a shard.

    Returns:
        (the log's head, the shard number), 0 for the head and for the marker,
        which sits between the head and the first shard.
    """
    head_name, _, suffix = segment_log_path.name.rpartition('.')
    if head_name.endswith(_LOG_SUFFIX) and _SHARD_RE.match(suffix):
        return segment_log_path.with_name(head_name), int(suffix)
    if is_marker_log_path(segment_log_path):
        return segment_log_path.with_name(head_name), 0
    return segment_log_path, 0


def is_marker_log_path(path: Path) -> bool:
    """Whether a file is the marker standing in for a log's dropped shards."""
    return path.name.endswith(f"{_LOG_SUFFIX}.{_MARKER_SUFFIX}")


def next_segment_log_path(segment_log_path: Path) -> Optional[Path]:
    """The shard a capped log rolled over to after this file, once it exists.

    From the file alone, "the writer has moved on" and "nothing new yet" look
    identical; a successor on disk is what tells them apart.

    Args:
        segment_log_path: Any part of a log -- its head or one of its shards.

    Returns:
        The next shard's path, or None while the writer is still on this file.
    """
    head, shard = split_segment_log_path(segment_log_path)
    if segment_log_path != head and shard == 0:
        # The marker: written whole, and the first shard went when it appeared.
        return surviving_segment_log_path(segment_log_path)
    successor = tail_shard_log_path(head, shard + 1)
    if successor.exists():
        return successor
    # The head is never removed, so a reader still in it when the cap dropped
    # the first shard learns it from the marker, not from the file.
    if shard == 0 and marker_log_path(head).exists():
        return surviving_segment_log_path(head)
    return None


def surviving_segment_log_path(segment_log_path: Path) -> Optional[Path]:
    """The oldest shard still on disk after this file.

    A reader that fell a whole budget behind finds its next shards already
    dropped, and the log goes on only in the ones after them. Lists the
    directory, so it is for a reader that knows it has to move on, not for
    polling.

    Args:
        segment_log_path: Any part of a log -- its head, marker or a shard.

    Returns:
        That shard's path, or None when nothing after this file exists yet.
    """
    head, shard = split_segment_log_path(segment_log_path)
    counts = read_marker_counts(marker_log_path(head))
    covered = counts.dropped_shards if counts else 0
    for number, path in _numbered_tail_shards(head):
        if number > max(shard, covered):
            return path
    return None


def newest_segment_log_path(head_log_path: Path) -> Path:
    """The file a capped log is currently being appended to.

    Anything reading a log's last lines has to ask here: once the cap has
    rotated, the head's last line stopped being the log's last line.

    Args:
        head_log_path: The log's head.

    Returns:
        The newest tail shard, or the head when there is no tail.
    """
    shards = tail_shard_log_paths(head_log_path)
    return shards[-1] if shards else head_log_path


_MARKER_TEMPLATE = (
    "... {omitted_bytes} bytes / {omitted_lines} lines omitted here by the "
    "serving log size cap (GPUSTACK_SERVE_LOG_MAX_BYTES), through {last_dropped} ...\n"
)
_MARKER_RE = re.compile(
    r'^\.\.\. (?P<bytes>\d+) bytes / (?P<lines>\d+) lines omitted .*, '
    r'through \S*\.(?P<shards>\d+) \.\.\.$'
)

# The most tail shards a log is cut into, whatever the head is set to.
_MAX_TAIL_SHARDS = 64

# How far past its size a segment may run waiting for the open line to end
# before it rolls anyway: a progress bar redrawn with '\r' never ends its line.
_MAX_OPEN_LINE_BYTES = 1 << 20


class MarkerCounts(NamedTuple):
    """What a marker says the size cap dropped.

    The dropped lines keep their numbers, held by the marker, so every line
    after them keeps its own: a page number means the same lines whatever the
    cap drops meanwhile.

    Attributes:
        omitted_bytes: Bytes dropped.
        omitted_lines: Line numbers the marker holds.
        dropped_shards: Shards 1 to this one are counted in. The marker is
            written before a shard is removed, so one of them may still be on
            disk; it is no longer part of the log.
    """

    omitted_bytes: int
    omitted_lines: int
    dropped_shards: int


def read_marker_counts(path: Path) -> Optional[MarkerCounts]:
    """What a marker says the size cap dropped.

    Args:
        path: The marker's path.

    Returns:
        The counts, or None when the marker is missing or unreadable.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, ValueError):
        return None
    match = _MARKER_RE.match(text)
    if match is None:
        return None
    return MarkerCounts(
        int(match.group("bytes")), int(match.group("lines")), int(match.group("shards"))
    )


def _count_lines(path: Path) -> int:
    # Counted the way the line index counts, a trailing fragment as one line,
    # so the numbers the marker takes over are exactly the ones the shard held.
    lines, last = 0, b'\n'
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            lines += chunk.count(b'\n')
            last = chunk[-1:]
    return lines + (last != b'\n')


class _TextSinkBuffer(io.BufferedIOBase):
    """The binary side of a CappedLogWriter, for callers that write bytes.

    Bytes are decoded and written through the text side, so the budget counts
    them and a rotation cannot leave them in a closed file. Each thread gets
    its own incremental decoder: a character one thread splits across two
    writes is held until its rest arrives, and no other thread's bytes join it.
    """

    def __init__(self, sink: "CappedLogWriter"):
        self._sink = sink
        self._decoders = threading.local()

    def write(self, data) -> int:
        decoder = getattr(self._decoders, 'decoder', None)
        if decoder is None:
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            self._decoders.decoder = decoder
        data = bytes(data)
        self._sink.write(decoder.decode(data))
        return len(data)

    def writable(self) -> bool:
        return True

    def flush(self):
        self._sink.flush()

    def close(self):
        self._sink.close()

    @property
    def closed(self) -> bool:
        return self._sink.closed


class CappedLogWriter(io.TextIOBase):
    """A text sink that keeps one serving log inside a byte budget.

    The first ``head_bytes`` land in the head and are never rewritten -- what
    explains a failed start is emitted in the first seconds, and a tail-only
    cap would discard exactly that. Everything after rotates through numbered
    tail shards; once they fill the rest of the budget the oldest is dropped
    and counted into a one-line marker, so the gap says how large it is.

    A budget of 0 disables all of it and the object is the plain line-buffered
    file it stands in for.

    It stands in for ``sys.stdout`` and ``sys.stderr`` of a serving process, so
    every write is serialized: a rotation closes the file another thread may be
    about to write to.

    Attributes:
        head_path: The log's head, which is also the name the log is known by.
    """

    def __init__(
        self,
        path: Union[str, Path],
        append: bool = False,
        max_bytes: Optional[int] = None,
        head_bytes: Optional[int] = None,
    ):
        """
        Args:
            path: The log's head path.
            append: Continue a log left on disk instead of starting it over.
            max_bytes: Budget for the whole log; defaults to the configured one.
            head_bytes: Budget for the head; defaults to the configured one.
        """
        self.head_path = Path(path)
        # Reentrant: a signal handler that logs, or the report of an exception
        # a finalizer raised, writes here from inside a write on the same thread.
        self._lock = threading.RLock()
        # Logged only once the lock is released: a logging handler holds its
        # own lock while it writes here, so logging under ours would take the
        # two in the other order.
        self._warnings: List[str] = []
        self._local = threading.local()
        self._buffer = _TextSinkBuffer(self)
        max_bytes = envs.SERVE_LOG_MAX_BYTES if max_bytes is None else max_bytes
        head_bytes = envs.SERVE_LOG_HEAD_BYTES if head_bytes is None else head_bytes

        self._max_bytes = max_bytes
        # A head filling the whole budget would leave no tail, and the tail
        # is the half that holds the failure.
        self._head_bytes = min(head_bytes, max_bytes // 2) if max_bytes > 0 else 0
        # Shards rotate at the head's size -- the stock 8 MiB head and 64 MiB
        # budget make seven of 8 MiB -- but no smaller than the rest of the
        # budget spread over _MAX_TAIL_SHARDS, or a small head would cut the
        # log into a file per write.
        self._shard_bytes = max(
            self._head_bytes, (max_bytes - self._head_bytes) // _MAX_TAIL_SHARDS, 1
        )
        self._max_shards = max(1, (max_bytes - self._head_bytes) // self._shard_bytes)

        self._omitted_bytes = 0
        self._omitted_lines = 0
        self._dropped_shards = 0
        self._shard = 0
        self._segment_bytes = 0
        self._roll_pending = False
        self._rolling = False

        if append:
            self._resume()
        else:
            self._start_over()

    @property
    def _capped(self) -> bool:
        return self._max_bytes > 0

    def _open(self, path: Path, append: bool):
        return open(path, 'a' if append else 'w', buffering=1, encoding='utf-8')

    def _start_over(self):
        # A fresh head beside a stale tail reads as a log that begins in the
        # middle, so the previous log's parts go too.
        stale = [marker_log_path(self.head_path), *tail_shard_log_paths(self.head_path)]
        for path in stale:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Failed to remove stale log segment {path}: {e}")
        self._file = self._open(self.head_path, append=False)
        self._note_segment_size(0)

    def _resume(self):
        self._read_marker()
        shards = _numbered_tail_shards(self.head_path)
        current = self.head_path
        if shards:
            self._shard, current = shards[-1]
        try:
            size = current.stat().st_size
        except OSError:
            size = 0
        self._file = self._open(current, append=True)
        self._note_segment_size(size)

    def _note_segment_size(self, size: int, line_ended: bool = True):
        self._segment_bytes = size
        limit = self._head_bytes if self._shard == 0 else self._shard_bytes
        # An empty segment never rolls, so a head of 0 still keeps the first
        # line: an empty head reads as an empty log to everything that checks.
        # Nor does one in the middle of a line -- print() writes the newline
        # separately, and a runtime splits a long line into chunks -- since a
        # line cut across two files counts as two, unless it never ends.
        self._roll_pending = (
            self._capped
            and size > 0
            and size >= limit
            and (line_ended or size >= limit + _MAX_OPEN_LINE_BYTES)
        )

    def write(self, s: str) -> int:
        # Nothing to write must not roll: it would leave an empty trailing shard.
        if not s:
            return 0
        try:
            with self._holding_lock():
                if self._roll_pending and not self._rolling:
                    self._roll()
                written = self._file.write(s)
                if self._capped:
                    size = len(s) if s.isascii() else len(s.encode('utf-8', 'replace'))
                    self._note_segment_size(
                        self._segment_bytes + size, s.endswith('\n')
                    )
                return written
        finally:
            self._report_warnings()

    @contextlib.contextmanager
    def _holding_lock(self):
        # Counted per thread, so a write that came back from inside a locked
        # section knows this thread still holds the lock.
        with self._lock:
            self._local.depth = getattr(self._local, 'depth', 0) + 1
            try:
                yield
            finally:
                self._local.depth -= 1

    def _report_warnings(self):
        # Only from outside every locked section of this thread: one that came
        # back from inside another would log under the lock. A warning that
        # comes back here as a write may roll and warn again; that one waits
        # for the next write rather than recursing.
        local = self._local
        if (
            not self._warnings
            or getattr(local, 'depth', 0)
            or getattr(local, 'reporting', False)
        ):
            return
        with self._lock:
            warnings, self._warnings = self._warnings, []
        local.reporting = True
        try:
            for message in warnings:
                logger.warning(message)
        finally:
            local.reporting = False

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def _roll(self):
        # Rolling on the write after the full one, not the one that filled
        # the segment, keeps whole lines out of the seam and leaves no empty
        # trailing shard for a reader to mistake for the end of the log.
        # A successor on disk tells a follower this file is finished, so all of
        # it has to be on disk before the successor is. On a closed log this
        # raises as a closed file's write does, before any shard is opened.
        self._file.flush()
        # A write that comes back from inside the roll lands in whichever file
        # is current as is: rolling again under it would open the same shard
        # twice, or recurse.
        self._rolling = True
        try:
            shard = self._shard + 1
            # Swapped in before the old file closes, so what reads self._file
            # without the lock (closed, fileno) never finds it closed. Numbered
            # only once it opened: a skipped number would stop a follower.
            previous, self._file = self._file, self._open(
                tail_shard_log_path(self.head_path, shard), append=False
            )
            previous.close()
            self._shard = shard
            self._note_segment_size(0)
            self._drop_expired_shards()
        finally:
            self._rolling = False

    def _drop_expired_shards(self):
        shards = _numbered_tail_shards(self.head_path)
        excess = sum(number > self._dropped_shards for number, _ in shards)
        excess -= self._max_shards
        for number, path in shards:
            if number > self._dropped_shards:
                if excess <= 0:
                    return
                excess -= 1
                try:
                    size = path.stat().st_size
                    lines = _count_lines(path)
                except OSError as e:
                    self._warnings.append(f"Failed to drop tail shard {path}: {e}")
                    return
                # The marker takes the shard over before it goes, so no reader
                # finds it gone with its lines not yet counted.
                if not self._write_marker(
                    self._omitted_bytes + size, self._omitted_lines + lines, number
                ):
                    return
                self._omitted_bytes += size
                self._omitted_lines += lines
                self._dropped_shards = number
            # Oldest first, one counted in but left by a failed removal before
            # the rest, so the shards on disk stay one run.
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                self._warnings.append(f"Failed to drop tail shard {path}: {e}")
                return

    def _write_marker(
        self, omitted_bytes: int, omitted_lines: int, dropped_shards: int
    ) -> bool:
        marker = marker_log_path(self.head_path)
        # Replaced whole, never rewritten in place: a reader numbering lines by
        # it, or a restart resuming its totals, must not find it half written.
        temporary = marker.with_name(f"{marker.name}.tmp")
        try:
            temporary.write_text(
                _MARKER_TEMPLATE.format(
                    omitted_bytes=omitted_bytes,
                    omitted_lines=omitted_lines,
                    last_dropped=tail_shard_log_path(
                        self.head_path, dropped_shards
                    ).name,
                ),
                encoding='utf-8',
            )
            os.replace(temporary, marker)
            return True
        except OSError as e:
            self._warnings.append(
                f"Failed to record the truncation of {self.head_path}: {e}"
            )
            return False

    def _read_marker(self):
        # The totals live in the marker, not in memory, so a worker restart
        # keeps counting instead of starting from zero.
        counts = read_marker_counts(marker_log_path(self.head_path))
        if counts:
            self._omitted_bytes, self._omitted_lines, self._dropped_shards = counts

    def flush(self):
        with self._holding_lock():
            self._file.flush()

    def close(self):
        with self._holding_lock():
            self._file.close()
        self._report_warnings()

    @property
    def closed(self) -> bool:
        return self._file.closed

    @property
    def encoding(self) -> str:
        return self._file.encoding

    @property
    def buffer(self) -> _TextSinkBuffer:
        return self._buffer

    @property
    def name(self) -> str:
        return str(self.head_path)

    def fileno(self) -> int:
        return self._file.fileno()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def __enter__(self) -> "CappedLogWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class LogSource(ABC):
    """Abstract base class for log sources.

    Each log source knows how to get its files and wait for them if needed.
    This follows the Strategy pattern used in the project
    (similar to LoadBalancingStrategy).
    """

    @abstractmethod
    async def get_files(self) -> List[Path]:
        """Get log files for this source.

        Returns:
            List of log file paths (may be empty)
        """
        pass

    @abstractmethod
    def get_file_pattern(self) -> str:
        """Get the file pattern for this source (for logging purposes).

        Returns:
            Pattern string for identification
        """
        pass

    async def is_valid_source(self) -> bool:
        """Check if this source is valid and should be waited for.

        Override this method to add preconditions for waiting.
        For example, DownloadLogSource returns False if log_path is None.

        Returns:
            True if the source is valid and should be waited for
        """
        return True

    async def get_files_with_log(self) -> List[Path]:
        """Get files with debug logging.

        Returns:
            List of log file paths
        """
        files = await self.get_files()
        if files:
            logger.debug(f"Found files for {self.get_file_pattern()}: {files}")
        return files

    async def wait_for_files(self, timeout: int = 300, **kwargs) -> List[Path]:
        """Wait for log files to appear.

        Uses the project's check_with_retries utility for consistent retry behavior.
        Subclasses should override is_valid_source() instead of this method
        unless they need completely different waiting logic.

        Args:
            timeout: Maximum time to wait in seconds
            **kwargs: Additional arguments for subclass implementations

        Returns:
            List of log file paths
        """
        if not await self.is_valid_source():
            return []

        async def check():
            files = await self.get_files()
            if not files:
                raise FileNotFoundError(
                    f"Log files not found for source: {self.get_file_pattern()}"
                )
            return files

        files = await file.check_with_retries(check, timeout=timeout)
        logger.debug(f"Found files after waiting: {self.get_file_pattern()}")
        return files

    async def wait_for_files_if_needed(
        self,
        follow: bool,
        timeout: int = 300,
        **kwargs,
    ) -> List[Path]:
        """Get files, waiting if necessary in follow mode.

        Args:
            follow: Whether in follow mode (triggers waiting if files not found)
            timeout: Maximum time to wait in seconds
            **kwargs: Additional arguments for wait_for_files

        Returns:
            List of log file paths
        """
        files = await self.get_files_with_log()
        if files:
            return files

        if follow and await self.is_valid_source():
            try:
                files = await self.wait_for_files(timeout=timeout, **kwargs)
            except Exception:
                pass

        return files


class DownloadLogSource(LogSource):
    """Log source for download logs."""

    def __init__(self, log_path: Optional[str]):
        self.log_path = Path(log_path) if log_path else None

    async def get_files(self) -> List[Path]:
        if not self.log_path:
            return []
        if await asyncio.to_thread(self.log_path.exists):
            return [self.log_path]
        return []

    def get_file_pattern(self) -> str:
        return str(self.log_path) if self.log_path else "download_log"

    async def is_valid_source(self) -> bool:
        return self.log_path is not None


class MainLogSource(LogSource):
    """Log source for main (historical) logs."""

    def __init__(
        self,
        log_dir: Path,
        model_instance_id: int,
        restart_count: Optional[int] = None,
    ):
        self.log_dir = log_dir
        self.model_instance_id = model_instance_id
        self.restart_count = restart_count

    async def get_files(self) -> List[Path]:
        return await get_all_log_files(
            self.log_dir,
            self.model_instance_id,
            restart_count=self.restart_count,
        )

    def get_file_pattern(self) -> str:
        restart = "*" if self.restart_count is None else self.restart_count
        return f"*.{self.model_instance_id}/{restart}/{MAIN_LOG_NAME}"


class ContainerLogSource(LogSource):
    """Log source for container logs."""

    def __init__(
        self,
        log_dir: Path,
        model_instance_id: int,
        restart_count: Optional[int] = None,
    ):
        self.log_dir = log_dir
        self.model_instance_id = model_instance_id
        self.restart_count = restart_count

    async def get_files(self) -> List[Path]:
        return await get_all_log_files(
            self.log_dir,
            self.model_instance_id,
            container=True,
            restart_count=self.restart_count,
        )

    def get_file_pattern(self) -> str:
        restart = "*" if self.restart_count is None else self.restart_count
        return f"*.{self.model_instance_id}/{restart}/{CONTAINER_LOG_NAME}"

    def _get_expected_file(self, main_log_files: List[Path]) -> Optional[Path]:
        """Infer expected container log file from main log files.

        Args:
            main_log_files: Main log files to infer container log name from

        Returns:
            Expected container log file path, or None if cannot infer
        """
        if not main_log_files:
            return None

        # The container log sits beside the main log, whichever layout the
        # main log is in.
        return expected_container_log_path(main_log_files[-1])

    async def wait_for_files(
        self,
        timeout: int = 300,
        main_log_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """Wait for container log files to appear.

        Container logs need special handling because the expected file name
        depends on the main log's restart count.

        Args:
            timeout: Maximum time to wait in seconds
            main_log_files: Main log files to infer expected container log name

        Returns:
            List of container log file paths
        """
        # First check if files already exist
        files = await self.get_files()
        if files:
            return files

        # Try to infer expected file from main logs
        expected_file = self._get_expected_file(main_log_files)
        if not expected_file:
            return []

        # Reuse base class retry logic with custom check function
        async def check():
            if not await asyncio.to_thread(expected_file.exists):
                raise FileNotFoundError(
                    f"Container log file not found: {expected_file}"
                )
            return [expected_file]

        files = await file.check_with_retries(check, timeout=timeout)
        logger.debug(f"Found container log after waiting: {expected_file}")
        return files

    async def wait_for_files_if_needed(
        self,
        follow: bool,
        timeout: int = 300,
        main_log_files: Optional[List[Path]] = None,
    ) -> List[Path]:
        """Get files, waiting if necessary in follow mode.

        Args:
            follow: Whether in follow mode (triggers waiting if files not found)
            timeout: Maximum time to wait in seconds
            main_log_files: Main log files to infer expected container log name

        Returns:
            List of log file paths
        """
        files = await self.get_files_with_log()
        if files:
            return files

        if follow:
            files = await self.wait_for_files(
                timeout=timeout, main_log_files=main_log_files
            )

        return files

    async def has_content(self) -> bool:
        """Check if any container log file has actual content."""
        files = await self.get_files()
        for f in files:
            if await has_log_content(f):
                return True
        return False


async def monitor_container_content(
    container_source: ContainerLogSource,
    stop_event: asyncio.Event,
    check_interval: float = 1.0,
) -> None:
    """Monitor container logs for content and signal stop when found.

    Args:
        container_source: Container log source to monitor
        stop_event: Event to set when container log has content
        check_interval: Time between checks in seconds
    """
    logger.debug("Starting background task to monitor container logs for content")

    while not stop_event.is_set():
        await asyncio.sleep(check_interval)

        if stop_event.is_set():
            return

        if await container_source.has_content():
            logger.debug("Container log now has content, stopping main log follow")
            stop_event.set()
            return
