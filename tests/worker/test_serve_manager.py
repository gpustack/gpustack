import asyncio
from datetime import datetime, timezone
import io
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.routes.worker.logs import (
    combined_log_generator,
    get_all_log_files,
    get_serve_log_options,
    get_serve_logs,
    merged_log_generator,
    resolve_restart_count,
)
from gpustack.schemas.models import (
    BackendEnum,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.log_sources import (
    CappedLogWriter,
    ServeLogKind,
    ServeLogName,
    ServeLogSegment,
    container_log_path,
    extract_restart_count,
    instance_log_dir,
    instance_log_files,
    main_log_path,
    marker_log_path,
    newest_segment_log_path,
    parse_serve_log_path,
    restart_log_dir,
    sanitize_instance_name,
    sidecar_container_log_path,
    tail_shard_log_paths,
)
from gpustack.worker.logs import LogOptions, log_generator
from gpustack.worker.serve_manager import (
    ServeManager,
    _describe_workload_failure,
    _LogPersistence,
)
from gpustack_runtime.deployer import WorkloadStatusStateEnum
from tests.utils.model import new_model, new_model_instance


def _fake_stop_event(max_waits: int = 100):
    """A stop event whose wait() returns instantly (tests aren't driven by real
    time) and stays unset, so the log persistence loop is driven purely by the
    get_workload state sequence. It auto-sets after max_waits waits so a
    mis-sized mock or a runaway loop fails the test fast instead of hanging CI."""
    state = {"waits": 0, "stopped": False}

    def is_set():
        return state["stopped"]

    def wait(timeout=None):
        state["waits"] += 1
        if state["waits"] >= max_waits:
            state["stopped"] = True
        return state["stopped"]

    stop_event = MagicMock()
    stop_event.is_set.side_effect = is_set
    stop_event.wait.side_effect = wait
    return stop_event


def _fake_thread(alive: bool, ends_on_join: bool = True):
    """A stand-in thread. Joining one ends it, the way a real log thread back
    from the runtime does -- unless it is the kind that outlasts the wait."""
    thread = MagicMock()
    thread.is_alive.return_value = alive
    if ends_on_join:
        thread.join.side_effect = lambda timeout=None: setattr(
            thread.is_alive, "return_value", False
        )
    return thread


def _log_persistence(main_alive: bool):
    """A generation with the given main-thread liveness, always beside a live
    (forever-polling) sidecar discovery thread."""
    persistence = _LogPersistence(MagicMock(), _fake_thread(main_alive))
    persistence.add_aux_thread(_fake_thread(True))
    return persistence


def _get_workload_sequence(states):
    """side_effect for a patched get_workload. The recovery grace-poll queries
    get_workload several times per stream EOF, so once the sequence reaches its
    terminal state it must keep returning it: a list that runs dry would raise
    IndexError, which _container_still_running treats as "still alive", spinning
    the reconnect loop forever."""
    remaining = list(states)

    def next_state(name):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return next_state


def _build_serve_manager(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def test_sync_model_instances_state_marks_main_unreachable_when_subordinate_unreachable():
    manager, clientset = _build_serve_manager()

    model_instance = new_model_instance(
        1,
        "distributed-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.worker_ip = "127.0.0.1"
    model_instance.port = 8000
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2,
                worker_name="worker-2",
                worker_ip="10.0.0.2",
                state=ModelInstanceStateEnum.UNREACHABLE,
                state_message="Worker is unreachable from the server",
            )
        ],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.UNREACHABLE,
        state_message=(
            "Distributed serving unreachable in subordinate worker "
            "10.0.0.2: Worker is unreachable from the server."
        ),
    )


def test_restart_error_model_instance_uses_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.ERROR,
    )
    model_instance.restart_count = 20
    model_instance.last_restart_time = datetime.now(timezone.utc)

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
        patch("gpustack.worker.serve_manager.logger"),
    ):
        manager._restart_error_model_instance(model_instance)

    update_model_instance.assert_called_once_with(
        model_instance.id,
        restart_count=21,
        last_restart_time=ANY,
        state=ModelInstanceStateEnum.SCHEDULED,
        state_message="",
    )


def test_restart_model_instance_preserves_transient_backoff_count():
    manager, _ = _build_serve_manager()
    model_instance = new_model_instance(
        1,
        "restarted-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.SCHEDULED,
    )
    manager._restart_backoff_counts[model_instance.id] = 1

    with (
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_start_model_instance"),
        # _stop_model_instance runs for real to exercise clear_restart_backoff=
        # False, but its delete_workload side effect would hit the runtime socket.
        patch("gpustack.worker.serve_manager.delete_workload"),
    ):
        manager._restart_model_instance(model_instance)

    assert manager._restart_backoff_counts[model_instance.id] == 1


# --- serve log discovery across the pre-v2.2.0 {id}.log naming ---


SIDECAR_LOG = "container.ray-head.log"


def _write_serve_logs(tmp_path: Path, *names: str) -> Path:
    """Write flat logs, the shape earlier releases left behind."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (serve_dir / name).write_text("x", encoding="utf-8")
    return serve_dir


def _write_restart_logs(
    serve_dir: Path,
    instance_name: str,
    model_instance_id: int,
    restart_count: int,
    *names: str,
) -> Path:
    """Write logs into one restart's directory, the shape written today."""
    directory = restart_log_dir(
        serve_dir, instance_name, model_instance_id, restart_count
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x", encoding="utf-8")
    return directory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "on_disk, expected_main_logs, expected_restart_count",
    [
        # Only a pre-v2.2.0 main log: discovered, and restart 0 by convention.
        (["1.log"], ["1.log"], 0),
        # Legacy and numbered coexist: both kept as restart 0, legacy first.
        (["1.log", "1.0.log"], ["1.log", "1.0.log"], 0),
        # Numbered only: unchanged behavior.
        (["1.5.log", "1.3.log"], ["1.3.log", "1.5.log"], 5),
        # Container and sidecar logs never leak into the main log branch.
        (["1.log", "1.container.0.log", "1.container.ray-head.0.log"], ["1.log"], 0),
        # No main log at all: restart count stays unresolvable.
        (["1.container.0.log"], [], None),
    ],
)
async def test_main_log_discovery_includes_legacy_file(
    tmp_path: Path, on_disk, expected_main_logs, expected_restart_count
):
    """The {id}.*.log glob cannot match the pre-v2.2.0 {id}.log name, so
    discovery has to add it back."""
    serve_dir = _write_serve_logs(tmp_path, *on_disk)

    files = await get_all_log_files(serve_dir, 1, container=False)

    assert [f.name for f in files] == expected_main_logs
    assert (
        await resolve_restart_count(serve_dir, 1, previous=False)
        == expected_restart_count
    )


@pytest.mark.asyncio
async def test_serve_log_options_after_upgrade_from_legacy_naming(tmp_path: Path):
    """The upgrade case from #5988: a legacy-only main log next to container logs
    the new worker wrote still yields one restart entry, and the container and
    sidecar branches keep returning only their own files."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "1.log",
        "1.container.0.log",
        "1.container.ray-head.0.log",
    )
    config = SimpleNamespace(log_dir=str(tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))

    response = await get_serve_log_options(request, 1)
    container_logs = await get_all_log_files(serve_dir, 1, container=True)
    sidecar_logs = await get_all_log_files(
        serve_dir, 1, container=True, container_name="ray-head"
    )

    assert len(response.restarts) == 1
    assert response.restarts[0].previous is False
    assert response.restarts[0].containers == ["default", "ray-head"]
    assert [f.name for f in container_logs] == ["1.container.0.log"]
    assert [f.name for f in sidecar_logs] == ["1.container.ray-head.0.log"]


@pytest.mark.asyncio
async def test_serve_log_options_orders_restarts_newest_first(tmp_path: Path):
    """Newest restart first, one "previous", and containers listed per restart
    so a sidecar that ran once does not appear under the other restart."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "1.0.log",
        "1.1.log",
        "1.2.log",
        "1.container.1.log",
        "1.container.2.log",
        "1.container.ray-head.2.log",
    )
    config = SimpleNamespace(log_dir=str(tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))

    response = await get_serve_log_options(request, 1)

    assert [r.previous for r in response.restarts] == [False, True, True]
    assert [r.containers for r in response.restarts] == [
        ["default", "ray-head"],
        ["default"],
        [],
    ]
    assert serve_dir.exists()


async def _collect(generator) -> str:
    return "".join([line async for line in generator])


def _log_request(tmp_path: Path):
    config = SimpleNamespace(log_dir=str(tmp_path))
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))


async def _read_range(
    tmp_path: Path,
    model_instance_id: int,
    model_file_id=None,
    container_name=None,
    **options,
):
    # Every query parameter is passed: called outside FastAPI, an omitted one
    # keeps its Query() default object rather than resolving to None.
    response = await get_serve_logs(
        _log_request(tmp_path),
        model_instance_id,
        LogOptions(**options),
        model_instance_name="",
        model_file_id=model_file_id,
        container_name=container_name,
    )
    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(c if isinstance(c, str) else c.decode() for c in chunks)
    return body, response.headers


@pytest.mark.asyncio
async def test_a_line_range_reads_one_deterministic_stream(tmp_path: Path):
    """Page numbers only mean something if the same offset names the same line
    every time. The follow path merges download and main concurrently, so the
    range path has to concatenate instead of merge."""
    serve_dir = tmp_path / "serve"
    directory = restart_log_dir(serve_dir, "qwen", 1, 0)
    directory.mkdir(parents=True)
    (directory / "main.log").write_text("m1\nm2\n", encoding="utf-8")
    (directory / "main.log.truncated").write_text("... omitted ...\n", encoding="utf-8")
    (directory / "main.log.1").write_text("m3\n", encoding="utf-8")
    (directory / "container.log").write_text("c1\nc2\n", encoding="utf-8")
    (serve_dir / "model_file_9.download.log").write_text("d1\n", encoding="utf-8")

    first, headers = await _read_range(tmp_path, 1, 9, offset=0, limit=10)
    again, _ = await _read_range(tmp_path, 1, 9, offset=0, limit=10)

    assert first == again
    assert first == "d1\nm1\nm2\n... omitted ...\nm3\nc1\nc2\n"
    assert headers["X-Log-Total-Lines"] == "7"

    page, headers = await _read_range(tmp_path, 1, 9, offset=4, limit=2)
    assert page == "m3\nc1\n"
    assert (headers["X-Log-Offset"], headers["X-Log-Line-Count"]) == ("4", "2")


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicting", [{"follow": True}, {"tail": 50}])
async def test_a_line_range_refuses_to_follow_or_tail(tmp_path: Path, conflicting):
    """Both name a different line for the same offset on the next request, so
    they cannot be combined with one."""
    with pytest.raises(BadRequestException):
        await _read_range(tmp_path, 1, offset=0, limit=10, **conflicting)


@pytest.mark.asyncio
async def test_serve_log_options_measures_each_stream(tmp_path: Path):
    """The viewer sizes its pager off these, and has to be told when the size
    cap means the log it is paging is not the whole log. A run that died before
    its container started has only its main log, and is measured all the same:
    it is the one a download is most often taken of."""
    serve_dir = tmp_path / "serve"
    directory = restart_log_dir(serve_dir, "qwen", 1, 0)
    directory.mkdir(parents=True)
    (directory / "main.log").write_text("m1\nm2\n", encoding="utf-8")
    (directory / "container.log").write_text("c1\n", encoding="utf-8")
    (directory / "container.log.truncated").write_text(
        "... gone ...\n", encoding="utf-8"
    )
    (directory / SIDECAR_LOG).write_text("s1\ns2\ns3\n", encoding="utf-8")
    _write_restart_logs(serve_dir, "qwen", 1, 1, "main.log")

    response = await get_serve_log_options(_log_request(tmp_path), 1)

    died_early, earlier = response.restarts
    assert died_early.containers == []
    assert list(died_early.container_stats) == ["default"]
    assert died_early.container_stats["default"].size_bytes == 1

    stats = earlier.container_stats
    assert sorted(stats) == ["default", "ray-head"]
    assert (stats["default"].line_count, stats["default"].truncated) == (4, True)
    assert (stats["ray-head"].line_count, stats["ray-head"].truncated) == (3, False)
    assert stats["ray-head"].size_bytes == 9


@pytest.mark.asyncio
async def test_combined_log_generator_streams_download_and_main_before_container(
    tmp_path: Path,
):
    """Only the restart being viewed is streamed, and the container output
    trails the rest. Download and main are merged concurrently, so asserting an
    order between those two would be asserting a race."""
    serve_dir = _write_serve_logs(tmp_path)
    (serve_dir / "1.0.log").write_text("older-main\n", encoding="utf-8")
    (serve_dir / "1.1.log").write_text("current-main\n", encoding="utf-8")
    (serve_dir / "1.container.1.log").write_text(
        "current-container\n", encoding="utf-8"
    )
    download_log = tmp_path / "download.log"
    download_log.write_text("downloading\n", encoding="utf-8")

    output = await _collect(
        combined_log_generator(serve_dir, 1, str(download_log), LogOptions(), "inst")
    )

    assert sorted(output.splitlines()) == [
        "current-container",
        "current-main",
        "downloading",
    ]
    assert output.splitlines()[-1] == "current-container"


@pytest.mark.asyncio
async def test_combined_log_generator_serves_the_previous_restart(tmp_path: Path):
    """``previous`` selects the second highest restart_count, not the file
    written second."""
    serve_dir = _write_serve_logs(tmp_path)
    (serve_dir / "1.0.log").write_text("older-main\n", encoding="utf-8")
    (serve_dir / "1.1.log").write_text("current-main\n", encoding="utf-8")
    (serve_dir / "1.container.0.log").write_text("older-container\n", encoding="utf-8")

    output = await _collect(
        combined_log_generator(
            serve_dir,
            1,
            str(tmp_path / "absent.log"),
            LogOptions(previous=True),
            "inst",
        )
    )

    assert output == "older-main\nolder-container\n"


@pytest.mark.asyncio
async def test_combined_log_generator_streams_only_the_named_sidecar(tmp_path: Path):
    """Asking for a sidecar by name skips the download and main logs entirely,
    and must not pick up the workload container's own log."""
    serve_dir = _write_serve_logs(tmp_path)
    (serve_dir / "1.0.log").write_text("main\n", encoding="utf-8")
    (serve_dir / "1.container.0.log").write_text(
        "default-container\n", encoding="utf-8"
    )
    (serve_dir / "1.container.ray-head.0.log").write_text("ray\n", encoding="utf-8")

    output = await _collect(
        combined_log_generator(
            serve_dir,
            1,
            str(tmp_path / "absent.log"),
            LogOptions(),
            "inst",
            container_name="ray-head",
        )
    )

    assert output == "ray\n"


@pytest.mark.asyncio
async def test_combined_log_generator_reports_not_found_on_an_empty_directory(
    tmp_path: Path,
):
    """No file of any kind is a 404, not an empty stream: "nothing was logged"
    has to be distinguishable from "this instance is unknown here"."""
    serve_dir = _write_serve_logs(tmp_path)

    with pytest.raises(NotFoundException):
        await _collect(
            combined_log_generator(
                serve_dir, 1, str(tmp_path / "absent.log"), LogOptions(), "inst"
            )
        )


@pytest.mark.parametrize(
    "relative_path, expected",
    [
        # The layout a current worker writes: identity in the directories, so
        # the file name only has to say which stream it is.
        (
            "qwen3-6-35b-sn20w.42/7/main.log",
            ServeLogName(42, 7, ServeLogKind.MAIN, instance_name="qwen3-6-35b-sn20w"),
        ),
        (
            "qwen3-6-35b-sn20w.42/7/container.log",
            ServeLogName(
                42, 7, ServeLogKind.CONTAINER, instance_name="qwen3-6-35b-sn20w"
            ),
        ),
        (
            "qwen3-6-35b-sn20w.42/7/container.ray-head.log",
            ServeLogName(
                42,
                7,
                ServeLogKind.SIDECAR,
                container_name="ray-head",
                instance_name="qwen3-6-35b-sn20w",
            ),
        ),
        # An instance whose name sanitizes to nothing keeps a bare id.
        ("42/7/main.log", ServeLogName(42, 7, ServeLogKind.MAIN)),
        # A size-capped log's other pieces carry the head's identity. Their
        # suffix sits after ".log", on the other side of the name from a
        # sidecar's container name -- which is what keeps the next two apart.
        (
            "qwen3-6-35b-sn20w.42/7/main.log.3",
            ServeLogName(
                42,
                7,
                ServeLogKind.MAIN,
                instance_name="qwen3-6-35b-sn20w",
                segment=ServeLogSegment.TAIL,
                shard=3,
            ),
        ),
        (
            "qwen3-6-35b-sn20w.42/7/container.3.log",
            ServeLogName(
                42,
                7,
                ServeLogKind.SIDECAR,
                container_name="3",
                instance_name="qwen3-6-35b-sn20w",
            ),
        ),
        (
            "qwen3-6-35b-sn20w.42/7/container.ray-head.log.truncated",
            ServeLogName(
                42,
                7,
                ServeLogKind.SIDECAR,
                container_name="ray-head",
                instance_name="qwen3-6-35b-sn20w",
                segment=ServeLogSegment.MARKER,
            ),
        ),
        # Anything else after ".log" is not a segment.
        ("qwen3-6-35b-sn20w.42/7/main.log.tmp", None),
        # The three flat namings earlier releases wrote, still read so they can
        # be migrated.
        ("42.log", ServeLogName(42, 0, ServeLogKind.MAIN, flat=True, legacy=True)),
        ("42.7.log", ServeLogName(42, 7, ServeLogKind.MAIN, flat=True)),
        ("42.container.7.log", ServeLogName(42, 7, ServeLogKind.CONTAINER, flat=True)),
        (
            "42.container.ray-head.7.log",
            ServeLogName(
                42, 7, ServeLogKind.SIDECAR, container_name="ray-head", flat=True
            ),
        ),
        # An all-digit container name is unambiguous once both ends are anchored:
        # a sidecar carries two segments after "container", never one.
        (
            "42.container.2.7.log",
            ServeLogName(42, 7, ServeLogKind.SIDECAR, container_name="2", flat=True),
        ),
        # Names that are not serve logs at all, and must stay unreadable rather
        # than degrade to restart 0 -- retention deletes whatever sits outside
        # the kept window, and restart 0 is outside it from restart 2 onwards.
        ("model_file_42.download.log", None),
        ("benchmark.log", None),
        ("42.log.tmp", None),
        # An unanchored pattern accepts trailing junk; this one must not.
        ("42.7.logXYZ", None),
        ("42.container.7.logXYZ", None),
        # Inside a restart directory the parser is anchored to the known names,
        # so anything else sharing the directory is not read as a log.
        ("qwen3-6-35b-sn20w.42/7/unknown.json", None),
        # A directory that is not an instance's, or a level that is not a
        # restart, says nothing about any instance.
        ("not-an-instance/7/main.log", None),
        ("qwen3-6-35b-sn20w.42/latest/main.log", None),
    ],
)
def test_serve_log_layout(relative_path, expected):
    assert parse_serve_log_path(Path("/serve") / relative_path) == expected


# A budget small enough to rotate within a test: a 20-byte head, 20-byte
# shards and room for four of them.
_TINY_CAP = {"max_bytes": 100, "head_bytes": 20}


def _numbered_lines(start: int, stop: int) -> list:
    return [f"line{i:03d}\n" for i in range(start, stop)]  # 8 bytes each


def _omitted_bytes(head: Path) -> int:
    return int(re.search(r'\.\.\. (\d+) bytes', marker_log_path(head).read_text())[1])


def _restart_dir(tmp_path: Path) -> Path:
    directory = restart_log_dir(tmp_path, "qwen", 42, 7)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.mark.parametrize(
    "build_head, kind",
    [
        (lambda d: main_log_path(d, "qwen", 42, 7), ServeLogKind.MAIN),
        (lambda d: container_log_path(d, "qwen", 42, 7), ServeLogKind.CONTAINER),
        (
            lambda d: sidecar_container_log_path(d, "qwen", 42, "ray-head", 7),
            ServeLogKind.SIDECAR,
        ),
    ],
    ids=["main", "container", "sidecar"],
)
def test_a_capped_log_freezes_its_head_and_rotates_its_tail(
    build_head, kind, tmp_path: Path
):
    """All three kinds are budgeted the same way: the earliest output is kept
    verbatim, the newest keeps arriving, the disk stays inside the budget."""
    _restart_dir(tmp_path)
    head = build_head(tmp_path)

    with CappedLogWriter(head, **_TINY_CAP) as writer:
        writer.writelines(_numbered_lines(0, 40))

    shards = tail_shard_log_paths(head)
    assert head.read_text() == "line000\nline001\nline002\n"
    assert shards and shards[-1].read_text().endswith("line039\n")
    assert sum(p.stat().st_size for p in [head, *shards]) <= 100 + 20
    assert {parse_serve_log_path(p).kind for p in [head, *shards]} == {kind}


@pytest.mark.asyncio
async def test_a_capped_log_reads_back_as_head_then_marker_then_tail(tmp_path: Path):
    """Discovery hands the reader the parts in writing order, the marker
    between them saying how much went missing."""
    serve_dir = tmp_path / "serve"
    head = main_log_path(serve_dir, "qwen", 42, 7)
    head.parent.mkdir(parents=True)

    with CappedLogWriter(head, **_TINY_CAP) as writer:
        writer.writelines(_numbered_lines(0, 40))

    files = await get_all_log_files(serve_dir, 42)
    lines = "".join(f.read_text() for f in files).splitlines()

    markers = [line for line in lines if "omitted here" in line]
    assert len(markers) == 1
    assert "216 bytes / 27 lines omitted" in markers[0]
    marked = lines.index(markers[0])
    assert lines[:marked] == ["line000", "line001", "line002"]
    assert lines[marked + 1] == "line030" and lines[-1] == "line039"
    # 3 kept at the front + 27 dropped + 10 kept at the back is the whole log.
    assert len(lines[:marked]) + 27 + len(lines[marked + 1 :]) == 40


def test_an_uncapped_log_is_written_exactly_as_before(tmp_path: Path):
    """A budget of 0 leaves no trace of the mechanism: one file, byte for byte
    what the plain line-buffered open it stands in for would write."""
    written = ["a\n", "b" * 5000 + "\n", "no newline at the end"]
    plain = tmp_path / "plain.log"
    with open(plain, "w", buffering=1, encoding="utf-8") as f:
        f.writelines(written)

    head = _restart_dir(tmp_path) / "main.log"
    with CappedLogWriter(head, max_bytes=0) as writer:
        writer.writelines(written)

    assert head.read_bytes() == plain.read_bytes()
    assert [p.name for p in head.parent.iterdir()] == ["main.log"]


def test_reopening_a_capped_log_continues_where_it_stopped(tmp_path: Path):
    """A worker restart adopts the log: the frozen head stays frozen, new
    output lands in the shard being written, and the dropped-bytes tally keeps
    counting up from where the previous process left it."""
    head = _restart_dir(tmp_path) / "main.log"
    with CappedLogWriter(head, **_TINY_CAP) as writer:
        writer.writelines(_numbered_lines(0, 40))
    newest = newest_segment_log_path(head)
    omitted = _omitted_bytes(head)

    with CappedLogWriter(head, append=True, **_TINY_CAP) as writer:
        writer.write("after\n")
    assert newest.read_text().endswith("line039\nafter\n")
    # What a resume reads back has to come from here too: the head's last line
    # stopped being the log's last line long ago.
    assert newest_segment_log_path(head).read_text().endswith("line039\nafter\n")

    with CappedLogWriter(head, append=True, **_TINY_CAP) as writer:
        writer.writelines(_numbered_lines(40, 80))
    assert head.read_text() == "line000\nline001\nline002\n"
    assert _omitted_bytes(head) > omitted

    # Every byte ever written is either still on disk or counted as dropped.
    # A tally that restarted at zero on reopening would lose the difference,
    # and the marker would understate the hole it describes.
    written = sum(
        len(line)
        for line in _numbered_lines(0, 40) + ["after\n"] + _numbered_lines(40, 80)
    )
    kept = head.stat().st_size + sum(
        p.stat().st_size for p in tail_shard_log_paths(head)
    )
    assert kept + _omitted_bytes(head) == written


def test_a_rewritten_log_does_not_begin_in_the_middle(tmp_path: Path):
    """Starting a log over retires the previous one's parts too: a fresh head
    beside an older run's tail reads as a log missing its first page."""
    head = _restart_dir(tmp_path) / "main.log"
    with CappedLogWriter(head, **_TINY_CAP) as writer:
        writer.writelines(_numbered_lines(0, 40))

    with CappedLogWriter(head, **_TINY_CAP) as writer:
        writer.write("fresh\n")

    assert [p.name for p in head.parent.iterdir()] == ["main.log"]
    assert head.read_text() == "fresh\n"


@pytest.mark.parametrize("head_bytes, head_text", [(0, ""), (1, "line000\n")])
def test_a_small_head_does_not_cut_the_tail_into_a_file_per_write(
    head_bytes, head_text, tmp_path: Path
):
    """A head of 0 keeps none. Shards sized by so small a head would each take
    one write, so the tail is cut by the budget instead: never more than 64."""
    head = _restart_dir(tmp_path) / "main.log"
    with CappedLogWriter(head, max_bytes=6400, head_bytes=head_bytes) as writer:
        writer.writelines(_numbered_lines(0, 1000))

    shards = tail_shard_log_paths(head)
    assert head.read_text() == head_text
    assert 1 < len(shards) <= 64
    assert shards[-1].read_text().endswith("line999\n")
    # Each segment rolls on the write after it fills, so may hold one more line.
    assert sum(p.stat().st_size for p in [head, *shards]) <= 6400 + 65 * 8


def test_a_capped_log_stands_in_for_the_streams_it_replaces(tmp_path: Path):
    """It becomes sys.stdout and sys.stderr of a serving process: any thread
    may print while another's write rotates the file, and some libraries write
    bytes to .buffer. Every byte is still kept or counted, and a character
    split across two byte writes arrives whole."""
    head = _restart_dir(tmp_path) / "main.log"
    batches = [_numbered_lines(i * 250, (i + 1) * 250) for i in range(8)]
    encoded = "加载完成\n".encode()
    errors = []

    def emit(writer, batch):
        try:
            for line in batch:
                writer.write(line)
        except Exception as e:
            errors.append(e)

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with CappedLogWriter(head, max_bytes=2000, head_bytes=100) as writer:
            assert isinstance(writer, io.TextIOBase)
            writer.buffer.write(encoded[:2])
            writer.buffer.write(encoded[2:])
            threads = [
                threading.Thread(target=emit, args=(writer, batch)) for batch in batches
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        sys.setswitchinterval(switch_interval)

    assert errors == []
    assert head.read_text(encoding="utf-8").startswith("加载完成\n")
    written = len(encoded) + sum(len(line) for batch in batches for line in batch)
    kept = head.stat().st_size + sum(
        p.stat().st_size for p in tail_shard_log_paths(head)
    )
    assert kept + _omitted_bytes(head) == written


@pytest.mark.asyncio
async def test_following_a_capped_log_crosses_a_rotation(tmp_path: Path):
    """A follower pinned to one shard goes quiet exactly when the log is
    busiest. Every line has to arrive, once, in order, across the rotations."""
    head = _restart_dir(tmp_path) / "main.log"
    # Room for nine shards, so the rotations under test drop nothing.
    writer = CappedLogWriter(head, max_bytes=200, head_bytes=20)
    writer.writelines(_numbered_lines(0, 5))

    stop_event = asyncio.Event()
    received = []

    async def follow():
        options = LogOptions(tail=-1, follow=True, stop_event=stop_event)
        async for line in log_generator(str(head), options):
            received.append(line)

    async def until(count: int):
        for _ in range(100):
            if len(received) >= count:
                return
            await asyncio.sleep(0.05)

    task = asyncio.create_task(follow())
    try:
        await until(5)
        writer.writelines(_numbered_lines(5, 20))
        await until(20)
        # Asserted while the stream is still open: a follower that only catches
        # up once the log is closed is not following it.
        assert received == _numbered_lines(0, 20)
    finally:
        stop_event.set()
        writer.close()
        await task


@pytest.mark.asyncio
async def test_a_source_is_followed_only_where_it_is_still_being_written(
    tmp_path: Path,
):
    """A group is read in order, so following a file that is already finished
    never ends and starves the rest of the group."""
    older = tmp_path / "1.log"
    older.write_text("old0\nold1\n", encoding="utf-8")
    newer = tmp_path / "1.0.log"
    newer.write_text("new0\n", encoding="utf-8")

    stop_event = asyncio.Event()
    received = []

    async def read():
        options = LogOptions(tail=-1, follow=True, stop_event=stop_event)
        group = [str(older), str(newer)]
        async for line in merged_log_generator([group], options, stop_event):
            received.append(line)

    task = asyncio.create_task(read())
    try:
        for _ in range(60):
            if len(received) >= 3:
                break
            await asyncio.sleep(0.05)
        assert received == ["old0\n", "old1\n", "new0\n"]
    finally:
        stop_event.set()
        await task


def test_a_sidecar_named_with_digits_is_not_read_as_another_instance():
    """`1.container.2.3.log` ends in `.2.3.log`; a pattern anchored only at the
    tail reads it as instance 2's main log, crossing instances."""
    path = Path("/serve/1.container.2.3.log")
    parsed = parse_serve_log_path(path)

    assert parsed.model_instance_id == 1
    assert parsed.kind is ServeLogKind.SIDECAR
    assert extract_restart_count(path) is None


@pytest.mark.asyncio
async def test_log_discovery_spans_both_layouts_and_rejects_id_prefixes(
    tmp_path: Path,
):
    """Both layouts are found together, and neither 12945 nor 29450 leaks in:
    their ids contain 2945's digits, so only the parsed id decides."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "2945.log",
        "2945.1.log",
        "2945.container.1.log",
        "2945.container.ray-head.1.log",
        "12945.2.log",
        "29450.2.log",
        "12945.container.2.log",
    )
    _write_restart_logs(
        serve_dir, "qwen3-6-35b", 2945, 2, "main.log", "container.log", SIDECAR_LOG
    )
    _write_restart_logs(serve_dir, "other", 12945, 2, "main.log")

    main_logs = await get_all_log_files(serve_dir, 2945, container=False)
    container_logs = await get_all_log_files(serve_dir, 2945, container=True)
    sidecar_logs = await get_all_log_files(
        serve_dir, 2945, container=True, container_name="ray-head"
    )

    assert [str(f.relative_to(serve_dir)) for f in main_logs] == [
        "2945.log",
        "2945.1.log",
        "qwen3-6-35b.2945/2/main.log",
    ]
    assert [str(f.relative_to(serve_dir)) for f in container_logs] == [
        "2945.container.1.log",
        "qwen3-6-35b.2945/2/container.log",
    ]
    assert [str(f.relative_to(serve_dir)) for f in sidecar_logs] == [
        "2945.container.ray-head.1.log",
        f"qwen3-6-35b.2945/2/{SIDECAR_LOG}",
    ]
    assert await resolve_restart_count(serve_dir, 2945, previous=False) == 2
    assert await resolve_restart_count(serve_dir, 2945, previous=True) == 1


@pytest.mark.parametrize(
    "instance_name, expected",
    [
        ("qwen3-0-6b-ab12x", "qwen3-0-6b-ab12x"),
        # A model name may hold a dot; leaving it in would add a segment and
        # make the file name impossible to read back.
        ("qwen3.6-35b-a3b-fp8-3-sn20w", "qwen3-6-35b-a3b-fp8-3-sn20w"),
        ("has space", "has-space"),
        ("has/slash", "has-slash"),
        ("中文模型-ab12x", "-----ab12x"),
        # No length limit of its own, so the file name imposes one.
        ("z" * 300, "z" * 96),
    ],
)
def test_instance_name_sanitizing(instance_name, expected):
    sanitized = sanitize_instance_name(instance_name)

    assert sanitized == expected
    assert "." not in sanitized
    assert len(sanitized) <= 96


def test_a_sidecar_name_cannot_lead_out_of_the_restart_directory(tmp_path: Path):
    """The runtime names a sidecar, so its name is not this process's to trust:
    interpolated raw, one holding a separator would place the log elsewhere."""
    restart_dir = restart_log_dir(tmp_path, "qwen", 1, 0)

    path = sidecar_container_log_path(tmp_path, "qwen", 1, "../../etc/ray", 0)

    assert path.parent == restart_dir
    assert path.name == "container.------etc-ray.log"


def test_a_name_with_nothing_usable_falls_back_to_a_bare_id(tmp_path: Path):
    """An empty name would leave a leading dot the grammar rejects, so such an
    instance is filed under its id alone."""
    assert instance_log_dir(tmp_path, "...", 42).name == "---.42"

    path = main_log_path(tmp_path, "", 42, 7)
    assert path.parent.parent.name == "42"
    assert parse_serve_log_path(path) is not None


def test_write_side_files_logs_under_a_directory_named_for_the_instance(
    tmp_path: Path,
):
    """The point of #5858: `ls` on the serve directory alone says which
    deployment is which, one line each however many logs it has."""
    serve_dir = tmp_path / "serve"
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3.6-35b-sn20w", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 5

    main = Path(manager._get_numbered_log_path(model_instance))
    container = container_log_path(serve_dir, model_instance.name, 1, 5)
    sidecar = sidecar_container_log_path(serve_dir, model_instance.name, 1, "ray", 5)

    for path, expected in (
        (main, "qwen3-6-35b-sn20w.1/5/main.log"),
        (container, "qwen3-6-35b-sn20w.1/5/container.log"),
        (sidecar, "qwen3-6-35b-sn20w.1/5/container.ray.log"),
    ):
        assert str(path.relative_to(serve_dir)) == expected
        assert parse_serve_log_path(path).model_instance_id == 1


def test_adoption_migrates_flat_logs_into_the_instance_directory(tmp_path: Path):
    """Adoption is where an earlier release's files catch up with the layout.
    Another instance's files, and names the grammar cannot read, stay put."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "1.4.log",
        "1.container.4.log",
        "1.container.ray-head.4.log",
        "2.4.log",
        "1.log.tmp",
    )
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 5

    with patch.object(manager, "_start_container_log_persistence"):
        manager._ensure_container_log_persistence(model_instance)

    assert sorted(p.name for p in serve_dir.iterdir()) == [
        "1.log.tmp",
        "2.4.log",
        "qwen3-0-6b.1",
    ]
    moved = restart_log_dir(serve_dir, "qwen3-0.6b", 1, 4)
    assert sorted(p.name for p in moved.iterdir()) == [
        "container.log",
        SIDECAR_LOG,
        "main.log",
    ]


def test_adoption_renames_the_directory_when_the_instance_is_renamed(tmp_path: Path):
    """One rename moves every restart with it, so history does not split
    across the name the instance used to have -- including when a pre-v2.2.0
    log is filed under the new name on the same pass."""
    serve_dir = _write_serve_logs(tmp_path, "1.log")
    _write_restart_logs(serve_dir, "old-name", 1, 4, "main.log")
    _write_restart_logs(serve_dir, "old-name", 1, 5, "main.log")
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "new-name", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 6

    with patch.object(manager, "_start_container_log_persistence"):
        manager._ensure_container_log_persistence(model_instance)

    assert [p.name for p in serve_dir.iterdir()] == ["new-name.1"]
    assert sorted(p.name for p in (serve_dir / "new-name.1").iterdir()) == [
        "4",
        "5",
        "6",
    ]


def test_adoption_leaves_a_move_target_that_already_exists(tmp_path: Path):
    """Moving onto an existing file would destroy it, so the flat one stays
    where it is; the reader finds both."""
    serve_dir = _write_serve_logs(tmp_path, "1.4.log")
    _write_restart_logs(serve_dir, "qwen3-0-6b", 1, 4, "main.log")
    target = restart_log_dir(serve_dir, "qwen3-0-6b", 1, 4) / "main.log"
    target.write_text("newer", encoding="utf-8")
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 5

    with patch.object(manager, "_start_container_log_persistence"):
        manager._ensure_container_log_persistence(model_instance)

    assert sorted(p.name for p in serve_dir.iterdir()) == ["1.4.log", "qwen3-0-6b.1"]
    assert target.read_text(encoding="utf-8") == "newer"


@pytest.mark.parametrize("restart_count", [0, 3], ids=["purge", "retention"])
def test_a_log_directory_that_cannot_be_listed_does_not_stop_a_start(
    restart_count, tmp_path: Path
):
    """Cleanup runs ahead of every start. A listing that fails -- a directory
    removed underneath it, a stale NFS handle -- costs an error line, not the
    start."""
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path, "1.2.log"))

    with patch(
        "gpustack.worker.serve_manager.flat_instance_logs",
        side_effect=OSError("stale file handle"),
    ):
        manager._cleanup_old_logs(1, restart_count)


def test_cleanup_leaves_unreadable_and_other_instances_names_alone(tmp_path: Path):
    """Retention only removes what it can place in a restart; another
    instance, a download log or an unreadable name is not ours to delete."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "2945.0.log",
        "2945.1.log",
        "12945.0.log",
        "model_file_2945.download.log",
        "2945.log.tmp",
    )
    _write_restart_logs(serve_dir, "qwen3-6-35b", 2945, 0, "main.log")
    _write_restart_logs(serve_dir, "qwen3-6-35b", 2945, 2, "main.log")
    _write_restart_logs(serve_dir, "other", 12945, 0, "main.log")
    # A directory that names no restart cannot be outside the window either.
    (serve_dir / "qwen3-6-35b.2945" / "scratch").mkdir()

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(2945, 2)

    assert sorted(p.name for p in serve_dir.iterdir()) == [
        "12945.0.log",
        "2945.1.log",
        "2945.log.tmp",
        "model_file_2945.download.log",
        "other.12945",
        "qwen3-6-35b.2945",
    ]
    assert sorted(p.name for p in (serve_dir / "qwen3-6-35b.2945").iterdir()) == [
        "2",
        "scratch",
    ]


def test_purge_leaves_unreadable_and_other_instances_names_alone(tmp_path: Path):
    """The fresh-start purge takes everything for the id, in either layout, and
    nothing else."""
    serve_dir = _write_serve_logs(
        tmp_path,
        "2945.log",
        "2945.0.log",
        "2945.container.0.log",
        "12945.0.log",
        "model_file_2945.download.log",
        "2945.log.tmp",
    )
    _write_restart_logs(serve_dir, "qwen3-6-35b", 2945, 0, "main.log", SIDECAR_LOG)
    _write_restart_logs(serve_dir, "other", 12945, 0, "main.log")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(2945, 0)

    assert sorted(p.name for p in serve_dir.iterdir()) == [
        "12945.0.log",
        "2945.log.tmp",
        "model_file_2945.download.log",
        "other.12945",
    ]


def test_cleanup_old_logs_keeps_only_current_and_previous_restart(tmp_path: Path):
    """Keep main/container logs for R and R-1; delete older restart_count files.
    A pre-v2.2.0 {id}.log counts as restart 0, so it goes with the rest."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid = 42
    for name in (
        f"{mid}.log",
        f"{mid}.0.log",
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 2)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [
        f"{mid}.1.log",
        f"{mid}.2.log",
        f"{mid}.container.1.log",
        f"{mid}.container.2.log",
    ]


def test_cleanup_old_logs_restart_zero_purges_all(tmp_path: Path):
    """Fresh start (restart_count 0) removes every log for the id, incl. sidecar
    and {id}.log, but leaves other instances' and model-file download logs."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    mid, other = 7, 8
    for name in (
        f"{mid}.log",
        f"{mid}.0.log",
        f"{mid}.container.1.log",
        f"{mid}.container.ray-head.0.log",
        f"{other}.log",
        f"{other}.0.log",
        f"model_file_{mid}.download.log",
    ):
        (serve_dir / name).write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(mid, 0)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == [
        f"{other}.0.log",
        f"{other}.log",
        f"model_file_{mid}.download.log",
    ]


def test_cleanup_old_logs_keeps_legacy_main_log_as_previous_restart(tmp_path: Path):
    """At R==1 the kept window is {1, 0}, so a legacy {id}.log survives as the
    previous restart alongside the numbered logs."""
    serve_dir = _write_serve_logs(tmp_path, "1.log", "1.1.log", "1.container.1.log")

    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    manager._cleanup_old_logs(1, 1)

    remaining = sorted(p.name for p in serve_dir.iterdir())
    assert remaining == ["1.1.log", "1.container.1.log", "1.log"]


def test_permanent_teardown_purges_logs_but_restart_keeps_them(tmp_path: Path):
    """A permanent teardown (delete_logs=True, used by the reap) removes the
    instance's serve logs so a reused id can't inherit them; a restart (default
    stop) keeps them for the log viewer."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True)
    log = serve_dir / "1.container.ray-head.0.log"
    log.write_text("x", encoding="utf-8")

    manager, _clients = _build_serve_manager(worker_id=1)
    manager._serve_log_dir = str(serve_dir)
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch.object(manager, "_stop_container_log_persistence"),
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        # Restart-style stop keeps the logs.
        manager._stop_model_instance(model_instance)
        assert log.exists()

        # Permanent teardown removes them.
        manager._stop_model_instance(model_instance, delete_logs=True)
        assert not log.exists()


def test_reap_stale_instance_purges_logs(tmp_path: Path):
    """Reaping an instance the server no longer reports (a dropped DELETED) must
    also remove its serve logs, {id}.log included, mirroring the DELETED
    handler."""
    serve_dir = _write_serve_logs(tmp_path, "1.container.ray-head.0.log", "1.log")

    manager, clientset = _build_serve_manager(worker_id=1)
    manager._serve_log_dir = str(serve_dir)
    stale = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[stale.id] = stale
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch("gpustack.worker.serve_manager.delete_workload"),
        patch.object(manager, "_stop_container_log_persistence"),
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    assert sorted(p.name for p in serve_dir.iterdir()) == []


def test_reap_confirmation_skips_when_authoritative_fetch_still_has_instance():
    """A cache read momentarily missing an instance (e.g. mid-reconnect, before
    the watch replay repopulates) must not reap it: the authoritative
    confirmation fetch still lists it, so it is a false positive and the live
    workload is left alone."""
    manager, clientset = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[mi.id] = mi
    # First (cache) read misses it -> reap candidate; the authoritative
    # confirmation fetch still has it -> not stale.
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[mi]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.INITIALIZING),
        ),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_not_called()
    # Cache read plus exactly one authoritative confirmation fetch.
    assert clientset.model_instances.list.call_count == 2
    confirm_call = clientset.model_instances.list.call_args_list[1]
    assert confirm_call.kwargs.get("params") == {"page": -1}
    assert confirm_call.kwargs.get("use_cache") is False


def test_reap_confirmation_reaps_when_missing_from_authoritative_fetch():
    """When an instance is absent from both the cache read and the authoritative
    confirmation fetch, it is genuinely gone and gets reaped."""
    manager, clientset = _build_serve_manager(worker_id=1)
    stale = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[stale.id] = stale
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_called_once_with(stale, delete_logs=True)
    # Cache read plus the authoritative confirmation fetch before reaping.
    assert clientset.model_instances.list.call_count == 2


def test_ghost_in_cache_not_reconciled_when_backstop_disabled():
    """With the periodic reconciliation disabled (default), an instance present
    in both local state and the watch cache but gone from DB is not a
    `local - cache` reap candidate, so the sync trusts the cache and takes no
    DB round trip."""
    manager, clientset = _build_serve_manager(worker_id=1)
    ghost = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[ghost.id] = ghost
    clientset.model_instances.list.return_value = SimpleNamespace(items=[ghost])

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch(
            "gpustack.worker.serve_manager.envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL",
            0,
        ),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state=WorkloadStatusStateEnum.INITIALIZING),
        ),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_not_called()
    # Cache read only; no authoritative DB fetch when the backstop is off.
    assert clientset.model_instances.list.call_count == 1


def test_ghost_in_cache_reaped_when_backstop_interval_elapsed():
    """When the reconciliation backstop is enabled and its interval has elapsed,
    the forced authoritative DB read reaps a ghost living in both local state
    and the cache — the case `local - cache` can never flag on its own."""
    manager, clientset = _build_serve_manager(worker_id=1)
    ghost = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    manager._model_instance_by_instance_id[ghost.id] = ghost
    # Last reconciliation is far enough in the past to be due.
    manager._last_state_reconcile_time = 0
    # Cache still lists the ghost; the authoritative DB fetch does not.
    clientset.model_instances.list.side_effect = [
        SimpleNamespace(items=[ghost]),
        SimpleNamespace(items=[]),
    ]

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch(
            "gpustack.worker.serve_manager.envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL",
            60,
        ),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
        patch.object(manager, "_is_provisioning", return_value=False),
    ):
        manager.sync_model_instances_state()

    stop_model_instance.assert_called_once_with(ghost, delete_logs=True)
    # Cache read plus the forced authoritative fetch.
    assert clientset.model_instances.list.call_count == 2
    confirm_call = clientset.model_instances.list.call_args_list[1]
    assert confirm_call.kwargs.get("params") == {"page": -1}
    assert confirm_call.kwargs.get("use_cache") is False


def test_delete_event_defers_teardown_to_reap():
    """The DELETED handler no longer tears down directly; teardown is left to the
    periodic reap so there is a single teardown path (no reap-vs-event race)."""
    manager, _ = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        4, "qwen3-0.6b-kqkco", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    manager._model_instance_by_instance_id[mi.id] = mi

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(manager, "_stop_model_instance") as stop_model_instance,
    ):
        manager._handle_model_instance_event(Event(type=EventType.DELETED, data=mi))

    stop_model_instance.assert_not_called()


def test_updated_event_error_does_not_crash_watch():
    """A failure on any event path (not just DELETED) must be swallowed so it
    never escapes the awatch callback. Here an UPDATED->restart raises."""
    manager, _ = _build_serve_manager(worker_id=1)
    mi = new_model_instance(
        5, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.SCHEDULED
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"

    with (
        patch("gpustack.worker.serve_manager.logger"),
        patch.object(
            manager,
            "_restart_model_instance",
            side_effect=RuntimeError("start failed"),
        ),
    ):
        # Must not raise.
        manager._handle_model_instance_event(Event(type=EventType.UPDATED, data=mi))


# Stream timestamps are written as offsets from here. They have to sit in the
# present: a copier with no cursor resumes at the current second, and lines
# stamped in 1970 would silently be older than that.
_STREAM_EPOCH = int(time.time())


@pytest.fixture(autouse=True)
def _stream_epoch_of_this_test():
    """Rebase the stamped streams on each test's own start.

    A cursor with nothing recorded behind it is invented from the wall clock,
    so streams pinned to import time drift from it by however long the suite
    took to reach this file -- and past 90s the drift flips which branch runs.
    """
    global _STREAM_EPOCH
    _STREAM_EPOCH = int(time.time())


def _moment(offset: int) -> str:
    """One RFC3339Nano stamp, the shape the runtime puts in front of a line."""
    stamp = datetime.fromtimestamp(_STREAM_EPOCH + offset, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


def _stamped(*lines) -> list:
    """A stream of chunks: one (offset, text) pair becomes one prefixed chunk."""
    return [f"{_moment(offset)} {text}" for offset, text in lines]


def _cursor_of(log_path) -> tuple:
    record = Path(f"{log_path}.cursor").read_text(encoding="utf-8").split()
    return int(record[0]) - _STREAM_EPOCH, int(record[1])


def test_a_reconnect_resumes_at_the_cursor_and_drops_the_replay(tmp_path: Path):
    """The runtime replays from the cursor's whole second, so lines of that
    second the archive already holds have to be counted off -- including ones
    repeated verbatim, which no content match could tell apart."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    asked = []

    streams = [
        _stamped((10, "a\n"), (11, "dup\n"), (11, "dup\n")),
        # The whole of second 11 comes back, plus what followed it.
        _stamped((11, "dup\n"), (11, "dup\n"), (11, "late\n"), (12, "b\n")),
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    def fake_logs_workload(**kwargs):
        asked.append((kwargs["since"], kwargs["timestamps"]))
        return iter(streams.pop(0))

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert asked == [(None, True), (_STREAM_EPOCH + 11, True)]
    assert Path(log_path).read_text(encoding="utf-8") == "a\ndup\ndup\nlate\nb\n"
    assert _cursor_of(log_path) == (12, 1)


def test_a_runtime_that_stamps_nothing_does_not_pin_the_cursor(tmp_path: Path):
    """Without timestamps no line moves the cursor. Left where it was, every
    reconnect would ask for the same second and the archive would take the
    same stretch again each time; it moves to the end of each connection."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    now = [1000.0]
    asked = []
    streams = [["a\n", "b\n"], ["c\n"], ["d\n"]]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    def fake_logs_workload(**kwargs):
        asked.append(kwargs["since"])
        now[0] += 10
        return iter(streams.pop(0))

    with (
        patch("gpustack.worker.serve_manager.time.time", lambda: now[0]),
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert asked == [None, 1010, 1020]


def test_output_still_short_of_a_newline_reaches_the_archive(tmp_path: Path):
    """A progress bar redraws in place for the whole of a weight load without
    ever ending a line. Held back until the newline, the log viewer shows
    nothing for exactly as long -- the phase a starting instance is watched."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    midway = []

    def one_stream():
        yield from _stamped((10, "loading  0%\r"), (10, "loading 50%\r"))
        # Read as bytes: universal newlines would rewrite the bare '\r'.
        midway.append(Path(log_path).read_bytes().decode("utf-8"))
        yield from _stamped((10, "loading done\n"))

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            return_value=one_stream(),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(
                [SimpleNamespace(state=WorkloadStatusStateEnum.FAILED)]
            ),
        ),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert midway == ["loading  0%\rloading 50%\r"]
    assert (
        Path(log_path).read_bytes().decode("utf-8")
        == "loading  0%\rloading 50%\rloading done\n"
    )
    assert _cursor_of(log_path) == (10, 1)


def test_one_uninterrupted_stream_neither_reconnects_nor_marks(tmp_path: Path):
    """A runtime-side rotation does not end a followed stream, so nothing about
    it reaches the copier: one connection, no gap marker."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    # A cursor into an archive retention has already taken points nowhere; it
    # must not send this connection looking for a second that never existed.
    Path(f"{log_path}.cursor").write_text(
        f"{_STREAM_EPOCH + 10:020d} {3:012d}\n", encoding="utf-8"
    )
    connections = []

    def fake_logs_workload(**kwargs):
        connections.append(kwargs["since"])
        return iter(_stamped(*[(10 + i, f"l{i}\n") for i in range(6)]))

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event(), resume=True)

    assert connections == [None]
    written = Path(log_path).read_text(encoding="utf-8")
    assert written == "".join(f"l{i}\n" for i in range(6))
    assert "may be missing" not in written


def test_persist_container_logs_exits_when_workload_gone(tmp_path: Path):
    """EOF while the workload no longer exists -> exit immediately, no reconnect."""
    manager, _clients = _build_serve_manager()
    log_path = str(tmp_path / "1.container.0.log")
    tails = []

    def fake_logs_workload(**kwargs):
        tails.append(kwargs["tail"])
        return iter(_stamped((10, "a\n")))

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs("wl", log_path, _fake_stop_event())

    assert tails == [-1]  # only one connection, no reconnect
    assert Path(log_path).read_text(encoding="utf-8") == "a\n"


def test_a_line_the_stream_cut_short_is_completed_by_the_reconnect(tmp_path: Path):
    """A connection can end in the middle of a line. That half is on disk but
    not behind the cursor, so the reconnect replaces it with the whole line
    instead of counting it off and dropping the rest."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"

    streams = [
        _stamped((10, "a\n"), (10, "half-lin")),
        _stamped((10, "a\n"), (10, "half-line-whole\n"), (11, "b\n")),
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: iter(streams.pop(0)),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", str(log_path), _fake_stop_event())

    assert log_path.read_text(encoding="utf-8") == "a\nhalf-line-whole\nb\n"


@pytest.mark.parametrize(
    "reconnects, resume",
    [
        # The runtime no longer has the cursor's second: nothing to relocate.
        ([_stamped((90, "far-later\n"))], False),
        # Nothing came back at all.
        ([[], _stamped((11, "more\n"))], False),
        # A worker restart with no cursor beside the archive, as an upgrade
        # from a release that did not write one leaves it.
        ([_stamped((90, "far-later\n"))], True),
    ],
    ids=["unrelocatable", "empty-reconnect", "no-cursor"],
)
def test_an_archive_with_content_is_never_truncated(tmp_path: Path, reconnects, resume):
    """The invariant the whole feature exists for: whatever a reconnect does
    with the runtime's replay, it may only ever add to what is on disk -- and
    where it cannot prove the two join up, it says so rather than start over."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    history = "".join(f"kept-{i}\n" for i in range(5))
    log_path.write_text(history, encoding="utf-8")

    streams = [iter(s) for s in reconnects]
    states = [SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING)] * len(streams)
    states.append(SimpleNamespace(state=WorkloadStatusStateEnum.FAILED))

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs(
            "wl", str(log_path), _fake_stop_event(), resume=resume
        )

    written = log_path.read_text(encoding="utf-8")
    assert written.startswith(history)
    # None of these can prove what follows continues what is already there.
    assert "may be missing" in written


def test_a_cursor_invented_from_the_clock_always_marks_the_seam(tmp_path: Path):
    """An archive from before the cursor existed resumes at "now", so every
    line a replay skips is one the archive never held: skipping proves no seam,
    and the downtime it hides is what the marker exists to announce."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    log_path.write_text("before-the-restart\n", encoding="utf-8")

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            return_value=iter(
                _stamped((-3600, "long-before\n"), (-1800, "also-before\n"))
            ),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(
                [SimpleNamespace(state=WorkloadStatusStateEnum.FAILED)]
            ),
        ),
    ):
        manager._persist_container_logs(
            "wl", str(log_path), _fake_stop_event(), resume=True
        )

    written = log_path.read_text(encoding="utf-8")
    assert written.startswith("before-the-restart\n")
    assert "may be missing" in written


def test_a_reconnect_that_cannot_relocate_appends_behind_a_marker(tmp_path: Path):
    """The runtime has dropped the cursor's second, so what follows does not
    continue what is on disk. The archive says so and keeps both halves."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"

    streams = [
        _stamped((10, "early\n")),
        _stamped((90, "much-later\n")),
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=lambda **kwargs: iter(streams.pop(0)),
        ),
        patch(
            "gpustack.worker.serve_manager.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        manager._persist_container_logs("wl", str(log_path), _fake_stop_event())

    written = log_path.read_text(encoding="utf-8").splitlines()
    assert written[0] == "early\n".strip()
    assert "may be missing" in written[1] and _moment(10)[:19] in written[1]
    assert written[2] == "much-later"


def test_a_worker_restart_resumes_from_the_cursor_on_disk(tmp_path: Path):
    """The restart path reads the same cursor a reconnect keeps in memory, and
    reads it beside the newest shard once the size cap has rotated the log."""
    manager, _clients = _build_serve_manager()
    head = container_log_path(tmp_path, "qwen", 1, 0)
    head.parent.mkdir(parents=True)
    history = [(10, f"l{i:02d}\n") for i in range(18)] + [(11, "l18\n")]
    replay = _stamped(*(history + [(11, "l19\n"), (12, "l20\n")]))

    with (
        patch("gpustack.worker.log_sources.envs.SERVE_LOG_MAX_BYTES", 1000),
        patch("gpustack.worker.log_sources.envs.SERVE_LOG_HEAD_BYTES", 20),
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            side_effect=[iter(_stamped(*history)), iter(replay)],
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs("wl", str(head), _fake_stop_event())
        # A second thread, as a restarted worker starts one, over the same
        # archive and the cursor the first one left behind.
        manager._persist_container_logs(
            "wl", str(head), _fake_stop_event(), resume=True
        )

    written = [head, *tail_shard_log_paths(head)]
    assert "".join(p.read_text() for p in written) == "".join(
        text for _epoch, text in history + [(11, "l19\n"), (12, "l20\n")]
    )


@pytest.mark.parametrize(
    "adopted_tail, replayed_tail, written_lines",
    [
        # A progress bar is one streamed line carrying bare '\r'; the cursor
        # counted it, and it must reach the archive as the runtime framed it.
        (
            "shards:  0%\rshards: 50%\rshards: 100%\n",
            "shards:  0%\rshards: 50%\rshards: 100%\n",
            8,
        ),
        # A worker killed mid-write leaves a fragment the cursor never counted;
        # the runtime replays that line whole, so the fragment has to go.
        ("INFO star", "INFO starting engine\n", 7),
    ],
    ids=["progress-bar", "fragment"],
)
def test_a_resumed_archive_keeps_the_runtime_line_framing(
    tmp_path: Path, adopted_tail, replayed_tail, written_lines
):
    """Resuming trims whatever the previous process left half-written, and the
    replay's own framing -- bare '\r' included -- reaches disk untouched."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    head = "".join(f"l{i}\n" for i in range(1, 8))
    log_path.write_text(head + adopted_tail, encoding="utf-8")
    Path(f"{log_path}.cursor").write_text(
        f"{_STREAM_EPOCH + 10:020d} {written_lines:012d}\n", encoding="utf-8"
    )

    replay = _stamped(
        *[(10, f"l{i}\n") for i in range(1, 8)], (10, replayed_tail), (11, "l8-NEW\n")
    )

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            return_value=iter(replay),
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs(
            "wl", str(log_path), _fake_stop_event(), resume=True
        )

    # Read as bytes: universal newlines would rewrite the bare '\r' under test.
    written = log_path.read_bytes().decode("utf-8")
    assert written == head + replayed_tail + "l8-NEW\n"


def test_a_line_split_across_chunks_reaches_the_archive_whole(tmp_path: Path):
    """A line past 16 KiB arrives in several chunks, each carrying the same
    prefix. Every piece lands as it arrives, none of them leaves a timestamp
    inside the line, and the line is whole once the last one has."""
    manager, _clients = _build_serve_manager()
    log_path = tmp_path / "1.container.0.log"
    long_line = "x" * 40000
    chunks = _stamped(
        (10, "first\n"),
        (10, long_line[:16384]),
        (10, long_line[16384:32768]),
        (10, long_line[32768:] + "\n"),
        # A stream cut mid-line still lands what it has, at the very end.
        (11, "tail-with-no-newline"),
    )
    midway = []

    def stream():
        for index, chunk in enumerate(chunks):
            if index == 3:
                midway.append(log_path.read_text(encoding="utf-8"))
            yield chunk

    with (
        patch(
            "gpustack.worker.serve_manager.logs_workload",
            return_value=stream(),
        ),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
    ):
        manager._persist_container_logs("wl", str(log_path), _fake_stop_event())

    assert midway == [f"first\n{long_line[:32768]}"]
    assert log_path.read_text(encoding="utf-8") == (
        f"first\n{long_line}\ntail-with-no-newline"
    )


def test_adoption_reattaches_container_log_persistence(tmp_path: Path):
    """A worker restart kills the log persistence threads while the workload
    keeps running, and the replayed CREATED event returns early for an
    already-running instance. The periodic sync has to re-attach them, without
    restarting the instance and without disturbing a thread still alive."""
    manager, clientset = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))

    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    def sync():
        with (
            patch(
                "gpustack.worker.serve_manager.get_workload",
                return_value=SimpleNamespace(state="running"),
            ),
            patch.object(manager, "_is_provisioning", return_value=False),
            patch.object(manager, "_get_model", return_value=model),
            patch.object(manager, "_start_container_log_persistence") as start_logs,
            patch.object(manager, "_start_model_instance") as start_instance,
        ):
            manager.sync_model_instances_state()
        return start_logs, start_instance

    # A fresh worker process has no threads registered: re-attach, don't restart.
    start_logs, start_instance = sync()
    start_logs.assert_called_once_with(model_instance, resume=True)
    start_instance.assert_not_called()

    # A live main log thread is left alone: two would interleave their appends.
    manager._log_persistence[1] = _log_persistence(main_alive=True)
    start_logs, _ = sync()
    start_logs.assert_not_called()

    # A dead main thread beside the forever-polling discovery thread re-attaches;
    # folding the two together would never fire here.
    manager._log_persistence[1] = _log_persistence(main_alive=False)
    start_logs, _ = sync()
    start_logs.assert_called_once_with(model_instance, resume=True)


def test_stop_container_log_persistence_tears_down_the_whole_generation():
    """The main thread is tracked apart from the sidecar ones, but stopping still
    has to signal and join every one of them."""
    manager, _clients = _build_serve_manager()
    main_stop_event, sidecar_stop_event = MagicMock(), MagicMock()
    main_log_thread, sidecar_thread = _fake_thread(True), _fake_thread(True)
    persistence = _LogPersistence(main_stop_event, main_log_thread)
    persistence.add_aux_thread(sidecar_thread, sidecar_stop_event)
    manager._log_persistence[1] = persistence

    manager._stop_container_log_persistence(1)

    main_stop_event.set.assert_called_once()
    sidecar_stop_event.set.assert_called_once()
    main_log_thread.join.assert_called_once_with(timeout=2.0)
    sidecar_thread.join.assert_called_once_with(timeout=2.0)
    assert manager._log_persistence == {}

    # A sidecar discovered after teardown began is signalled, not orphaned.
    late_stop_event = MagicMock()
    persistence.add_aux_thread(_fake_thread(True), late_stop_event)
    late_stop_event.set.assert_called_once()


def test_starting_log_persistence_retires_the_previous_generation(tmp_path: Path):
    """Starts arrive from both the watch thread and the periodic sync thread. Were
    the registry swap and the teardown it replaces not one critical section, the
    loser's threads would keep running with nothing able to signal them."""
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))
    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )

    retire = manager._retire_log_persistence
    held_while_retiring = []

    def probe_lock(model_instance_id, timeout=2.0):
        # A non-reentrant lock refuses its own holder, so failing to take it here
        # proves the caller is already inside the critical section.
        acquired = manager._log_persistence_lock.acquire(blocking=False)
        if acquired:
            manager._log_persistence_lock.release()
        held_while_retiring.append(not acquired)
        return retire(model_instance_id, timeout)

    with (
        patch("gpustack.worker.serve_manager.logs_workload", return_value=iter([])),
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(manager, "_retire_log_persistence", side_effect=probe_lock),
    ):
        manager._start_container_log_persistence(model_instance)
        first = manager._log_persistence[1]
        manager._start_container_log_persistence(model_instance)
        second = manager._log_persistence[1]

        assert held_while_retiring == [True, True]
        assert first is not second
        assert first.stop_event.is_set()
        assert not second.stop_event.is_set()

        manager._stop_container_log_persistence(1)

    assert second.stop_event.is_set()
    assert manager._log_persistence == {}


def test_sync_retires_log_persistence_the_server_no_longer_assigns(tmp_path: Path):
    """A DELETED event landing mid-pass leaves a generation nothing will ever
    stop, and the persistence loop retries a missing workload forever rather than
    winding down, so the reconciler has to retire it."""
    manager, clientset = _build_serve_manager()
    manager._serve_log_dir = str(_write_serve_logs(tmp_path))

    kept = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    clientset.model_instances.list.return_value = SimpleNamespace(items=[kept])
    manager._log_persistence[1] = _log_persistence(main_alive=True)
    manager._log_persistence[99] = _log_persistence(main_alive=True)
    orphan_stop_event = manager._log_persistence[99].stop_event

    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        # Provisioning: still assigned, so its generation must be left alone.
        patch.object(manager, "_is_provisioning", return_value=True),
        patch.object(manager, "_get_model", return_value=model),
    ):
        manager.sync_model_instances_state()

    assert set(manager._log_persistence) == {1}
    orphan_stop_event.set.assert_called_once()


def test_adoption_aligns_legacy_main_log_with_restart_count(tmp_path: Path):
    """A legacy main log counts as restart 0, so without the rename the viewer
    would file it under a different restart than the container log."""
    serve_dir = _write_serve_logs(tmp_path, "1.log")
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(serve_dir)

    model_instance = new_model_instance(
        1, "qwen3-0.6b", 1, worker_id=1, state=ModelInstanceStateEnum.RUNNING
    )
    model_instance.restart_count = 5

    with patch.object(manager, "_start_container_log_persistence") as start_logs:
        manager._ensure_container_log_persistence(model_instance)
        # A later adoption with the target already in place must not overwrite it.
        (serve_dir / "1.log").write_text("newer owner", encoding="utf-8")
        manager._ensure_container_log_persistence(model_instance)

    assert start_logs.call_count == 2
    start_logs.assert_called_with(model_instance, resume=True)
    # The instance name holds a dot, which sanitizing turns into a dash so that
    # the directory name keeps exactly two segments.
    assert sorted(p.name for p in serve_dir.iterdir()) == ["1.log", "qwen3-0-6b.1"]
    moved = restart_log_dir(serve_dir, "qwen3-0.6b", 1, 5) / "main.log"
    assert moved.read_text(encoding="utf-8") == "x"


# --- vGPU allocation read-back (gpu_type_selector) ---

from gpustack.schemas.models import (  # noqa: E402
    ComputedResourceClaim,
    GPUTypeSelector,
)
from gpustack.worker.serve_manager import (  # noqa: E402
    _ALLOCATED_ACCELERATORS_ANNOTATION,
    _parse_allocated_accelerators,
)

_VGPU_ANNOTATION_VALUE = (
    '{"run-0": {"devices": {"groups": [{"id": "a100", "manufacturer": "nvidia",'
    ' "accelerators": [{"id": "GPU-uuid-1", "index": 1, "mode": 3,'
    ' "allocated": 640000}]}]}, "deviceIDs": ["a100:GPU-uuid-1:0124"]}}'
)


def test_parse_allocated_accelerators():
    accelerators = _parse_allocated_accelerators(
        {_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE}
    )
    assert [a["id"] for a in accelerators] == ["GPU-uuid-1"]
    assert accelerators[0]["index"] == 1


def test_parse_allocated_accelerators_tolerates_skew():
    assert _parse_allocated_accelerators(None) == []
    assert _parse_allocated_accelerators({}) == []
    assert (
        _parse_allocated_accelerators({_ALLOCATED_ACCELERATORS_ANNOTATION: "not json"})
        == []
    )
    assert (
        _parse_allocated_accelerators({_ALLOCATED_ACCELERATORS_ANNOTATION: '["x"]'})
        == []
    )
    assert (
        _parse_allocated_accelerators(
            {_ALLOCATED_ACCELERATORS_ANNOTATION: '{"c": "unexpected"}'}
        )
        == []
    )


def _build_vgpu_manager(worker_id=1, device_index=1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.workers.get.return_value = SimpleNamespace(
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(uuid="GPU-uuid-1", index=device_index)]
        )
    )
    cfg = SimpleNamespace(log_dir="/tmp")
    manager = ServeManager(lambda: worker_id, lambda: clientset, cfg)
    manager._inference_backend_manager = MagicMock()
    return manager, clientset


def _vgpu_model():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
    model.backend = BackendEnum.VLLM
    model.backend_version = "0.8.0"
    model.gpu_type_selector = GPUTypeSelector(
        type="pool-a100",
        accelerator_sliced_memory_percentage=50,
        accelerator_sliced_cores_percentage=50,
    )
    return model


def test_sync_vgpu_allocation_patches_main_addresses_and_rekeys_claim():
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once()
    _, kwargs = update_model_instance.call_args
    assert kwargs["gpu_addresses"] == ["GPU-uuid-1"]
    # The allocated card's index is backfilled for display and accounting.
    assert kwargs["gpu_indexes"] == [1]
    # Placeholder key 0 re-keyed to the allocated card index 1.
    assert kwargs["computed_resource_claim"].vram == {1: 42949672960}


def test_sync_vgpu_allocation_patches_subordinate_worker():
    manager, clientset = _build_vgpu_manager(worker_id=2, device_index=3)

    sw = ModelInstanceSubordinateWorker(
        worker_id=2,
        worker_name="worker-2",
        worker_ip="10.0.0.2",
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    model_instance = new_model_instance(
        1,
        "vgpu-distributed",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model_instance.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.RUN_FIRST,
        subordinate_workers=[sw],
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once()
    args, _ = update_model_instance.call_args
    assert args[0] == model_instance.id
    patched_sw = update_model_instance.call_args.kwargs[
        "distributed_servers.subordinate_workers.0"
    ]
    assert patched_sw.gpu_addresses == ["GPU-uuid-1"]
    assert patched_sw.gpu_indexes == [3]
    assert patched_sw.computed_resource_claim.vram == {3: 42949672960}


def test_sync_vgpu_allocation_noop_without_annotation():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(state="Running", annotations={})

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_not_called()


def test_error_state_surfaces_workload_state_message():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Failed",
        state_message="Allocate failed due to no enough sliced units",
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Allocate failed due to no enough sliced units",
    )


def test_error_state_falls_back_to_generic_message():
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="Failed"),
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Inference server exited or unhealthy.",
    )


def _workload_exit(name="default", exit_code=None, reason=""):
    return SimpleNamespace(name=name, exit_code=exit_code, reason=reason)


def test_error_state_surfaces_the_container_exit_code():
    """gpustack/gpustack#4217: the exit code must reach the instance, not just
    the runtime's workload status."""
    manager, clientset = _build_vgpu_manager(worker_id=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Failed",
        state_message="Error",
        exits=[_workload_exit(exit_code=7, reason="Error")],
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    update_model_instance.assert_called_once_with(
        model_instance.id,
        state=ModelInstanceStateEnum.ERROR,
        state_message="Error (exit code 7)",
    )


def test_a_generation_that_will_not_stop_holds_off_the_next_one(tmp_path: Path):
    """Two writers on one log truncate each other's shards, so a generation
    still holding the files keeps the next one out -- and keeps its place in
    the registry while it does, because that record is all the sync after it
    has to recognise that there is still something to wait for."""
    manager, _clients = _build_serve_manager()
    manager._serve_log_dir = str(tmp_path / "serve")
    blocked_in_the_runtime = SimpleNamespace(
        name="log-persist-qwen", is_alive=lambda: True, join=lambda timeout: None
    )
    stuck = _LogPersistence(threading.Event(), blocked_in_the_runtime)
    manager._log_persistence[1] = stuck

    with patch.object(threading.Thread, "start") as started:
        manager._start_container_log_persistence(new_model_instance(1, "qwen", 1))
        manager._start_container_log_persistence(new_model_instance(1, "qwen", 1))

    assert started.call_count == 0
    assert manager._log_persistence[1] is stuck


@pytest.mark.asyncio
async def test_listing_the_restarts_does_not_walk_once_per_stream(tmp_path: Path):
    """Every stream of every retained restart is measured on the request path.
    A walk per stream multiplies the listing's cost by restarts times
    containers, so the listing walks the instance's logs once, however many
    there are."""

    async def walks(serve_dir: Path, restarts, names) -> int:
        for restart_count in restarts:
            _write_restart_logs(serve_dir, "qwen", 1, restart_count, *names)
        config = SimpleNamespace(log_dir=str(serve_dir.parent))
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(config=config))
        )
        # Counted at both names: the route holds its own reference to the walk.
        counted = MagicMock(side_effect=instance_log_files)
        with (
            patch("gpustack.routes.worker.logs.instance_log_files", counted),
            patch("gpustack.worker.log_sources.instance_log_files", counted),
        ):
            await get_serve_log_options(request, 1)
        return counted.call_count

    one = await walks(tmp_path / "one" / "serve", [2], ["main.log", "container.log"])
    many = await walks(
        tmp_path / "many" / "serve",
        [2, 3, 4],
        ["main.log", "container.log", "container.ray-head.log"],
    )

    assert (one, many) == (1, 1)


def test_workload_failure_appends_the_exit_code():
    """Both backends must answer gpustack/gpustack#4217, and they arrive at it
    differently: Docker names the reason on state_message, Kubernetes leaves it
    empty and the exits are the only source."""
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="OOMKilled",
                exits=[_workload_exit(exit_code=137, reason="OOMKilled")],
            )
        )
        == "OOMKilled (exit code 137)"
    )
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="",
                exits=[_workload_exit(exit_code=1, reason="Error")],
            )
        )
        == "Error (exit code 1)"
    )


def test_workload_failure_keeps_the_image_pull_diagnosis():
    """gpustack/gpustack#5869: a container blocked on its image pull never
    terminated, so it has no exit code, and its state message already carries
    the registry error the Pod's Events explain it with. Nothing may displace
    or pad it."""
    message = (
        'ImagePullBackOff: Back-off pulling image "registry.invalid/nope:latest"; '
        "Failed to pull image: not found"
    )

    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message=message,
                exits=[_workload_exit(reason="ImagePullBackOff")],
            )
        )
        == message
    )


def test_workload_failure_names_each_container_when_several_exit():
    assert (
        _describe_workload_failure(
            SimpleNamespace(
                state_message="",
                exits=[
                    _workload_exit(exit_code=1, reason="Error"),
                    _workload_exit(name="ray-head", exit_code=137, reason="OOMKilled"),
                ],
            )
        )
        == "Error, OOMKilled (default exit code 1, ray-head exit code 137)"
    )


def test_workload_failure_falls_back_when_the_workload_explains_nothing():
    # A workload reaped out from under the sync, and one that failed without a
    # message or any exit entry (the pre-0.2.3 shape, which has no `exits` at
    # all) both land on the generic message.
    assert _describe_workload_failure(None) == "Inference server exited or unhealthy."
    assert (
        _describe_workload_failure(SimpleNamespace(state="Failed"))
        == "Inference server exited or unhealthy."
    )
    assert (
        _describe_workload_failure(
            SimpleNamespace(state_message="", exits=[_workload_exit(exit_code=2)])
        )
        == "Inference server exited or unhealthy. (exit code 2)"
    )


def test_sync_vgpu_allocation_defers_when_card_not_in_reported_devices():
    """M6: an allocated UUID missing from the worker's reported devices must
    not leave the claim charged to a wrong placeholder index — defer instead
    of patching."""
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)
    # The reported device carries a different UUID than the annotation's.
    clientset.workers.get.return_value = SimpleNamespace(
        status=SimpleNamespace(gpu_devices=[SimpleNamespace(uuid="GPU-other", index=1)])
    )

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={0: 42949672960}),
    )
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance") as update_model_instance,
    ):
        manager.sync_model_instances_state()

    for call in update_model_instance.call_args_list:
        _, kwargs = call
        assert "gpu_addresses" not in kwargs
        assert "gpu_indexes" not in kwargs
        assert "computed_resource_claim" not in kwargs


def test_sync_vgpu_allocation_steady_state_skips_worker_fetch():
    """M7: once addresses, indexes and the re-keyed claim agree with the
    annotation, the sync must not call the workers API again."""
    manager, clientset = _build_vgpu_manager(worker_id=1, device_index=1)

    model_instance = new_model_instance(
        1,
        "vgpu-instance",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        computed_resource_claim=ComputedResourceClaim(vram={1: 42949672960}),
    )
    model_instance.gpu_addresses = ["GPU-uuid-1"]
    model_instance.gpu_indexes = [1]
    clientset.model_instances.list.return_value = SimpleNamespace(
        items=[model_instance]
    )

    workload = SimpleNamespace(
        state="Running",
        annotations={_ALLOCATED_ACCELERATORS_ANNOTATION: _VGPU_ANNOTATION_VALUE},
    )

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=workload),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=_vgpu_model()),
        patch.object(manager, "_update_model_instance"),
    ):
        manager.sync_model_instances_state()

    clientset.workers.get.assert_not_called()
