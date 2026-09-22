import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse

from gpustack_runtime.deployer import logs_workload

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.schemas.cache_services import cache_service_instance_workload_name
from gpustack.schemas.models import (
    ModelInstanceLogRestartEntry,
    ServeLogOptionsResponse,
)
from gpustack.worker.logs import (
    LogOptions,
    LogOptionsDep,
    line_window_generator,
    log_generator,
    plan_line_window,
    tail_log_generator,
)
from gpustack.worker.log_sources import (
    ContainerLogSource,
    DownloadLogSource,
    MainLogSource,
    extract_restart_count,
    get_all_log_files,
    group_container_names_by_restart,
    instance_log_files,
    monitor_container_content,
    resolve_restart_count,
    select_log_files,
    stream_log_files,
)

router = APIRouter()

logger = logging.getLogger(__name__)


def _path_started_at_utc(path: Path) -> datetime:
    st = path.stat()
    ts = getattr(st, "st_birthtime", None)
    if ts is None or ts <= 0:
        ts = st.st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


async def serve_log_paths(
    log_dir: Path,
    model_instance_id: int,
    restart_count: Optional[int],
    container_name: Optional[str] = None,
) -> List[Path]:
    """One restart's logs as a single ordered list of files.

    The order a line number is counted against: the restart's main logs, then
    its container logs. The follow path merges sources concurrently, which is
    what a live tail wants and what a page number cannot have -- the same
    offset has to name the same line on every request.

    Args:
        log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.
        restart_count: The restart to read, None for every one of them.
        container_name: A sidecar's name to read only its logs; "default" or
            None for the main and workload container logs.

    Returns:
        The files, in reading order.
    """
    entries = await asyncio.to_thread(instance_log_files, log_dir, model_instance_id)
    return stream_log_files(entries, restart_count, container_name)


def restart_entries_from_main_log_files(
    files: List[Path],
    sidecar_names_by_restart: Optional[Dict[int, List[str]]] = None,
) -> List[ModelInstanceLogRestartEntry]:
    """Build restart entries from main log paths; one entry per restart_count.

    Entries are ordered by restart_count descending (newest first).
    The highest restart_count maps to ``previous=False`` (current);
    the second highest maps to ``previous=True``.

    When several files share a restart_count -- a pre-v2.2.0 {id}.log next to a
    migrated main.log -- take the earliest timestamp, not a representative by
    name.

    Args:
        files: Main log file paths.
        sidecar_names_by_restart: Mapping from restart_count to sidecar
            container names found on disk for that restart.
    """
    by_count: Dict[int, List[Path]] = defaultdict(list)
    for f in files:
        by_count[extract_restart_count(f)].append(f)

    sorted_counts = sorted(by_count.keys(), reverse=True)
    entries: List[ModelInstanceLogRestartEntry] = []
    for i, rc in enumerate(sorted_counts):
        timestamps = []
        for path in by_count[rc]:
            try:
                timestamps.append(_path_started_at_utc(path))
            except OSError:
                continue
        started_at = min(timestamps, default=None)
        containers = (
            sidecar_names_by_restart.get(rc, []) if sidecar_names_by_restart else []
        )
        entries.append(
            ModelInstanceLogRestartEntry(
                previous=i > 0,
                started_at=started_at,
                containers=containers,
            )
        )
    return entries


async def historical_log_generator(
    log_dir: Path,
    model_instance_id: int,
    options: LogOptions,
    stop_event: Optional[asyncio.Event] = None,
    container: bool = False,
    restart_count: Optional[int] = None,
    container_name: Optional[str] = None,
):
    """Generate logs from historical log files.

    Args:
        log_dir: Directory containing log files
        model_instance_id: Model instance ID
        options: Log options (tail, follow)
        stop_event: Event to signal stopping
        container: If True, read container logs; if False, read main logs
        restart_count: Resolved restart count to filter log files
        container_name: If specified with container=True, read sidecar container logs.

    Yields:
        Log lines from log files
    """
    log_files = await get_all_log_files(
        log_dir,
        model_instance_id,
        container=container,
        restart_count=restart_count,
        container_name=container_name,
    )

    if not log_files:
        if container:
            logger.debug(
                f"No container log files found for model instance "
                f"{model_instance_id}"
            )
        return

    if options.tail > 0:
        tail_options = LogOptions(
            tail=options.tail, follow=options.follow, stop_event=stop_event
        )
        async for line in tail_log_generator(log_files, tail_options):
            if stop_event and stop_event.is_set():
                logger.debug("Historical log generator stopping due to stop event 1")
                return
            yield line
    else:
        # Read all logs in order
        for i, log_file in enumerate(log_files):
            # For all files except the last one, don't follow
            is_last_file = i == len(log_files) - 1
            file_options = LogOptions(
                tail=-1,
                follow=options.follow if is_last_file else False,
                stop_event=stop_event,
            )
            async for line in log_generator(str(log_file), file_options):
                if stop_event and stop_event.is_set():
                    logger.debug(
                        "Historical log generator stopping due to stop event 2"
                    )
                    return
                yield line


async def _group_log_generator(paths: List[str], options: LogOptions):
    """Every line of one source's files in order, following only the last."""
    for index, log_path in enumerate(paths):
        is_last = index == len(paths) - 1
        file_options = LogOptions(
            tail=-1, follow=options.follow and is_last, stop_event=options.stop_event
        )
        async for line in log_generator(log_path, file_options):
            yield line


async def merged_log_generator(  # noqa: C901
    log_path_groups: List[List[str]],
    options: LogOptions,
    stop_event: Optional[asyncio.Event] = None,
):
    """Merge multiple log sources and yield lines as they become available.

    Args:
        log_path_groups: One group of file paths per source, in reading order.
            A group is read start to finish and only its last file is followed,
            since a source's earlier files are complete -- a capped log's
            shards are one such group. Groups themselves are read concurrently.
        options: Log options (tail, follow)
        stop_event: Event to signal stopping

    Yields:
        Log lines from all sources in the order they become available
    """
    if not log_path_groups:
        return

    queues: List[asyncio.Queue] = []

    async def read_to_queue(queue: asyncio.Queue, paths: List[str], opts: LogOptions):
        try:
            if opts.tail > 0:
                lines = tail_log_generator([Path(p) for p in paths], opts)
            else:
                lines = _group_log_generator(paths, opts)
            async for line in lines:
                if stop_event and stop_event.is_set():
                    return
                await queue.put(("data", line))
        except Exception as e:
            logger.error(f"Error reading logs {paths}: {e}")
            await queue.put(("error", str(e)))
        finally:
            await queue.put(None)  # Signal end of this source

    # Create tasks for all log generators
    tasks = []
    for paths in log_path_groups:
        queue = asyncio.Queue()
        queues.append(queue)
        task = asyncio.create_task(read_to_queue(queue, paths, options))
        tasks.append(task)

    get_tasks = {}
    for q in queues:
        task = asyncio.create_task(q.get())
        get_tasks[task] = q

    # Yield lines as they become available from any source
    active_count = len(queues)
    try:
        while active_count > 0:
            if stop_event and stop_event.is_set():
                break

            # Wait for any queue to have data
            # (with timeout to check stop_event periodically)
            done, _ = await asyncio.wait(
                get_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=0.5,
            )

            # Check stop_event after timeout
            if stop_event and stop_event.is_set():
                break

            for future in done:
                queue = get_tasks.pop(future)
                try:
                    result = future.result()
                except asyncio.CancelledError:
                    continue
                if result is None:
                    active_count -= 1
                else:
                    msg_type, content = result
                    if msg_type == "data":
                        yield content
                    # error type is logged in read_to_queue, continue streaming
                    # Only recreate the task for the completed queue
                    new_task = asyncio.create_task(queue.get())
                    get_tasks[new_task] = queue
    finally:
        # Cancel all background tasks to prevent leaks when the generator
        # is closed early (e.g. client disconnect).
        all_tasks = list(tasks) + list(get_tasks.keys())
        for t in all_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


async def combined_log_generator(
    log_dir: Path | str,
    model_instance_id: int,
    download_log_path: str,
    options: LogOptionsDep,
    model_instance_name: str,
    container_name: Optional[str] = None,
):
    """Unified log streaming from three file sources.

    Reads logs in order:
    1) Download logs (if exists)
    2) Historical main logs (all restart_count files)
    3) Container logs (from persisted files)

    When container_name is specified, only sidecar container logs are streamed
    (download and main logs are skipped).

    Args:
        log_dir: Directory containing log files (Path or str)
        model_instance_id: Model instance ID
        download_log_path: Path to download log file
        options: Log options (tail, follow)
        model_instance_name: Model instance name (unused, kept for API compatibility)
        container_name: If specified, stream only this sidecar container's logs.
    """
    log_dir = Path(log_dir)

    restart_count = await resolve_restart_count(
        log_dir, model_instance_id, options.previous
    )

    # When a specific sidecar container is requested, stream only its logs.
    # "default" means the main container — fall through to the normal path.
    if container_name and container_name != "default":
        async for line in historical_log_generator(
            log_dir,
            model_instance_id,
            options,
            container=True,
            restart_count=restart_count,
            container_name=container_name,
        ):
            yield line
        return

    download_source = DownloadLogSource(download_log_path)
    main_source = MainLogSource(log_dir, model_instance_id, restart_count)
    container_source = ContainerLogSource(log_dir, model_instance_id, restart_count)

    has_any_logs = False
    log_path_groups = []

    # Download log
    download_files = await download_source.wait_for_files_if_needed(
        follow=options.follow
    )
    if download_files:
        log_path_groups.append([str(download_files[0])])
        has_any_logs = True

    # Main logs
    main_log_files = await main_source.wait_for_files_if_needed(follow=options.follow)
    if main_log_files:
        log_path_groups.append([str(f) for f in main_log_files])
        has_any_logs = True

    # Stream download + main logs (merged)
    if log_path_groups:
        stop_event = asyncio.Event()
        monitor_task = None

        container_has_content = await container_source.has_content()
        if not container_has_content and options.follow:
            monitor_task = asyncio.create_task(
                monitor_container_content(container_source, stop_event)
            )

        merge_options = (
            LogOptions(tail=-1, follow=False)
            if container_has_content or not options.follow
            else options
        )

        try:
            async for line in merged_log_generator(
                log_path_groups, merge_options, stop_event
            ):
                yield line
        finally:
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    # Container logs
    container_log_files = await container_source.wait_for_files_if_needed(
        follow=options.follow,
        main_log_files=main_log_files,
    )
    if container_log_files:
        has_any_logs = True
        async for line in historical_log_generator(
            log_dir,
            model_instance_id,
            options,
            container=True,
            restart_count=restart_count,
        ):
            yield line

    if not has_any_logs:
        raise NotFoundException(message="Log file not found")


@router.get("/serveLogOptions/{id}", response_model=ServeLogOptionsResponse)
async def get_serve_log_options(request: Request, id: int):
    """List restart_count values for which main serve log files exist locally."""
    log_dir = request.app.state.config.log_dir
    serve_log_dir = Path(log_dir) / "serve"
    # One walk for both: the restarts and the streams each one has.
    entries = await asyncio.to_thread(instance_log_files, serve_log_dir, id)
    files = select_log_files(entries)
    container_names_by_restart = group_container_names_by_restart(entries)

    restarts = await asyncio.to_thread(
        restart_entries_from_main_log_files, files, container_names_by_restart
    )

    return ServeLogOptionsResponse(restarts=restarts)


@router.get("/serveLogs/{id}")
async def get_serve_logs(
    request: Request,
    id: int,
    log_options: LogOptionsDep,
    model_instance_name: str = Query(default=""),
    model_file_id: Optional[int] = Query(default=None),
    container_name: Optional[str] = Query(default=None),
):
    log_dir = request.app.state.config.log_dir
    serve_log_dir = Path(log_dir) / "serve"

    download_log_path = ""
    # Use model file ID for shared download logs if provided
    if model_file_id is not None:
        download_log_path = str(
            serve_log_dir / f"model_file_{model_file_id}.download.log"
        )

    if log_options.offset is not None:
        return await serve_log_line_range(
            serve_log_dir, id, download_log_path, log_options, container_name
        )

    return StreamingResponse(
        combined_log_generator(
            serve_log_dir,
            id,
            download_log_path,
            log_options,
            model_instance_name,
            container_name=container_name,
        ),
        media_type="application/octet-stream",
    )


async def serve_log_line_range(
    serve_log_dir: Path,
    model_instance_id: int,
    download_log_path: str,
    options: LogOptions,
    container_name: Optional[str] = None,
) -> StreamingResponse:
    """Serve one page of a restart's log, addressed by line number.

    Args:
        serve_log_dir: Directory containing serve logs.
        model_instance_id: Model instance ID.
        download_log_path: Path to the download log, empty when there is none.
        options: Log options carrying offset and limit.
        container_name: A sidecar's name, or "default"/None for the workload.

    Returns:
        The page as plain text, with the range and the stream's total on
        X-Log-Offset, X-Log-Line-Count and X-Log-Total-Lines, and for a range
        covering the whole stream, while the cap has dropped none of it, its
        exact length on X-Log-Total-Bytes.

    Raises:
        BadRequestException: tail or follow was asked for alongside a range.
    """
    if options.follow or options.tail > 0:
        raise BadRequestException(
            message="offset cannot be combined with tail or follow"
        )

    restart_count = await resolve_restart_count(
        serve_log_dir, model_instance_id, options.previous
    )
    paths = await serve_log_paths(
        serve_log_dir, model_instance_id, restart_count, container_name
    )
    if download_log_path and (not container_name or container_name == "default"):
        download = Path(download_log_path)
        if await asyncio.to_thread(download.exists):
            paths.insert(0, download)

    window = await asyncio.to_thread(
        plan_line_window, paths, options.offset, options.limit
    )
    headers = {
        "X-Log-Offset": str(window.offset),
        "X-Log-Line-Count": str(window.line_count),
        "X-Log-Total-Lines": str(window.total_lines),
    }
    # Measured on the same look at the files the reads are pinned to, so a
    # download can promise it.
    if window.byte_count is not None:
        headers["X-Log-Total-Bytes"] = str(window.byte_count)
    return StreamingResponse(
        line_window_generator(window),
        media_type="application/octet-stream",
        headers=headers,
    )


@router.get("/cacheServiceInstanceLogs/{instance_id}")
async def get_cache_service_instance_logs(
    instance_id: int,
    log_options: LogOptionsDep,
    cache_service_id: int = Query(),
):
    """Stream a managed cache service instance's container logs.

    Logs are read live from the container runtime rather than from
    persisted files, so the ``previous`` option has no effect here.
    """
    workload_name = cache_service_instance_workload_name(cache_service_id, instance_id)

    def iter_logs():
        try:
            logs = logs_workload(
                name=workload_name,
                tail=log_options.tail,
                follow=log_options.follow,
            )
        except Exception as e:
            yield f"Failed to fetch cache service logs: {e}\n"
            return
        if isinstance(logs, (bytes, str)):
            yield logs
            return
        yield from logs

    return StreamingResponse(iter_logs(), media_type="application/octet-stream")


@router.get("/benchmark_logs/{id}")
async def get_benchmark_logs(
    request: Request,
    id: int,
    log_options: LogOptionsDep,
    benchmark_name: str = Query(default=""),
):
    log_dir = request.app.state.config.log_dir
    benchmark_log_path = Path(log_dir) / "benchmarks" / f"{id}.log"

    return StreamingResponse(
        log_generator(str(benchmark_log_path), log_options),
        media_type="application/octet-stream",
    )
