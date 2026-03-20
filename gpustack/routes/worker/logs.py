import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptions, LogOptionsDep, log_generator
from gpustack.worker.log_sources import (
    ContainerLogSource,
    DownloadLogSource,
    LogSourceChain,
    MainLogSource,
)
from gpustack.server.deps import SessionDep

router = APIRouter()

logger = logging.getLogger(__name__)


def extract_restart_count(filename: str) -> int:
    """Extract restart count from filename like '123.5.log'.

    Args:
        filename: Log filename in format {id}.{restart_count}.log

    Returns:
        Restart count as integer, or 0 if pattern doesn't match
    """
    match = re.match(r'\d+\.(\d+)\.log', filename)
    return int(match.group(1)) if match else 0


def extract_container_restart_count(filename: str) -> int:
    """Extract restart count from container log filename.

    Args:
        filename: Log filename in format {id}.container.{restart_count}.log

    Returns:
        Restart count as integer, or 0 if pattern doesn't match
    """
    match = re.match(r'\d+\.container\.(\d+)\.log', filename)
    return int(match.group(1)) if match else 0


async def get_all_log_files(
    log_dir: Path,
    model_instance_id: int,
    container: bool = False,
    restart_count: Optional[int] = None,
) -> List[Path]:
    """Get all log files sorted by restart count.

    Args:
        log_dir: Directory containing log files
        model_instance_id: Model instance ID
        container: If True, get container logs; if False, get main logs
        restart_count: If specified, only return logs for this restart count

    Returns:
        List of log file paths sorted by restart count (oldest first)
    """
    if container:
        pattern = f"{model_instance_id}.container.*.log"
        extract_fn = extract_container_restart_count
    else:
        pattern = f"{model_instance_id}.*.log"
        extract_fn = extract_restart_count

    files = await asyncio.to_thread(lambda: list(log_dir.glob(pattern)))

    # Exclude container log files when getting main logs
    if not container:
        files = [f for f in files if '.container.' not in f.name]

    # Filter by restart_count if specified
    if restart_count is not None:
        files = [f for f in files if extract_fn(f.name) == restart_count]

    return sorted(files, key=lambda p: extract_fn(p.name))


async def historical_log_generator(
    log_dir: Path,
    model_instance_id: int,
    options: LogOptions,
    stop_event: Optional[asyncio.Event] = None,
    container: bool = False,
):
    """Generate logs from historical log files.

    Args:
        log_dir: Directory containing log files
        model_instance_id: Model instance ID
        options: Log options (tail, follow)
        stop_event: Event to signal stopping
        container: If True, read container logs; if False, read main logs

    Yields:
        Log lines from log files
    """
    log_files = await get_all_log_files(log_dir, model_instance_id, container=container)

    if not log_files:
        if container:
            logger.debug(
                f"No container log files found for model instance "
                f"{model_instance_id}"
            )
        return

    if options.tail > 0:
        # Only read the last N lines from the most recent log file
        if log_files:
            file_options = LogOptions(
                tail=options.tail, follow=options.follow, stop_event=stop_event
            )
            async for line in log_generator(str(log_files[-1]), file_options):
                if stop_event and stop_event.is_set():
                    logger.debug(
                        "Historical log generator stopping due to stop event 1"
                    )
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


async def merged_log_generator(  # noqa: C901
    log_paths: List[str],
    options: LogOptions,
    stop_event: Optional[asyncio.Event] = None,
):
    """Merge multiple log sources and yield lines as they become available.

    Args:
        log_paths: List of log file paths to read
        options: Log options (tail, follow)
        stop_event: Event to signal stopping

    Yields:
        Log lines from all sources in the order they become available
    """
    if not log_paths:
        return

    queues: List[asyncio.Queue] = []

    async def read_to_queue(queue: asyncio.Queue, log_path: str, opts: LogOptions):
        try:
            async for line in log_generator(log_path, opts):
                if stop_event and stop_event.is_set():
                    return
                await queue.put(("data", line))
        except Exception as e:
            logger.error(f"Error reading log {log_path}: {e}")
            await queue.put(("error", str(e)))
        finally:
            await queue.put(None)  # Signal end of this source

    # Create tasks for all log generators
    tasks = []
    for path in log_paths:
        queue = asyncio.Queue()
        queues.append(queue)
        task = asyncio.create_task(read_to_queue(queue, path, options))
        tasks.append(task)

    get_tasks = {}
    for q in queues:
        task = asyncio.create_task(q.get())
        get_tasks[task] = q

    # Yield lines as they become available from any source
    active_count = len(queues)
    while active_count > 0:
        if stop_event and stop_event.is_set():
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            break

        # Wait for any queue to have data
        # (with timeout to check stop_event periodically)
        done, _ = await asyncio.wait(
            get_tasks.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=0.5
        )

        # Check stop_event after timeout
        if stop_event and stop_event.is_set():
            for task in tasks:
                if not task.done():
                    task.cancel()
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


async def combined_log_generator(
    log_dir: Path | str,
    model_instance_id: int,
    download_log_path: str,
    options: LogOptionsDep,
    model_instance_name: str,
):
    """Unified log streaming from three file sources using LogSourceChain.

    Reads logs in order:
    1) Download logs (if exists)
    2) Historical main logs (all restart_count files)
    3) Container logs (from persisted files)

    Args:
        log_dir: Directory containing log files (Path or str)
        model_instance_id: Model instance ID
        download_log_path: Path to download log file
        options: Log options (tail, follow)
        model_instance_name: Model instance name (unused, kept for API compatibility)
    """
    log_dir = Path(log_dir)

    download_source = DownloadLogSource(download_log_path)
    main_source = MainLogSource(
        log_dir,
        model_instance_id,
        options.restart_count,
        get_all_log_files_fn=get_all_log_files,
    )
    container_source = ContainerLogSource(
        log_dir,
        model_instance_id,
        options.restart_count,
        get_all_log_files_fn=get_all_log_files,
        extract_restart_count_fn=extract_restart_count,
    )

    chain = LogSourceChain(
        [download_source, main_source],
        log_generator_fn=log_generator,
    )

    has_any_logs = False
    log_paths = []

    # Download log
    download_files = await download_source.wait_for_files_if_needed(
        follow=options.follow
    )
    if download_files:
        log_paths.append(str(download_files[0]))
        has_any_logs = True

    # Main logs
    main_log_files = await main_source.wait_for_files_if_needed(follow=options.follow)
    if main_log_files and options.restart_count is None:
        log_paths.extend(str(f) for f in main_log_files)
        has_any_logs = True

    # Stream download + main logs (merged)
    if log_paths:
        stop_event = asyncio.Event()
        monitor_task = None

        container_has_content = await container_source.has_content()
        if not container_has_content and options.follow:
            monitor_task = asyncio.create_task(
                chain.monitor_container_content(container_source, stop_event)
            )

        merge_options = (
            LogOptions(tail=-1, follow=False)
            if container_has_content or not options.follow
            else options
        )

        try:
            async for line in merged_log_generator(
                log_paths, merge_options, stop_event
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
            log_dir, model_instance_id, options, container=True
        ):
            yield line

    if not has_any_logs:
        raise NotFoundException(message="Log file not found")


@router.get("/serveLogs/{id}")
async def get_serve_logs(
    request: Request,
    session: SessionDep,
    id: int,
    log_options: LogOptionsDep,
    model_instance_name: str = Query(default=""),
    model_file_id: Optional[int] = Query(default=None),
):
    log_dir = request.app.state.config.log_dir
    serve_log_dir = Path(log_dir) / "serve"

    download_log_path = ""
    # Use model file ID for shared download logs if provided
    if model_file_id is not None:
        download_log_path = str(
            serve_log_dir / f"model_file_{model_file_id}.download.log"
        )

    return StreamingResponse(
        combined_log_generator(
            serve_log_dir,
            id,
            download_log_path,
            log_options,
            model_instance_name,
        ),
        media_type="application/octet-stream",
    )


@router.get("/benchmark_logs/{id}")
async def get_benchmark_logs(
    request: Request,
    session: SessionDep,
    id: int,
    log_options: LogOptionsDep,
    benchmark_name: str = Query(default=""),
):
    log_dir = request.app.state.config.log_dir
    main_log_path = Path(log_dir) / "benchmarks" / f"{id}.log"

    return StreamingResponse(
        combined_log_generator(
            str(main_log_path),
            "",
            log_options,
            benchmark_name,
        ),
        media_type="application/octet-stream",
    )
