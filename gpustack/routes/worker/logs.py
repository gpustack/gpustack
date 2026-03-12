import asyncio
import logging
import time
from pathlib import Path

from tenacity import RetryError
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from gpustack_runtime.deployer import logs_workload

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptions, LogOptionsDep, log_generator
from gpustack.utils import file
from gpustack.server.deps import SessionDep
from gpustack import envs


router = APIRouter()

logger = logging.getLogger(__name__)


def is_container_logs_ready(model_instance_name: str) -> bool:
    """Probe whether container logs have any content available to output."""
    try:
        logs_workload(
            name=model_instance_name,
            tail=1,
            follow=False,
        )
        return True
    except Exception:
        return False


async def wait_for_container_generator(
    model_instance_name: str,
    options: LogOptionsDep,
    poll_interval: float = 5,
    timeout: float = envs.PROXY_TIMEOUT,
):
    """Wait until container logs are ready, then return the async generator.

    Keeps probing until logs are ready and the container_log_generator can be constructed.
    """
    gen = None
    start_time = time.monotonic()
    while gen is None and (time.monotonic() - start_time) < timeout:
        if is_container_logs_ready(model_instance_name):
            try:
                gen = container_log_generator(model_instance_name, options)
            except Exception as e:
                logger.error(
                    f"Failed to initialize container logs for {model_instance_name}: {e}"
                )
                gen = None
        if gen is None:
            await asyncio.sleep(poll_interval)
    if gen is None:
        elapsed = time.monotonic() - start_time
        logger.warning(
            f"Waiting for container logs timed out after {elapsed:.0f}s for {model_instance_name}."
        )
        raise asyncio.TimeoutError(
            f"Timed out waiting for container logs for {model_instance_name} after {elapsed:.0f}s"
        )
    return gen


async def _await_next_or_preempt(
    pending_tasks: dict, stop_event: Optional[asyncio.Event]
):
    """Wait for the next available task to complete, or preempt if stop_event is set.

    Returns a tuple of (done_tasks, preempting_flag).
    """
    if stop_event is not None:
        preempt_task = asyncio.create_task(stop_event.wait())
        wait_set = set(pending_tasks.keys()) | {preempt_task}
        done_set, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        preempting = preempt_task in done_set
        if preempting:
            done = set(
                t for t in done_set if t is not preempt_task and t in pending_tasks
            )
        else:
            done = set(done_set)
        preempt_task.cancel()
        return done, preempting
    else:
        done_set, _ = await asyncio.wait(
            pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        return set(done_set), False


async def _cancel_all(pending_tasks: dict):
    """Cancel all tasks in the pending_tasks dict and wait for cancellation."""
    tasks = list(pending_tasks.keys())
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def merge_async_generators(
    *generators, stop_event: Optional[asyncio.Event] = None
):
    """Merge multiple async generators into a single stream, with optional preemption."""

    # Wrap each generator to yield plain items (no index needed)
    async def _wrap_gen(g):
        async for item in g:
            yield item

    tasks = [_wrap_gen(gen) for gen in generators]
    if not tasks:
        return

    pending_tasks = {asyncio.create_task(gen.__anext__()): gen for gen in tasks}

    try:
        while pending_tasks:
            # If already preempting, flush any completed tasks; otherwise wait
            if stop_event and stop_event.is_set():
                done = set(t for t in pending_tasks.keys() if t.done())
                if not done:
                    await _cancel_all(pending_tasks)
                    break
                preempting = True
            else:
                done, preempting = await _await_next_or_preempt(
                    pending_tasks, stop_event
                )

            for task in done:
                gen = pending_tasks.pop(task)
                try:
                    result = task.result()
                    yield result
                    if not preempting:
                        new_task = asyncio.create_task(gen.__anext__())
                        pending_tasks[new_task] = gen
                except StopAsyncIteration:
                    pass
                except Exception as e:
                    logger.error(f"Error processing generator output: {e}")

            if preempting:
                await _cancel_all(pending_tasks)
                break
    finally:
        await _cancel_all(pending_tasks)


async def container_log_generator(model_instance_name: str, options: LogOptionsDep):
    """
    Generate logs from CustomServer Docker container.

    Args:
        model_instance_name: Workload name
        options: Log options (tail, follow)
    """
    try:
        # Convert LogOptions to container log parameters
        tail = options.tail if options.tail else -1
        follow = options.follow

        # Get logs from the workload
        log_stream = logs_workload(
            name=model_instance_name,
            tail=tail,
            follow=follow,
        )

        # Handle different return types based on follow parameter
        if follow:
            # Offload blocking iteration to a background thread to avoid event loop blocking
            try:
                iterator = iter(log_stream)
            except Exception as e:
                logger.error(
                    f"logs_workload did not return an iterable for {model_instance_name}: {e}"
                )
                return

            while True:
                try:
                    log_line = await asyncio.to_thread(next, iterator)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(
                        f"Error reading container log for {model_instance_name}: {e}"
                    )
                    break

                if isinstance(log_line, bytes):
                    yield log_line.decode('utf-8', errors='replace')
                else:
                    yield str(log_line)
        else:
            # When follow=False, logs_workload returns a string or bytes
            if isinstance(log_stream, bytes):
                yield log_stream.decode('utf-8', errors='replace')
            else:
                yield str(log_stream)

    except Exception as e:
        cause = getattr(e, "__cause__", None)
        cause_text = f": {cause}" if cause else ""
        logger.error(
            f"Failed to get container logs for {model_instance_name}: {e}{cause_text}"
        )


async def _stream_file_logs_preempt_to_container(
    file_tasks,
    options,
    model_instance_name: str,
):
    """
    Stream file logs (download + main) in follow mode until the container produces
    its first log line, then preempt and switch to container logs.
    """
    preempt_event = asyncio.Event()
    container_queue: asyncio.Queue = asyncio.Queue()

    async def pump_container_logs():
        first_item_emitted = False
        try:
            gen = await wait_for_container_generator(
                model_instance_name,
                options,
                poll_interval=5,
                timeout=envs.PROXY_TIMEOUT,
            )
            async for cline in gen:
                if not first_item_emitted:
                    first_item_emitted = True
                    logger.info(f"Preempting file logs for {model_instance_name}")
                    preempt_event.set()
                await container_queue.put(cline)
        except asyncio.TimeoutError:
            logger.warning(
                f"Pump container logs timed out for {model_instance_name} after {envs.PROXY_TIMEOUT}s"
            )
        except Exception as e:
            logger.error(f"Error streaming container logs: {e}")
        finally:
            await container_queue.put(None)

    # Start container pumping concurrently
    pump_task = asyncio.create_task(pump_container_logs())

    try:
        # Emit file logs until preempted
        async for line in merge_async_generators(*file_tasks, stop_event=preempt_event):
            yield line

        # Now switch to container logs (if any)
        logger.info(f"Switching to container logs for {model_instance_name}")
        while True:
            item = await container_queue.get()
            if item is None:
                break
            yield item
    finally:
        # Cancel the container pump task
        pump_task.cancel()


async def _emit_file_then_container_logs(
    file_tasks,
    options,
    container_ready: bool,
    model_instance_name: str,
):
    """
    Emit file logs first, then stream or snapshot container logs depending on options.follow.
    """
    if file_tasks:
        async for line in merge_async_generators(*file_tasks):
            yield line

    try:
        if options.follow:
            gen = await wait_for_container_generator(
                model_instance_name, options, timeout=envs.PROXY_TIMEOUT
            )
            async for line in gen:
                yield line
        else:
            if container_ready:
                async for line in container_log_generator(model_instance_name, options):
                    yield line
    except Exception as e:
        logger.error(f"Failed to stream container logs: {e}")


async def combined_log_generator(
    main_log_path: str,
    download_log_path: str,
    options: LogOptionsDep,
    model_instance_name: str,
    file_log_exists: bool = True,
):
    """Unified log streaming across three startup phases.

    Behavior:
    1) main_log and download_log are interleaved and emitted first.
       - If follow=True and container logs are not yet available, file logs are streamed in follow mode.
       - When container logs produce the first line, file log streaming is preempted:
         any already-ready file lines are flushed, then following is stopped and we switch to container logs.

    2) If the request arrives after container logs are already available and follow=True:
       - Emit main_log and download_log non-streaming (follow=False) first (respecting tail).
       - Then stream container_log.

    3) If follow=False:
       - Emit file logs non-streaming.
       - Emit container logs as a single snapshot (non-streaming) if available.

    Additional notes:
    - Container readiness is probed via logs_workload(name, tail=1, follow=False).
    - Preemption only happens when file log merge is waiting for new content (no immediate line ready),
      ensuring newly produced file lines are not discarded.
    - If the workload isn't ready at request time, a background probe keeps checking and starts
      streaming container logs once content becomes available, without requiring the client to reconnect.
    - If neither file logs nor container logs are available, NotFoundException is raised.
    """

    # Phase 1: file logs (download + main) merged together
    file_tasks = []

    # Decide how to handle file logs based on whether container logs are already available
    file_options = options
    container_ready = False
    try:
        # Quick probe: check if container logs have content available now
        container_ready = is_container_logs_ready(model_instance_name)

        # If the client requested follow but container logs are already available,
        # emit file logs non-streaming first, then stream container logs.
        if options and options.follow and container_ready:
            file_options = LogOptions(tail=options.tail, follow=False)
    except Exception:
        file_options = options

    # Add download log if needed and file exists
    if download_log_path and Path(download_log_path).exists():
        file_tasks.append(log_generator(download_log_path, file_options))

    if file_log_exists:
        file_tasks.append(log_generator(main_log_path, file_options))

    # Prepare container logs (Phase 2)
    if (
        not file_tasks and not container_ready
    ):  # No download/main logs and no container logs
        raise NotFoundException(message="Log file not found")

    # If following and container not yet ready, allow interleaved streaming of file logs,
    # but preempt as soon as container produces any output.
    if options and options.follow and not container_ready and file_tasks:
        async for line in _stream_file_logs_preempt_to_container(
            file_tasks, options, model_instance_name
        ):
            yield line
        return

    # Default behavior: emit file logs first (non-follow if configured), then container logs
    async for line in _emit_file_then_container_logs(
        file_tasks, options, container_ready, model_instance_name
    ):
        yield line


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
    main_log_path = Path(log_dir) / "serve" / f"{id}.log"

    download_log_path = ""
    # Use model file ID for shared download logs if provided
    if model_file_id is not None:
        download_log_path = (
            Path(log_dir) / "serve" / f"model_file_{model_file_id}.download.log"
        )

    # Check if log file exists for file-based backends
    # For custom backends, we'll try container logs first
    try:
        file.check_file_with_retries(main_log_path)
        file_log_exists = True
    except (FileNotFoundError, RetryError):
        file_log_exists = False

    if log_options.follow:
        file_log_exists = True

    # show_download_logs parameter is passed from server based on model instance state
    return StreamingResponse(
        combined_log_generator(
            str(main_log_path),
            str(download_log_path),
            log_options,
            model_instance_name,
            file_log_exists,
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

    try:
        file.check_file_with_retries(main_log_path)
        file_log_exists = True
    except (FileNotFoundError, RetryError):
        file_log_exists = False

    return StreamingResponse(
        combined_log_generator(
            str(main_log_path),
            "",
            log_options,
            benchmark_name,
            file_log_exists,
        ),
        media_type="application/octet-stream",
    )
