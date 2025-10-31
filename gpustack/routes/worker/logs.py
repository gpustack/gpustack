import asyncio
import logging
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


router = APIRouter()

logger = logging.getLogger(__name__)


async def merge_async_generators(*generators):  # noqa: C901
    """
    Merge multiple async generators into a single stream
    e.g:
    *generators = [download_log_generator, main_log_generator]
    # Task is used for fetching the next line of the log file.
    pending_tasks = {
        Task_A: download_log_generator,
        Task_B: main_log_generator
    }
    # while a task completes:
    1. Pop from pending_tasks;
    2. Yield the log line from the task result;
    3. Push the new task for fetching the next line to pending_tasks
    """
    tasks = []

    async def wrap_generator(index, gen):
        """Wrap generator to include index for identification"""
        try:
            async for item in gen:
                yield (index, item)
        except Exception as e:
            logger.error(f"Error in generator {index}: {e}")

    # Create tasks for each generator
    for i, gen in enumerate(generators):
        task = wrap_generator(i, gen)
        tasks.append(task)

    if not tasks:
        return

    # Use asyncio to handle multiple generators concurrently
    pending_tasks = {asyncio.create_task(gen.__anext__()): gen for gen in tasks}

    try:
        while pending_tasks:
            done, pending = await asyncio.wait(
                pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                gen = pending_tasks.pop(task)
                try:
                    index, result = task.result()
                    yield result
                    # Schedule the next item from this generator
                    new_task = asyncio.create_task(gen.__anext__())
                    pending_tasks[new_task] = gen
                except StopAsyncIteration:
                    # This generator is exhausted
                    pass
                except Exception as e:
                    logger.error(f"Error processing generator output: {e}")
    finally:
        # Cancel any remaining tasks
        for task in pending_tasks.keys():
            task.cancel()
        # Wait for all tasks to complete cancellation
        if pending_tasks:
            await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)


async def container_log_generator(model_instance_name: str, options):
    """
    Generate logs from CustomServer Docker container.

    Args:
        model_instance_name: Workload name
        options: Log options (tail, follow)
    """
    try:
        # Convert LogOptions to container log parameters
        tail = options.tail if options.tail > 0 else None
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
        logger.error(f"Failed to get container logs for {model_instance_name}: {e}")


async def combined_log_generator(
    main_log_path: str,
    download_log_path: str,
    options,
    model_instance_name: str,
    file_log_exists: bool = True,
):
    """Generate logs with optional download logs prepended"""

    # Phase 1: file logs (download + main) merged together
    file_tasks = []

    # If follow=True, don't follow file logs to avoid indefinite merging; only emit existing file content
    file_options = options
    try:
        if getattr(options, "follow", False):
            file_options = LogOptions(tail=options.tail, follow=False)
    except Exception:
        file_options = options

    # Add download log if needed and file exists
    if download_log_path and Path(download_log_path).exists():
        file_tasks.append(log_generator(download_log_path, file_options))

    if file_log_exists:
        file_tasks.append(log_generator(main_log_path, file_options))

    # Prepare container logs (Phase 2)
    container_gen = None
    try:
        container_gen = container_log_generator(model_instance_name, options)
    except Exception as e:
        logger.error(f"Failed to get workload: {e}")

    if (
        not file_tasks and container_gen is None
    ):  # No download/main logs and no container logs
        raise NotFoundException(message="Log file not found")

    # Emit file logs first
    if file_tasks:
        async for line in merge_async_generators(*file_tasks):
            yield line

    # Then emit container logs separately
    if container_gen is not None:
        async for line in container_gen:
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
