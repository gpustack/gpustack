import asyncio
import logging
from pathlib import Path
from tenacity import RetryError
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.logs import log_generator
from gpustack.utils import file


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


async def combined_log_generator(main_log_path: str, download_log_path: str, options):
    """Generate logs with optional download logs prepended"""

    tasks = []

    # Add download log if needed and file exists
    if download_log_path and Path(download_log_path).exists():
        tasks.append(log_generator(download_log_path, options))

    # Always add main log
    tasks.append(log_generator(main_log_path, options))

    # Use merge_async_generators for both follow and non-follow modes
    async for line in merge_async_generators(*tasks):
        yield line


@router.get("/serveLogs/{id}")
async def get_serve_logs(
    request: Request,
    id: int,
    log_options: LogOptionsDep,
    model_file_id: Optional[int] = Query(
        default=None, description="Model file ID for shared download logs"
    ),
):
    log_dir = request.app.state.config.log_dir
    main_log_path = Path(log_dir) / "serve" / f"{id}.log"

    download_log_path = ""
    # Use model file ID for shared download logs if provided
    if model_file_id is not None:
        download_log_path = (
            Path(log_dir) / "serve" / f"model_file_{model_file_id}.download.log"
        )

    try:
        file.check_file_with_retries(main_log_path)
    except (FileNotFoundError, RetryError):
        raise NotFoundException(message="Log file not found")

    # show_download_logs parameter is passed from server based on model instance state
    return StreamingResponse(
        combined_log_generator(str(main_log_path), str(download_log_path), log_options),
        media_type="text/plain",
    )
