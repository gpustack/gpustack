import asyncio
from dataclasses import dataclass
import os
import logging
from typing import Annotated

import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
from fastapi import Depends, Query


logger = logging.getLogger(__name__)


@dataclass
class LogOptions:
    tail: int = -1  # -1 by default means read all logs
    follow: bool = False

    def url_encode(self):
        return f"tail={self.tail}&follow={self.follow}"


default_tail = Query(
    default=-1, description="Number of lines to read from the end of the log"
)
default_follow = Query(default=False, description="Whether to follow the log output")


def get_log_options(
    tail: int = default_tail,
    follow: bool = default_follow,
) -> LogOptions:
    return LogOptions(tail=tail, follow=follow)


LogOptionsDep = Annotated[LogOptions, Depends(get_log_options)]


async def log_generator(path: str, options: LogOptions):
    logger.debug(f"Reading logs from {path} with options {options}")

    try:
        # By default, universal newline mode is used, which means that all of
        # \n, \r, or \r\n are recognized as end-of-line characters.
        # We use os.linesep to ensure that \r is reserved. It's useful for showing progress bars.
        async with aiofiles.open(
            path, "r", encoding="utf-8", errors="ignore", newline=os.linesep
        ) as file:
            if options.tail > 0:
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
                async for line in follow_file(file):
                    yield line
    except Exception as e:
        logger.error(f"Failed to read logs from {path}. {e}")


async def read_all_lines(file: AsyncTextIOWrapper):
    """Read all lines from the file."""
    while True:
        line = await file.readline()
        if not line:
            break
        yield line


async def follow_file(file: AsyncTextIOWrapper):
    """Follow the file and yield new lines as they are written."""
    while True:
        line = await file.readline()
        if not line:
            await asyncio.sleep(0.1)  # wait before retrying
            continue
        yield line
