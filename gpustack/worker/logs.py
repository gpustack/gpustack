from dataclasses import dataclass
import os
import time
import logging
from typing import Annotated

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


def log_generator(path: str, options: LogOptions):
    logger.debug(f"Reading logs from {path} with options {options}")

    # By default, universal newline mode is used, which means that all of
    # \n, \r, or \r\n are recognized as end-of-line characters.
    # We use os.linesep to ensure that \r is reserved. It's useful for showing progress bars.
    with open(path, "r", encoding="utf-8", errors="ignore", newline=os.linesep) as file:
        if options.tail > 0:
            # Move to the end of the file and read the last 'tail' lines
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            buffer = []
            while file_size > 0 and len(buffer) < options.tail:
                file.seek(max(0, file_size - 1024), os.SEEK_SET)
                lines = file.readlines()
                buffer = lines[-options.tail :] + buffer
                file_size -= 1024
            for line in buffer[-options.tail :]:
                yield line
        else:
            lines = file.readlines()
            for line in lines:
                yield line

        if options.follow:
            while True:
                line = file.readline()
                if not line:
                    time.sleep(0.1)  # wait before retrying
                    continue
                yield line
