from datetime import datetime, timezone
import logging


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.fromtimestamp(
            record.created, timezone.utc
        )
        .astimezone()
        .isoformat(timespec="seconds")
    )

    # Third-party loggers to disable
    disable_logger_names = [
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.proxy",
        "httpx",
        "asyncio",
        "aiosqlite",
        "urllib3.connectionpool",
        "multipart.multipart",
        "apscheduler.scheduler",
        "apscheduler.executors.default",
        "tzlocal",
        "alembic.runtime.migration",
        "filelock",
    ]

    for logger_name in disable_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True

    # Third-party loggers to print on debug
    debug_logger_names = [
        "alembic.runtime.migration",
    ]

    for logger_name in debug_logger_names:
        logger = logging.getLogger(logger_name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.disabled = True
