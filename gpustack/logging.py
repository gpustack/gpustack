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

    # Disable third-party loggers
    third_party_logger_names = [
        "httpcore.connection",
        "httpx",
        "httpcore.http11",
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

    for logger_name in third_party_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
