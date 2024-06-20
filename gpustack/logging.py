import logging


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
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
    ]

    for logger_name in third_party_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
