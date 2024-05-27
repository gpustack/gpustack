import logging


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Disable third-party loggers
    third_party_logger_names = [
        "httpcore.connection",
        "httpx",
        "httpcore.http11",
        "asyncio",
        "urllib3.connectionpool",
    ]

    for logger_name in third_party_logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
