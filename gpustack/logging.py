import logging
import uvicorn
import uvicorn.config

logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging_level = logging.DEBUG

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging_format))

logger = logging.getLogger("gpustack")
logger.setLevel(logging_level)
logger.addHandler(handler)


uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
uvicorn_log_config["formatters"]["default"]["fmt"] = logging_format
