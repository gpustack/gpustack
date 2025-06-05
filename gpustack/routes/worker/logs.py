from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pathlib import Path
from tenacity import RetryError

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.logs import log_generator
from gpustack.utils import file

router = APIRouter()


@router.get("/serveLogs/{id}")
async def get_serve_logs(request: Request, id: int, log_options: LogOptionsDep):
    log_dir = request.app.state.config.log_dir
    path = Path(log_dir) / "serve" / f"{id}.log"

    try:
        file.check_file_with_retries(path)
    except (FileNotFoundError, RetryError):
        raise NotFoundException(message="Log file not found")

    return StreamingResponse(log_generator(path, log_options), media_type="text/plain")
