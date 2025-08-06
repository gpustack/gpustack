from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pathlib import Path
from tenacity import RetryError
from datetime import datetime
import os

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.logs import log_generator
from gpustack.utils import file

router = APIRouter()


@router.get("/serveLogs/{id}")
async def get_serve_logs(request: Request, id: int, log_options: LogOptionsDep):
    log_dir = request.app.state.config.log_dir
    serve_log_dir = Path(log_dir) / "serve"
    
    # 尝试通过映射文件查找对应的日志目录
    mapping_file = serve_log_dir / "instance_mapping.txt"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        inst_id, log_dir_path = line.split(':', 1)
                        if int(inst_id) == id:
                            log_dir_path = Path(log_dir_path)
                            if log_dir_path.exists():
                                # 优先查找今天的日志文件
                                today = datetime.now().strftime("%Y-%m-%d")
                                today_log = log_dir_path / f"{today}.log"
                                if today_log.exists():
                                    try:
                                        file.check_file_with_retries(today_log)
                                        return StreamingResponse(log_generator(today_log, log_options), media_type="text/plain")
                                    except (FileNotFoundError, RetryError):
                                        pass
                                
                                # 如果今天的文件不存在，查找最新的日志文件
                                log_files = list(log_dir_path.glob("*.log"))
                                if log_files:
                                    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                                    try:
                                        file.check_file_with_retries(log_files[0])
                                        return StreamingResponse(log_generator(log_files[0], log_options), media_type="text/plain")
                                    except (FileNotFoundError, RetryError):
                                        pass
        except Exception:
            pass
    
    # 向后兼容：查找旧格式的日志文件
    old_path = serve_log_dir / f"{id}.log"
    if old_path.exists():
        try:
            file.check_file_with_retries(old_path)
            return StreamingResponse(log_generator(old_path, log_options), media_type="text/plain")
        except (FileNotFoundError, RetryError):
            pass
    
    raise NotFoundException(message="Log file not found")
