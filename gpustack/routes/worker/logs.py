from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pathlib import Path
from tenacity import RetryError
from datetime import datetime
import os
import glob

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.logs import LogOptionsDep
from gpustack.worker.logs import log_generator
from gpustack.utils import file

router = APIRouter()


@router.get("/serveLogs/{id}")
async def get_serve_logs(request: Request, id: int, log_options: LogOptionsDep):
    log_dir = request.app.state.config.log_dir
    serve_log_dir = Path(log_dir) / "serve"
    
    # 首先尝试查找旧格式的日志文件（向后兼容）
    old_path = serve_log_dir / f"{id}.log"
    if old_path.exists():
        try:
            file.check_file_with_retries(old_path)
            return StreamingResponse(log_generator(old_path, log_options), media_type="text/plain")
        except (FileNotFoundError, RetryError):
            pass
    
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
                                log_files = glob.glob(str(log_dir_path / "*.log"))
                                if log_files:
                                    # 优先选择最新日期的文件 (YYYY-MM-DD.log 格式)
                                    date_files = [(os.path.basename(f), f) for f in log_files 
                                                 if os.path.basename(f).endswith('.log') and 
                                                    len(os.path.basename(f)) == 14 and  # YYYY-MM-DD.log = 14 chars
                                                    os.path.basename(f)[:10].count('-') == 2]  # 确保是日期格式
                                    if date_files:
                                        date_files.sort(reverse=True)
                                        try:
                                            file.check_file_with_retries(Path(date_files[0][1]))
                                            return StreamingResponse(log_generator(Path(date_files[0][1]), log_options), media_type="text/plain")
                                        except (FileNotFoundError, RetryError):
                                            pass
                                    # 回退到最新修改的文件
                                    log_files.sort(key=os.path.getmtime, reverse=True)
                                    try:
                                        file.check_file_with_retries(Path(log_files[0]))
                                        return StreamingResponse(log_generator(Path(log_files[0]), log_options), media_type="text/plain")
                                    except (FileNotFoundError, RetryError):
                                        pass
        except Exception:
            pass
    
    raise NotFoundException(message="Log file not found")
