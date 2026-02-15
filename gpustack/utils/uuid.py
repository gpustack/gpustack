import os
import sys
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)
legacy_uuid_filename = "worker_uuid"
worker_name_filename = "worker_name"
worker_id_filename = "worker_id"


def get_legacy_uuid(data_dir: str) -> Optional[str]:
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    if os.path.exists(legacy_uuid_path):
        with open(legacy_uuid_path, "r") as file:
            return file.read().strip()
    return None


def set_legacy_uuid(data_dir: str, legacy_uuid: str):
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    with open(legacy_uuid_path, "w") as file:
        file.write(legacy_uuid)


def get_system_uuid() -> str:
    system = sys.platform
    linux_uuid_path = '/sys/class/dmi/id/product_uuid'
    try:
        if system == 'linux' and os.path.exists(linux_uuid_path):
            with open(linux_uuid_path, 'r') as f:
                return f.read().strip()
        elif system == 'darwin':  # MacOS
            output = subprocess.check_output(
                ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice']
            )
            for line in output.decode().split('\n'):
                if 'IOPlatformUUID' in line:
                    return line.split('=')[-1].strip().strip('"')
        elif system == 'win32':
            output = subprocess.check_output(
                ['wmic', 'csproduct', 'get', 'uuid'], stderr=subprocess.DEVNULL
            )
            lines = output.decode().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
        else:
            raise RuntimeError(f"Not supported OS or unable to retrieve {system} UUID")
    except Exception as e:
        logger.warning(f"{e}")
        raise e


def get_worker_name(data_dir: str) -> Optional[str]:
    worker_name_path = os.path.join(data_dir, worker_name_filename)
    if os.path.exists(worker_name_path):
        with open(worker_name_path, "r") as file:
            return file.read().strip()
    return None


def set_worker_name(data_dir: str, worker_name: str):
    worker_name_path = os.path.join(data_dir, worker_name_filename)
    current_worker_name = get_worker_name(data_dir)
    if current_worker_name is None or current_worker_name != worker_name:
        logger.warning(
            f"Worker name is being updated from {current_worker_name or '<empty>'} to {worker_name}"
        )
        with open(worker_name_path, "w") as file:
            file.write(worker_name)


def get_worker_id(data_dir: str) -> Optional[int]:
    worker_id_path = os.path.join(data_dir, worker_id_filename)
    if os.path.exists(worker_id_path):
        with open(worker_id_path, "r") as file:
            try:
                return int(file.read().strip())
            except ValueError:
                logger.warning(f"Invalid content in worker_id file: {worker_id_path}. Ignoring.")
                return None
    return None


def set_worker_id(data_dir: str, worker_id: int):
    worker_id_path = os.path.join(data_dir, worker_id_filename)
    current_worker_id = get_worker_id(data_dir)
    if current_worker_id != worker_id:
        logger.warning(
            f"Worker ID is being updated from {current_worker_id or '<empty>'} to {worker_id}"
        )
    with open(worker_id_path, "w") as file:
        file.write(str(worker_id))
