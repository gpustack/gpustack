import os
import sys
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)
legacy_uuid_filename = "worker_uuid"


def get_legacy_uuid(data_dir: str) -> Optional[str]:
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    if os.path.exists(legacy_uuid_path):
        with open(legacy_uuid_path, "r") as file:
            return file.read().strip()
    return None


def write_legacy_uuid(data_dir: str) -> str:
    import uuid

    legacy_uuid = str(uuid.uuid4())
    legacy_uuid_path = os.path.join(data_dir, legacy_uuid_filename)
    try:
        with open(legacy_uuid_path, "w") as file:
            file.write(legacy_uuid)
        return legacy_uuid
    except Exception as e:
        raise RuntimeError("Failed to write legacy UUID") from e


def get_system_uuid(data_dir: str, write: bool = True) -> str:
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
        logger.info("try to create legacy uuid for worker")
        if write:
            return write_legacy_uuid(data_dir)
        else:
            raise e


def get_machine_id() -> str:
    system = sys.platform
    try:
        if system == 'linux':
            for path in ['/etc/machine-id', '/var/lib/dbus/machine-id']:
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    return f.read().strip()
        elif system == 'darwin':
            return ''
        elif system == 'win32':
            import winreg

            reg_path = r"SOFTWARE\\Microsoft\\Cryptography"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                return value
    except Exception:
        pass
    return ""
