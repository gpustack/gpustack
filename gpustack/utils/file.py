import glob
import logging
import os
import re
import shutil
import asyncio
from pathlib import Path
from typing import Callable
from tenacity import retry, stop_after_attempt, wait_fixed

from gpustack.utils import platform

logger = logging.getLogger(__name__)

# Modes for the data dir and the credentials inside it (the JWT signing
# secret, registration and worker tokens, the initial admin password).
#
# The data dir is 0711 rather than 0700: the embedded PostgreSQL runs as its
# own user and must traverse it on the way to <data_dir>/postgresql. 0711
# grants that traversal to every principal while denying them a listing, and
# the credential files inside are 0600, so traverse-only access to the
# directory yields nothing.
DATA_DIR_MODE = 0o711
CREDENTIAL_FILE_MODE = 0o600


def restrict_permissions(path: str, mode: int) -> None:
    """Best-effort chmod to ``mode``; a failure is never fatal.

    Args:
        path: File or directory to tighten.
        mode: Target permission bits.

    A missing path and a read-only mount (the Helm chart subPath-mounts the
    bootstrap-password Secret) are both expected in normal operation, so
    failures are logged at debug and swallowed.
    """
    try:
        os.chmod(path, mode)
    except OSError as e:
        logger.debug(f"Could not tighten permissions on {path} to {oct(mode)}: {e}")


def write_credential_file(path: str, content: str) -> None:
    """Overwrite ``path`` with ``content`` readable only by its owner.

    The file is created 0600 and re-tightened afterwards: os.open applies the
    mode only at creation, so an existing file whose mode is looser than 0600
    is closed by the write as well, not only on a fresh install.

    Args:
        path: File to write.
        content: Full file contents.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, CREDENTIAL_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(content)
    restrict_permissions(path, CREDENTIAL_FILE_MODE)


def ensure_dir(path: str, mode: int) -> None:
    """Create ``path`` with ``mode``.

    The explicit chmod is load-bearing: os.makedirs applies its mode through
    the process umask, so a stricter umask would silently narrow the
    directory (umask 077 turns mode=0711 into 0700, which the embedded
    PostgreSQL cannot traverse).

    Args:
        path: Directory to create, with parents if needed. Intermediate
            directories keep the umask default; only ``path`` is tightened.
        mode: Permission bits to enforce on ``path``.
    """
    os.makedirs(path, mode=mode, exist_ok=True)
    restrict_permissions(path, mode)


def get_local_file_size_in_byte(file_path):
    if os.path.islink(file_path):
        file_path = os.path.realpath(file_path)

    size = os.path.getsize(file_path)
    return size


def copy_with_owner(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    copy_owner_recursively(src, dst)


def copy_owner_recursively(src, dst):
    if platform.system() in ["linux", "darwin"]:
        st = os.stat(src)
        os.chown(dst, st.st_uid, st.st_gid)
        for dirpath, dirnames, filenames in os.walk(dst):
            for dirname in dirnames:
                os.chown(os.path.join(dirpath, dirname), st.st_uid, st.st_gid)
            for filename in filenames:
                os.chown(os.path.join(dirpath, filename), st.st_uid, st.st_gid)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
def check_file_with_retries(path: Path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")


async def check_with_retries(checker: Callable, timeout: int = 30, interval: int = 1):
    """Generic async retry wrapper for checking operations.

    Args:
        checker: A callable (sync or async) that performs the check and returns a result.
                 Should raise an exception if the check fails (triggers retry).
        timeout: Maximum time to wait in seconds (default: 30)
        interval: Time between retries in seconds (default: 1)

    Returns:
        The result from the checker function

    Raises:
        Exception: Whatever exception the checker raises on final failure

    Example:
        def check_files():
            files = get_files()
            if not files:
                raise FileNotFoundError("No files found")
            return files

        files = await check_with_retries(check_files, timeout=60, interval=1)
    """
    elapsed = 0
    while elapsed < timeout:
        try:
            if asyncio.iscoroutinefunction(checker):
                return await checker()
            else:
                return checker()
        except Exception:
            elapsed += interval
            if elapsed >= timeout:
                raise
            await asyncio.sleep(interval)


def delete_path(path: str):
    """
    Delete a file or directory. If the path is a symbolic link, it will delete the target path.
    """
    if not os.path.lexists(path):
        return

    if os.path.islink(path):
        target_path = os.path.realpath(path)
        os.unlink(path)
        if os.path.lexists(target_path):
            delete_path(target_path)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        for item in os.scandir(path):
            delete_path(item.path)
        shutil.rmtree(path)


def getsize(path: str) -> int:
    """
    Get the total size of the path in bytes. Handles symbolic links and directories.
    """
    # Cache the size of directories to avoid redundant calculations.
    dir_size_cache = {}
    # Keep track of visited directories to avoid infinite loops.
    visited_dirs = set()
    return _getsize(path, visited_dirs, dir_size_cache)


def _getsize(path: str, visited: set, cache: dict) -> int:
    real_path = os.path.realpath(path)

    if os.path.islink(path):
        return _getsize(real_path, visited, cache)
    elif os.path.isfile(real_path):
        return os.path.getsize(real_path)
    elif os.path.isdir(real_path):
        if real_path in visited:
            return 0
        visited.add(real_path)

        if real_path in cache:
            return cache[real_path]

        total = 0
        with os.scandir(real_path) as entries:
            for entry in entries:
                try:
                    total += _getsize(entry.path, visited, cache)
                except FileNotFoundError:
                    pass

        cache[real_path] = total
        return total

    raise FileNotFoundError(f"Path does not exist: {path}")


def get_sharded_file_paths(file_path: str) -> str:
    dir_name, base_name = os.path.split(file_path)
    match = re.match(r"(.*?)-\d{5}-of-\d{5}\.(\w+)", base_name)
    if not match:
        return [file_path]
    prefix = match.group(1)
    extension = match.group(2)
    pattern = os.path.join(dir_name, f"{prefix}-*-of-*.{extension}")
    return sorted(glob.glob(pattern))
