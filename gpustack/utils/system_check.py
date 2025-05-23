import subprocess
import re
import logging

from gpustack.utils import platform
from gpustack.utils.command import is_command_available


logger = logging.getLogger(__name__)


def check_glibc_version(min_version='2.29'):
    """
    Check if the glibc version is greater than or equal to the specified minimum version.
    Raises an exception if the version is lower than the minimum required version.
    """
    if platform.system() != 'linux':
        return

    if not is_command_available('ldd'):
        logger.debug("ldd command not found. Skipping glibc version check.")
        return

    try:
        output = subprocess.check_output(['ldd', '--version'], text=True)
        first_line = output.splitlines()[0]
        match = re.search(r'(\d+\.\d+)', first_line)
        if not match:
            logger.debug(f"Failed to parse glibc version. Output: {first_line}")
            return
        version = match.group(1)

        def parse_ver(v):
            return [int(x) for x in v.split('.')]

        v1 = parse_ver(version)
        v2 = parse_ver(min_version)
        length = max(len(v1), len(v2))
        v1 += [0] * (length - len(v1))
        v2 += [0] * (length - len(v2))

        if v1 >= v2:
            return
        else:
            raise RuntimeError(
                f"glibc version {version} is lower than required {min_version}. Consider using Docker as an alternative."
            )
    except subprocess.CalledProcessError as e:
        logger.debug(f"Error checking glibc version: {e}")
        return
