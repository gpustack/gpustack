import functools
import getpass
import os
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


@functools.lru_cache
def get_unix_available_root_paths_of_ascend(writable: bool = False) -> List[Path]:
    """Returns the available root paths of the Ascend installation on *unix."""
    # Assume the CANN has been installed,
    # and the environment variable "ASCEND_HOME_PATH" has been set.
    ascend_home_path = os.getenv("ASCEND_HOME_PATH")
    if not ascend_home_path:
        raise Exception("The environment variable 'ASCEND_HOME_PATH' is not set.")

    ascend_home_path = Path(ascend_home_path)
    if not ascend_home_path.is_dir():
        raise Exception(
            f"The path '{ascend_home_path}' does not exist or is not a directory."
        )

    # In practice, ASCEND_HOME_PATH should be "/usr/local/Ascend/ascend-toolkit/latest",
    # we keep the path from start to "Ascend" word if it exists, like "/usr/local/Ascend".
    ascend_root_path = Path(str(ascend_home_path).split("Ascend")[0] + "Ascend")

    root_paths = []

    # Judge whether the ascend_root_path can be accessed by the current user.
    cuser = getpass.getuser()
    owner = ascend_home_path.owner()
    if cuser != owner:
        import pwd

        # Gain user info of the current user.
        cuser_info = pwd.getpwnam(cuser)
        # Gain all groups of the current user.
        try:
            cuser_groups = os.getgrouplist(cuser, cuser_info.pw_gid)
        except AttributeError:
            cuser_groups = [cuser_info.pw_gid]
        # Gain mode mask
        ascend_root_path_stat = ascend_root_path.stat()
        # - Group member can RWX
        if ascend_root_path_stat.st_gid in cuser_groups:
            mode_mask = stat.S_IRGRP | stat.S_IXGRP
            if writable:
                mode_mask |= stat.S_IWGRP
        # - Others can RWX
        else:
            mode_mask = stat.S_IROTH | stat.S_IXOTH
            if writable:
                mode_mask |= stat.S_IWOTH
        # Add candidate root path if it can be accessed by the current user.
        if ascend_root_path_stat.st_mode & mode_mask == mode_mask:
            root_paths.append(ascend_root_path)

        # In practice, personal Ascend tool is installed under "/home/<username>/Ascend",
        # we can create the directory if it does not exist, and add it to the candidate root paths.
        personal_ascend_root_path = Path(f"/home/{cuser}/Ascend")
        root_paths.append(personal_ascend_root_path)
    else:
        root_paths.append(ascend_root_path)

    return root_paths


def extract_unix_vars_of_source(script_paths: List[Path]) -> Dict[str, str]:
    """
    Extracts the environment variables from a source-able script on *unix.
    Needs to be sourced in a bash shell.
    """
    # Assume the script exists and is executable
    for script_path in script_paths:
        if not script_path.is_file():
            raise Exception(
                f"The file '{script_path}' does not exist or is not a file."
            )

    # Parse the result output of executing "env"
    def parse_env(env_str):
        env = {}
        for line in env_str.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                env[key] = value
        return env

    try:
        # Get original environment variables
        original_env_output = subprocess.check_output(
            ['bash', '-c', 'env'],
            stderr=subprocess.PIPE,
            text=True,
        )
        original = parse_env(original_env_output)

        # Merge all sourcing script paths in to one command
        source_command = ' && '.join(
            [f'source {script_path}' for script_path in script_paths]
        )

        # Get the environment variables after sourcing the script
        sourced_env_output = subprocess.check_output(
            ['bash', '-c', f'{source_command} && env'],
            stderr=subprocess.PIPE,
            text=True,
        )
        sourced = parse_env(sourced_env_output)

        # Get the difference
        diff = {
            k: v
            for k, v in sourced.items()
            if k not in original or original.get(k) != v
        }

        return diff
    except subprocess.CalledProcessError as e:
        raise Exception(
            f"Failed to extract environment variables from [{script_paths}]: {e.stderr}"
        )


def get_gpustack_env(env_var: str) -> Optional[str]:
    env_name = "GPUSTACK_" + env_var
    return os.getenv(env_name)


def get_gpustack_env_bool(env_var: str) -> Optional[bool]:
    env_name = "GPUSTACK_" + env_var
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.lower() in ["true", "1"]
    return None
