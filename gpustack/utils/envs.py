import getpass
import os
import subprocess
from pathlib import Path
from typing import Dict


def get_unix_root_path_of_ascend() -> Path:
    """Returns the root path of the Ascend installation on *unix."""
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

    # Construct the right path based on directory ownership
    me = getpass.getuser()
    owner = ascend_home_path.owner()
    if me != owner:
        # In practice, personal Ascend tool is installed under "/home/<username>/Ascend"
        root_path = Path(f"/home/{me}/Ascend")
    else:
        # In practice, ASCEND_HOME_PATH should be "/usr/local/Ascend/ascend-toolkit/latest"
        # we keep the path from start to "Ascend" word if it exists
        root_path = Path(str(ascend_home_path).split("Ascend")[0] + "Ascend")

    # Create the directory if it does not exist
    root_path.mkdir(parents=True, exist_ok=True)

    return root_path


def extract_unix_vars_of_source(script_path: str) -> Dict[str, str]:
    """Extracts the environment variables from a source-able script on *unix."""
    # Assume the script exists and is executable
    script_path = Path(script_path)
    if not script_path.is_file():
        raise Exception(f"The file '{script_path}' does not exist or is not a file.")

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
            ['sh', '-c', 'env'], stderr=subprocess.PIPE, text=True
        )
        original = parse_env(original_env_output)

        # Get the environment variables after sourcing the script
        sourced_env_output = subprocess.check_output(
            ['sh', '-c', f'source {script_path} && env'],
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
            f"Failed to extract environment variables from '{script_path}': {e.stderr}"
        )
