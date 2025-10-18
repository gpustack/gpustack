import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


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


def is_docker_env() -> bool:
    return os.path.exists("/.dockerenv")


def sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    """
    Sanitize the environment variables by removing any keys that are not valid
    environment variable names.
    """
    prefixes = ("GPUSTACK_",)
    suffixes = (
        "_KEY",
        "_key",
        "_TOKEN",
        "_token",
        "_SECRET",
        "_secret",
        "_PASSWORD",
        "_password",
        "_PASS",
        "_pass",
    )
    return {
        k: v
        for k, v in env.items()
        if not k.startswith(prefixes) and not k.endswith(suffixes)
    }
