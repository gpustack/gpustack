import platform
import shutil
from typing import List, Optional


def is_command_available(command_name):
    """
    Use `shutil.which` to determine whether a command is available.

    Args:
    command_name (str): The name of the command to check.

    Returns:
    bool: True if the command is available, False otherwise.
    """

    return shutil.which(command_name) is not None


def get_platform_command(command_map: dict, *extra_keys) -> str:
    """
    Get the command for the current platform.

    Args:
        command_map (dict): A mapping of platform to command.
    """

    system = platform.system()
    arch = platform.machine().lower()
    key = (system, arch)

    if extra_keys:
        key = key + extra_keys

    command = command_map.get(key, "")
    if command:
        return command

    # try same arch.
    equal_arch = {
        "x86_64": "amd64",
        "amd64": "x86_64",
        "aarch64": "arm64",
        "arm64": "aarch64",
    }

    arch = equal_arch.get(arch, "")
    key = (system, arch)
    if extra_keys:
        key = key + extra_keys

    command = command_map.get(key, "")
    return command


def find_parameter(parameters: List[str], param_names: List[str]) -> Optional[str]:
    """
    Find specified parameter by name from the parameters.
    Return the value of the parameter if found, otherwise return None.
    """
    for i, param in enumerate(parameters):
        if '=' in param:
            key, value = param.split('=', 1)
            if key.lstrip('-') in param_names:
                return value
        else:
            if param.lstrip('-') in param_names:
                if i + 1 < len(parameters):
                    return parameters[i + 1]
    return None
