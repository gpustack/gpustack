import sys
import sysconfig
from os.path import dirname, abspath, join
import shutil
from typing import List, Optional
import shlex


def is_command_available(command_name):
    """
    Use `shutil.which` to determine whether a command is available.

    Args:
    command_name (str): The name of the command to check.

    Returns:
    bool: True if the command is available, False otherwise.
    """

    return shutil.which(command_name) is not None


def find_parameter(parameters: List[str], param_names: List[str]) -> Optional[str]:
    """
    Find specified parameter by name from the parameters.
    Return the value of the parameter if found, otherwise return None.
    """
    if parameters is None:
        return None

    for i, param in enumerate(parameters):
        # Strip whitespace from the parameter
        param_stripped = param.strip()

        if '=' in param_stripped:
            key, value = param_stripped.split('=', 1)
            if key.strip().lstrip('-') in param_names:
                return value
        elif ' ' in param_stripped:
            key, value = param_stripped.split(' ', 1)
            if key.strip().lstrip('-') in param_names:
                split_values = shlex.split(value)
                if len(split_values) == 1:
                    return split_values[0]
                return value
        else:
            if param_stripped.lstrip('-') in param_names:
                if i + 1 < len(parameters):
                    return parameters[i + 1]
    return None


def find_int_parameter(parameters: List[str], param_names: List[str]) -> Optional[int]:
    """
    Find specified integer parameter by name from the parameters.
    Return the integer value of the parameter if found, otherwise return None.
    """
    value = find_parameter(parameters, param_names)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None


def find_bool_parameter(parameters: List[str], param_names: List[str]) -> bool:
    """
    Find specified boolean parameter by name from the parameters.
    Return True if the parameter is set, otherwise return False.
    """
    if parameters is None:
        return False

    for i, param in enumerate(parameters):
        param_stripped = param.strip()
        if param_stripped.lstrip('-') in param_names:
            return True
    return False


def get_versioned_command(command_name: str, version: str) -> str:
    """
    Get the versioned command name.
    """
    if command_name.endswith(".exe"):
        return f"{command_name[:-4]}_{version}.exe"

    return f"{command_name}_{version}"


def get_command_path(command_name: str) -> str:
    """
    Return the full path of sepcified command. Supports both frozen and python base environments.
    """
    base_path = (
        dirname(sys.executable)
        if getattr(sys, 'frozen', False)
        else sysconfig.get_path("scripts")
    )
    return abspath(join(base_path, command_name))
