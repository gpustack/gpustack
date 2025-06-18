import sys
import sysconfig
from os.path import dirname, abspath, join
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


def find_parameter(parameters: List[str], param_names: List[str]) -> Optional[str]:
    """
    Find specified parameter by name from the parameters.
    Return the value of the parameter if found, otherwise return None.
    """
    if parameters is None:
        return None

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


def find_bool_parameter(parameters: List[str], param_names: List[str]) -> bool:
    """
    Find specified boolean parameter by name from the parameters.
    Return True if the parameter is set, otherwise return False.
    """
    for i, param in enumerate(parameters):
        if param.lstrip('-') in param_names:
            return True
    return False


def normalize_parameters(
    args: List[str], removes: Optional[List[str]] = None, sep: str = '='
):
    """
    Split parameter strings and filter removes parameters.

    Processes command line arguments by:
    1. Splitting key=value parameters into separate elements
    2. Removing specified parameters and their values

    Args:
        args: List of input parameters
        removes: List of parameter names to remove
        sep: Delimiter used for key-value pairs (default: '=')

    Returns:
        List of processed parameters with removes items removed
    """
    parameters = []
    for param in args:
        if '=' in param:
            key, value = param.split(sep, 1)
            parameters.append(key)
            parameters.append(value)
        else:
            parameters.append(param)
    normalize_args = []

    i = 0
    while i < len(parameters):
        param = parameters[i]
        if '=' in param:
            key, _ = param.split('=', 1)
            key = key.lstrip('-')
            if removes and key in removes:
                i += 1
                continue
        else:
            key = param.lstrip('-')
            if removes and key in removes:
                i += 2 if i + 1 < len(parameters) else 1
                continue
        normalize_args.append(param)
        i += 1
    return normalize_args


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
