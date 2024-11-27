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


def get_versioned_command(command_name: str, version: str) -> str:
    """
    Get the versioned command name.
    """
    if command_name.endswith(".exe"):
        return f"{command_name[:-4]}_{version}.exe"

    return f"{command_name}_{version}"
