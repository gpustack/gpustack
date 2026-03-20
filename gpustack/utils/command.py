import sys
import sysconfig
from os.path import dirname, abspath, join
import shutil
from typing import List, Optional, Tuple, Union
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


def extend_args_no_exist(
    arguments: List[str], *args: Union[str, Tuple[str, str]]
) -> None:
    """
    Extend arguments list with key-value pairs only if the key is not already present.

    This function prevents duplicate parameters when user-defined backend parameters
    may conflict with system-generated ones.

    Args:
        arguments: The list of arguments to extend (modified in place)
        *args: Variable number of arguments, each can be:
               - A tuple of (key, value) like ("--host", "127.0.0.1")
               - A single string key like "--enable-metrics" (flag without value)

    Examples:
        extend_args_no_exist(args, ("--host", "127.0.0.1"), ("--port", "8080"))
        extend_args_no_exist(args, "--enable-metrics")
    """
    for arg in args:
        if isinstance(arg, tuple):
            key, value = arg
            if not any(
                existing_arg == key or existing_arg.startswith(f"{key}=")
                for existing_arg in arguments
            ):
                arguments.extend([key, value])
        else:
            # Single flag without value
            if arg not in arguments:
                arguments.append(arg)


def _mask_json_segments(text: str):
    """
    Replace JSON segments with placeholders before shlex.split.

    This prevents shlex from incorrectly splitting JSON values that contain
    spaces or special characters. For example:
        --config={"key": "value with spaces"}
    Without masking, shlex would split on the spaces inside the JSON.
    """

    result = []
    mapping = {}

    i = 0
    n = len(text)
    json_id = 0

    while i < n:
        # Look for pattern: '=' followed by JSON start ('{' or '[')
        if text[i] == "=" and i + 1 < n and text[i + 1] in "{[":
            start = i + 1
            j = start

            # Track state for proper JSON parsing
            depth = 0  # Bracket nesting level
            in_string = False  # Whether we're inside a quoted string
            escape = False  # Whether previous char was backslash

            # Manually parse to find the matching closing bracket
            while j < n:
                c = text[j]

                # Handle escape sequences (e.g., \" inside strings)
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                # Toggle string state on unescaped quotes
                elif c == '"':
                    in_string = not in_string
                # Only count brackets outside of strings
                elif not in_string:
                    if c in "{[":
                        depth += 1
                    elif c in "}]":
                        depth -= 1
                        # Found matching closing bracket
                        if depth == 0:
                            j += 1
                            break

                j += 1

            # Extract the JSON segment
            json_text = text[start:j]

            # Create a unique placeholder
            placeholder = f"__JSON_{json_id}__"
            json_id += 1

            # Store mapping for later restoration
            mapping[placeholder] = json_text

            # Replace JSON with placeholder in result
            result.append("=" + placeholder)
            i = j
            continue

        # Copy non-JSON characters as-is
        result.append(text[i])
        i += 1

    return "".join(result), mapping


def _unmask_json(tokens, mapping):
    """Restore JSON placeholders back to original JSON text."""
    restored = []
    for t in tokens:
        # Replace all placeholders in this token with original JSON
        for k, v in mapping.items():
            if k in t:
                t = t.replace(k, v)
        restored.append(t)
    return restored


def safe_split(expr: str):
    """
    Split parameter string using shlex while preserving JSON values.

    Process:
    1. Mask JSON segments with placeholders
    2. Use shlex.split (which respects quotes and escapes)
    3. Restore original JSON text
    """
    masked, mapping = _mask_json_segments(expr)
    tokens = shlex.split(masked)
    return _unmask_json(tokens, mapping)
