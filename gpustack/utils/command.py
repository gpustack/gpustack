import platform
import shutil


def is_command_available(command_name):
    """
    Use `shutil.which` to determine whether a command is available.

    Args:
    command_name (str): The name of the command to check.

    Returns:
    bool: True if the command is available, False otherwise.
    """

    return shutil.which(command_name) is not None


def get_platform_command(command_map: dict) -> str:
    """
    Get the command for the current platform.

    Args:
        command_map (dict): A mapping of platform to command, example:
        {
            ("Windows", "amd64"): "run-windows-amd64.exe",
            ("Darwin", "amd64"): "run-darwin-amd64",
            ("Linux", "amd64"): "run-linux-amd64",
            ("Linux", "arm64"): "run-linux-arm64",
        }
    """

    system = platform.system()
    arch = platform.machine().lower()

    command = command_map.get((system, arch), "")
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
    return command_map.get((system, arch), "")
