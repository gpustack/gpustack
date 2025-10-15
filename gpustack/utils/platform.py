import platform


def system() -> str:
    """
    Get the current operating system name in lowercase.
    """
    return platform.uname().system.lower()


def native_arch() -> str:
    """
    Get the native architecture of the machine in lowercase.
    """
    return platform.machine().lower()


_ARCH_ALIAS_MAPPING = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "i386": "386",
    "i686": "386",
    "arm64": "arm64",
    "aarch64": "arm64",
    "armv7l": "arm",
}


def arch() -> str:
    """
    Get the architecture of the machine in a standardized format.
    If the architecture is not recognized, return the native architecture.
    """
    na = native_arch()
    return _ARCH_ALIAS_MAPPING.get(native_arch(), na)


def system_arch() -> str:
    """
    Get the system and architecture in the format "system/arch".
    """
    return f"{system()}/{arch()}"
