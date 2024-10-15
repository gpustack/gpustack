import os
import platform


def system() -> str:
    return platform.uname().system.lower()


def arch() -> str:
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "i386": "386",
        "i686": "386",
        "arm64": "arm64",
        "aarch64": "arm64",
        "armv7l": "arm",
        "arm": "arm",
        "ppc64le": "ppc64le",
        "s390x": "s390x",
    }
    return arch_map.get(platform.machine().lower(), "unknown")


def device() -> str:
    if os.system("nvidia-smi > /dev/null 2>&1") == 0 or os.path.exists(
        "/usr/local/cuda"
    ):
        return "cuda"

    if system() == "darwin" and arch() == "arm64":
        return "mps"

    return ""
