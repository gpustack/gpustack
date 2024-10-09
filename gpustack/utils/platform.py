import platform
import torch


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
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return ""
