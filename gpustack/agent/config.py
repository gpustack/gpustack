from dataclasses import dataclass
import os


@dataclass
class AgentConfig:
    debug: bool = False
    node_ip: str | None = None
    server: str | None = None
    log_dir: str = os.path.expanduser(f"~/.local/share/gpustack/log")
