from dataclasses import dataclass
import os


@dataclass
class AgentConfig:
    debug: bool = False
    node_ip: str | None = None
    server: str | None = None
    metric_enabled: bool = True
    metrics_port: int = 10051
    log_dir: str = os.path.expanduser("~/.local/share/gpustack/log")
