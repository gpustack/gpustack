from dataclasses import dataclass


@dataclass
class AgentConfig:
    debug: bool = False
    node_ip: str | None = None
    server: str | None = None
