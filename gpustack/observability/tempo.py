from decimal import Decimal, ROUND_DOWN
import logging
import os
from pathlib import Path
import re

import yaml

from gpustack.config.config import Config
from gpustack.utils.atomic_file import write_text_atomically

logger = logging.getLogger(__name__)

_TEMPO_DURATION_PART = re.compile(
    r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(ns|us|µs|μs|ms|s|m|h)"
)
_TEMPO_DURATION_UNITS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "μs": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
}


def _valid_tempo_retention(retention: str) -> bool:
    # Go durations support compound units and truncate each part to nanoseconds.
    duration = retention.removeprefix("+")
    parts = list(_TEMPO_DURATION_PART.finditer(duration))
    if "".join(part[0] for part in parts) != duration:
        return False
    nanoseconds = sum(
        (Decimal(part[1]) * _TEMPO_DURATION_UNITS[part[2]]).to_integral_value(
            rounding=ROUND_DOWN
        )
        for part in parts
    )
    return 0 < nanoseconds < 2**63


def prepare_builtin_tempo(cfg: Config) -> bool:
    """Write the configuration consumed by the supervised Tempo service."""
    if not cfg.use_builtin_tempo():
        return False

    path = Path(os.getenv("GPUSTACK_RUN_DIR") or "/run/gpustack") / "tempo/config.yaml"
    data = Path(cfg.data_dir) / "tempo"
    retention_env = "GPUSTACK_BUILTIN_TEMPO_RETENTION"
    default_retention = "168h"
    retention = os.getenv(retention_env) or default_retention
    if not _valid_tempo_retention(retention):
        retention = default_retention
        logger.warning(
            "Invalid %s. Using %s for embedded Tempo retention.",
            retention_env,
            retention,
        )
    loopback = "127.0.0.1"
    listen_address = cfg.builtin_tempo_otlp_listen_address
    otlp_host = f"[{listen_address}]" if ":" in listen_address else listen_address
    server = {
        "http_listen_address": loopback,
        "http_listen_port": cfg.builtin_tempo_http_port,
        "grpc_listen_address": loopback,
        "grpc_listen_port": cfg.builtin_tempo_grpc_port,
    }
    try:
        config = {
            "target": "all",
            "server": server,
            "distributor": {
                "receivers": {
                    "otlp": {
                        "protocols": {
                            "grpc": {
                                "endpoint": f"{otlp_host}:{cfg.builtin_tempo_otlp_port}"
                            },
                            "http": {
                                "endpoint": f"{loopback}:{cfg.builtin_tempo_otlp_http_port}"
                            },
                        },
                    },
                },
            },
            "ingester": {
                "lifecycler": {"address": loopback},
                "flush_all_on_shutdown": True,
            },
            "memberlist": {"bind_addr": [loopback], "bind_port": 0},
            "storage": {
                "trace": {
                    "backend": "local",
                    "blocklist_poll": "1m",
                    "wal": {"path": str(data / "wal")},
                    "local": {"path": str(data / "blocks")},
                },
            },
            "compactor": {
                "compaction": {
                    "block_retention": retention,
                }
            },
            "usage_report": {"reporting_enabled": False},
        }
        write_text_atomically(path, yaml.safe_dump(config, sort_keys=False))
    except OSError as exc:
        logger.warning("Unable to configure embedded Tempo: %s", exc)
        return False
    return True
