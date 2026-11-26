from glob import escape as escape_glob
import json
import logging
import os
from pathlib import Path
import re
from string import Template
from urllib.parse import urlsplit

from gpustack_runtime import envs as runtime_envs

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.utils.atomic_file import write_text_atomically
from gpustack.worker.log_export import ENGINE_LOG_EXPORT_DIR

logger = logging.getLogger(__name__)

ALLOY_LOG_TEMPLATE = Path("/etc/gpustack/logging/docker.alloy.tmpl")
_DOCKER_MOUNTINFO = Path("/proc/self/mountinfo")
_DOCKER_CONTAINER_MOUNT = re.compile(r"/containers/([0-9a-f]{64})/[^/]+$")


def _log_export_options(cfg: Config) -> dict:
    endpoint = cfg.log_otlp_endpoint or ""
    try:
        url = urlsplit(endpoint)
        port = url.port
    except ValueError:
        raise ValueError("Invalid OTLP logs URL") from None
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or not url.path
        or url.username is not None
        or url.query
        or url.fragment
        or port == 0
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError("Expected a full OTLP/HTTP logs URL without credentials")

    authorization = (
        cfg.log_otlp_authorization.get_secret_value()
        if cfg.log_otlp_authorization
        else ""
    )
    if (authorization or cfg.log_otlp_ca_file) and url.scheme != "https":
        raise ValueError("OTLP log authentication and custom CAs require HTTPS")
    if any(ord(character) < 32 or ord(character) == 127 for character in authorization):
        raise ValueError("Invalid OTLP Authorization header")

    headers = (
        f'{{"Authorization" = {json.dumps(authorization)}}}' if authorization else "{}"
    )
    return {
        "endpoint": json.dumps(endpoint),
        "headers": headers,
        "ca_file": json.dumps(cfg.log_otlp_ca_file or ""),
    }


def _docker_application_log_receiver() -> str:
    # Docker bind-mounts these generated files even with a custom hostname
    # or a private cgroup namespace.
    container_ids = set()
    try:
        for line in _DOCKER_MOUNTINFO.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 5 or fields[4] not in (
                "/etc/hostname",
                "/etc/hosts",
                "/etc/resolv.conf",
            ):
                continue
            match = _DOCKER_CONTAINER_MOUNT.search(fields[3])
            if match:
                container_ids.add(match[1])
    except OSError:
        pass

    if len(container_ids) != 1:
        logger.warning("Unable to identify this Docker container for application logs")
        return ""

    (container_id,) = container_ids
    host = runtime_envs.GPUSTACK_RUNTIME_DOCKER_HOST.removeprefix("http+")
    return f'''loki.source.docker "application" {{
  host = {json.dumps(host)}
  targets = [{{"__meta_docker_container_id" = {json.dumps(container_id)}}}]
  forward_to = [otelcol.receiver.loki.application.receiver]
}}

otelcol.receiver.loki "application" {{
  output {{
    logs = [gpustack.logs.export.application]
  }}
}}'''


def _check_log_redact_patterns() -> None:
    name = "GPUSTACK_LOG_REDACT_PATTERNS"
    try:
        patterns = json.loads(os.getenv(name) or "[]")
    except json.JSONDecodeError:
        raise ValueError(f"{name} must be a JSON array of strings") from None
    if not isinstance(patterns, list):
        raise ValueError(f"{name} must be a JSON array of strings")
    for index, pattern in enumerate(patterns):
        # Alloy compiles the expressions with RE2 when it loads the pipeline.
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{name}[{index}] must be a non-empty RE2 expression")


def prepare_log_collection(cfg: Config) -> bool:
    """Generate the Docker Alloy config when log export is configured."""
    path = Path(os.getenv("GPUSTACK_RUN_DIR", "/run/gpustack")) / "logging/config.alloy"
    try:
        path.unlink(missing_ok=True)
        if not cfg.log_otlp_endpoint or cfg.log_pod_release:
            return False

        _check_log_redact_patterns()
        options = _log_export_options(cfg)
        directory = escape_glob(str(Path(cfg.log_dir)))
        state = Path(cfg.data_dir) / "log-collector"
        engine = [
            f"{directory}/{ENGINE_LOG_EXPORT_DIR}/archives/*/{name}{suffix}"
            for name in ("container.log", "container.workload-*.log")
            for suffix in ("", ".[0-9]*")
        ]
        higress_receiver = ""
        if cfg.gateway_mode == GatewayModeEnum.embedded:
            higress = [
                f"{directory}/higress/access.log{suffix}" for suffix in ("", ".[1-5]")
            ]
            higress_receiver = f'''otelcol.receiver.filelog "higress" {{
  include = {json.dumps(higress)}
  start_at = "end"
  storage = otelcol.storage.file.offsets.handler

  output {{
    logs = [gpustack.logs.export.higress]
  }}
}}'''

        config = Template(ALLOY_LOG_TEMPLATE.read_text()).substitute(
            **options,
            offsets=json.dumps(str(state)),
            application_receiver=_docker_application_log_receiver(),
            engine=json.dumps(engine),
            higress_receiver=higress_receiver,
        )

        write_text_atomically(path.with_name("storage-path"), str(state / "alloy"))
        # The generated config can contain credentials. The temporary file has mode 0600.
        write_text_atomically(path, config)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        logger.error("Unable to enable application log collection: %s", exc)
        return False
    return True
