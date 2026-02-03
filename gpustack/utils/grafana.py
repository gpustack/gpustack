from fastapi import Request

from gpustack.config.config import Config


def normalize_grafana_url(grafana_url: str) -> str:
    if not grafana_url:
        return ""
    if "://" not in grafana_url:
        grafana_url = f"http://{grafana_url}"
    return grafana_url.rstrip("/")


def resolve_grafana_base_url(cfg: Config, request: Request) -> str:
    if cfg.grafana_url is not None:
        return normalize_grafana_url(cfg.grafana_url or "")

    grafana_url = normalize_grafana_url(cfg.get_grafana_url() or "")
    if not grafana_url:
        return ""

    base = cfg.server_external_url or str(request.base_url).rstrip("/")
    return f"{base}/grafana"
