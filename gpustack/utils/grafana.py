from typing import Dict, Optional
from urllib.parse import urlencode

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


def counted_role(model) -> str:
    """The role whose KV transfer counters this model's connector populates.

    `decode` when the catalog cannot say: the two-hop connectors are the common
    case, and a wrong guess here shows an empty panel rather than a wrong
    number. Never None, so every disaggregated model resolves to a role -- a
    missing answer would blank the transfer panels, which reads as a broken
    group rather than as an unknown connector.

    Read by the exporter, which publishes it as `gpustack:pd_mode_info`'s
    `kv_counted_role`, and by the dashboard link below.
    """
    # Imported here rather than at module scope: the catalog is a server-side
    # singleton and this module is imported by things that have no server.
    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode_name = model.disaggregation.mode.value
    mode = get_pd_mode(mode_name)
    if mode and mode.transfer_metrics and mode.transfer_metrics.read_from_role:
        return mode.transfer_metrics.read_from_role
    return "decode"


def build_model_dashboard_url(
    cfg: Config,
    grafana_base: str,
    model,
    cluster_name: Optional[str] = None,
    extra_params: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """The monitoring link for one deployment, PD-aware.

    A disaggregated group goes to the PD dashboard, not the model one. The
    model dashboard is not merely less specific for a group, it is wrong in one
    place: every request traverses both roles, so its request counters double
    under PD. Sending a group there hands the reader a number that is 2x
    reality with nothing saying so.

    One function, because there is more than one place that links to a
    deployment's dashboard -- the model page and a benchmark report -- and a
    group sent to the model dashboard from either of them is the same wrong
    number.

    Returns None when the dashboard this deployment would use is not
    configured. The uid checked is the one it will actually use, so an install
    that configured only the model dashboard is not redirected to a PD
    dashboard that does not exist.
    """
    pd = bool(getattr(model, "disaggregation", None))
    uid = cfg.grafana_pd_dashboard_uid if pd else cfg.grafana_model_dashboard_uid
    if not cfg.get_grafana_url() or not uid:
        return None

    query_params: Dict[str, str] = {}
    if cluster_name:
        query_params["var-cluster_name"] = cluster_name
    query_params["var-model_name"] = model.name
    if pd:
        # Which role's transfer counter is authoritative for this connector --
        # decode where it pulls (NIXL), prefill where it pushes (SGLang). The
        # dashboard resolves this for itself off `gpustack:pd_mode_info`, so
        # this is the first-paint value: it is what the transfer panels read
        # until that variable's own query returns, and what they fall back to
        # if the series is not there yet.
        query_params["var-counted_role"] = counted_role(model)
    query_params.update(extra_params or {})

    slug = "gpustack-pd" if pd else "gpustack-model"
    dashboard_url = f"{grafana_base}/d/{uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"
    return dashboard_url
