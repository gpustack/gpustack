from typing import Dict, Optional
from urllib.parse import urlencode
import json
import re

from fastapi import Request

from gpustack.config.config import Config


def normalize_grafana_url(grafana_url: str) -> str:
    if not grafana_url:
        return ""
    if "://" not in grafana_url:
        grafana_url = f"http://{grafana_url}"
    return grafana_url.rstrip("/")


def resolve_server_base_url(cfg: Config, request: Optional[Request] = None) -> str:
    """Resolve GPUStack's public base from configuration or request context."""
    configured = getattr(cfg, "server_external_url", None)
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/") if request is not None else ""


def resolve_grafana_base_url(cfg: Config, request: Request) -> str:
    if cfg.grafana_url is not None:
        return normalize_grafana_url(cfg.grafana_url or "")

    grafana_url = normalize_grafana_url(cfg.get_grafana_url() or "")
    if not grafana_url:
        return ""

    base = resolve_server_base_url(cfg, request)
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


def _re2_literal(value: str) -> str:
    # RE2 metacharacters match Go regexp.QuoteMeta; whitespace is literal.
    # Hex escaping keeps a name from closing a PromQL raw string.
    return re.sub(r'([\\.*+?()|\[\]{}^$])', r'\\\1', value).replace('`', r'\x60')


def _promql_regex_literal(value: str) -> str:
    return json.dumps(f'^{_re2_literal(value)}$', ensure_ascii=False)


def model_dashboard_scope_params(
    *,
    cluster_id: Optional[int] = None,
    model_id: Optional[int] = None,
    model_instance_id: Optional[int] = None,
    restart_count: Optional[int] = None,
    source_kind: Optional[str] = None,
    source_name: Optional[str] = None,
    worker_id: Optional[int] = None,
    dp_rank: Optional[int] = None,
    component_attempt: Optional[int] = None,
    cluster_name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_instance_name: Optional[str] = None,
) -> Dict[str, str]:
    """Encode already-authorized identities for fixed Grafana textbox scopes.

    Values are complete quoted PromQL regex literals consumed with :raw. They
    never contain caller-provided query fragments, label names, or destinations.
    This helper does not authorize or resolve an instance target.
    """
    numbers = dict(
        cluster_id=cluster_id,
        model_id=model_id,
        model_instance_id=model_instance_id,
        restart_count=restart_count,
        worker_id=worker_id,
        component_attempt=component_attempt,
    )
    for key, value in {**numbers, "dp_rank": dp_rank}.items():
        minimum = 0 if key in ("restart_count", "component_attempt", "dp_rank") else 1
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"Invalid dashboard {key}")
    if source_kind not in (
        None,
        "legacy",
        "vllm_engine",
        "vllm_hybrid",
        "sglang_engine",
        "sglang_smg",
    ):
        raise ValueError("Invalid dashboard source kind")
    if dp_rank is not None and source_kind not in ("vllm_engine", "sglang_engine"):
        raise ValueError("A dashboard rank requires an Engine source")
    if component_attempt is not None and source_kind != "sglang_smg":
        raise ValueError("A dashboard attempt requires an SMG source")
    scopes = {
        **numbers,
        "source_kind": source_kind,
        "source_name": source_name,
        "endpoint_dp_rank": dp_rank if source_kind == "vllm_engine" else None,
        "scheduler_dp_rank": dp_rank if source_kind == "sglang_engine" else None,
    }
    params = {
        f"var-scope_{key}": (
            '".*"'
            if value is None
            else (
                json.dumps(str(value))
                if isinstance(value, int)
                else _promql_regex_literal(value)
            )
        )
        for key, value in scopes.items()
    }
    params["var-scope_bound"] = (
        "1" if any(v is not None for v in scopes.values()) else "0"
    )
    managed_target = any(
        v is not None
        for v in (
            restart_count,
            source_kind,
            source_name,
            worker_id,
            dp_rank,
            component_attempt,
        )
    )
    params["var-scope_legacy_enabled"] = "0" if managed_target else "1"
    # Legend formats interpolate variables; rename transformations do not.
    params["var-legend_instance_prefix"] = (
        ""
        if model_instance_id is not None
        else "[{{model_instance_name}}] · [{{model_instance_id}}] · "
    )
    for key, value in dict(
        cluster_name=cluster_name,
        model_name=model_name,
        model_instance_name=model_instance_name,
    ).items():
        if value is not None:
            params[f"var-{key}"] = _re2_literal(value)
            params[f"var-display_{key}"] = value
    return params


def build_model_dashboard_url(
    cfg: Config,
    grafana_base: str,
    model,
    cluster_name: Optional[str] = None,
    extra_params: Optional[Dict[str, str]] = None,
    request: Optional[Request] = None,
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
        query_params["var-cluster_name"] = _re2_literal(cluster_name)
    query_params["var-model_name"] = _re2_literal(model.name)
    server_base = resolve_server_base_url(cfg, request)
    if server_base:
        query_params["var-gpustack_base_url"] = server_base
    if pd:
        # PD selectors use exact matches inside quoted PromQL strings.
        query_params["var-model_name"] = json.dumps(model.name, ensure_ascii=False)[
            1:-1
        ]
        if cluster_name:
            query_params["var-cluster_name"] = json.dumps(
                cluster_name, ensure_ascii=False
            )[1:-1]
        # Which role's transfer counter is authoritative for this connector --
        # decode where it pulls (NIXL), prefill where it pushes (SGLang). The
        # dashboard resolves this for itself off `gpustack:pd_mode_info`, so
        # this is the first-paint value: it is what the transfer panels read
        # until that variable's own query returns, and what they fall back to
        # if the series is not there yet.
        query_params["var-counted_role"] = counted_role(model)
    else:
        query_params.update(
            model_dashboard_scope_params(
                cluster_id=getattr(model, "cluster_id", None),
                model_id=getattr(model, "id", None),
                cluster_name=cluster_name,
                model_name=model.name,
            )
        )
    query_params.update(extra_params or {})

    slug = "gpustack-pd" if pd else "gpustack-model"
    dashboard_url = f"{grafana_base}/d/{uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"
    return dashboard_url
