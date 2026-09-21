"""Typed payload of the LB route plugin's policy section.

This is the server-side schema of what rides in
``PUT /model-routes/{id}`` under ``plugins.lb`` and is stored verbatim
in ``ModelRoute.meta["lb"]`` (see lb/plugin.py). Field names and nesting
follow the gateway plugin's config contract (extensions/gpustack-lb in
gpustack-higress-plugins): everything here is handed to the CR config
as-is, so a name drift between the two sides is a silent behavior gap —
keep them in step.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# Capability plugin names as they appear in the gateway framework
# (gpustack-lb-least-load, gpustack-lb-session-affinity, the enterprise
# gpustack-lb-prefix-affinity). The name is open, not an enum: the
# enterprise plugin ships separately and this server must accept it.
lb_capability_name_pattern = r'^[a-z][a-z0-9\-]*$'


class LBHealthConfig(BaseModel):
    """Health windows. All optional — an absent knob means the gateway
    plugin's compiled-in default."""

    fail_open: Optional[bool] = Field(default=None)
    unhealthy_threshold: Optional[int] = Field(default=None, ge=1)
    cooldown_ms: Optional[int] = Field(default=None, ge=1)
    ramp_ms: Optional[int] = Field(default=None, ge=1)

    def to_gateway(self) -> Dict[str, Any]:
        # Gateway field names are lowerCamel; only include knobs that
        # were actually set — an absent key keeps the plugin's
        # compiled-in default, and an explicit null is not the same
        # contract.
        dump = self.model_dump(exclude_none=True)
        return {
            camel: dump.pop(snake)
            for camel, snake in (
                ("failOpen", "fail_open"),
                ("unhealthyThreshold", "unhealthy_threshold"),
                ("cooldownMs", "cooldown_ms"),
                ("rampMs", "ramp_ms"),
            )
            if snake in dump
        }


class LBRejectConfig(BaseModel):
    status: int = Field(default=503, ge=400, le=599)
    message: str = Field(default="no healthy model instance available", max_length=255)


class LBPolicyConfig(BaseModel):
    enabled: bool = True
    """Row presence stores the configuration; ``enabled`` is the switch
    that lets a route keep its LB settings while temporarily falling
    back to plain weighted-cluster routing."""

    health: LBHealthConfig = Field(default_factory=LBHealthConfig)
    reject: LBRejectConfig = Field(default_factory=LBRejectConfig)
    max_body_bytes: Optional[int] = Field(default=None, ge=1)
    """Body buffer ceiling shared by the lb plugin and any body-reading
    capability plugin; they must be configured together."""

    def to_gateway_rule(self) -> Dict[str, Any]:
        """The policy knobs for a route's matchRule config — health
        windows (only the ones actually set; an absent key keeps the
        plugin's compiled-in default), the reject response, and the
        shared body-buffer ceiling. Rendered into the per-route rule by
        the reconciler (a Higress matchRule config overrides the CR's
        defaultConfig), so each route's ``plugins.lb`` knobs take
        effect on that route alone. The shared state backend is
        deliberately not here: redis is deployment-level only
        (``--redis-url`` / GPUSTACK_REDIS_URL, rendered onto both
        gateway CRs by lb/gateway.py), because the plugin requires it
        on both roles at once and a per-route override cannot satisfy
        that. ``mode`` is role-level, never per-route."""
        config: Dict[str, Any] = {}
        health = self.health.to_gateway()
        if health:
            config["health"] = health
        config["reject"] = self.reject.model_dump()
        if self.max_body_bytes is not None:
            config["maxBodyBytes"] = self.max_body_bytes
        return config


def lb_policy_from_meta(meta: Optional[Dict[str, Any]]) -> Optional["LBPolicyConfig"]:
    """The policy out of ``ModelRoute.meta["lb"]``, or None when the
    route carries no LB section. Lives here (config) rather than the
    plugin module so the reconciler can import it without a cycle."""
    section = (meta or {}).get("lb")
    if not section:
        return None
    return LBPolicyConfig.model_validate(section)
