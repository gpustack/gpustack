"""Typed payload of the least-load plugin's route section."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LeastLoadConfig(BaseModel):
    enabled: bool = True

    weight: Optional[float] = Field(default=None, gt=0)
    """Contribution weight in the finisher's L1-weighted sum, in (0, N] —
    fractional weights are meaningful (a plugin can be dialled to a small
    share of the sum). None uses the plugin's compiled-in default."""

    def to_gateway_rule(self) -> Dict[str, Any]:
        rule: Dict[str, Any] = {"enabled": self.enabled}
        if self.weight is not None:
            rule["weight"] = self.weight
        return rule


def least_load_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[LeastLoadConfig]:
    section = (meta or {}).get("least-load")
    if not section:
        return None
    return LeastLoadConfig.model_validate(section)
