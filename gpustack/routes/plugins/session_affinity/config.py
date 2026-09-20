"""Typed payload of the session-affinity plugin's route section."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SessionKey(BaseModel):
    """One source in the ordered session-key chain; the first one that
    yields a value wins. Exactly one of header / bodyKey per entry —
    header sources are free, bodyKey sources buffer a copy of the body
    (bounded by enableOnPathSuffix)."""

    header: Optional[str] = None
    bodyKey: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_source(self):
        if (self.header is None) == (self.bodyKey is None):
            raise ValueError(
                "each sessionKey must set exactly one of 'header' or 'bodyKey'"
            )
        return self


class SessionAffinityConfig(BaseModel):
    enabled: bool = True

    sessionKeys: List[SessionKey]
    """Ordered chain — required. Omitting it makes the gateway plugin
    fail to parse the rule, so it is mandatory here rather than
    discovering it at the gateway."""

    enableOnPathSuffix: Optional[List[str]] = None
    """Body-source gate only. The plugin's default
    (["/responses", "/messages"]) deliberately excludes
    /chat/completions — the hottest path, with no known standard
    session key on it."""

    weight: Optional[float] = Field(default=None, gt=0)
    """Contribution weight in the finisher's L1-weighted sum, in (0, N] —
    fractional weights are meaningful (a plugin can be dialled to a small
    share of the sum). None uses the plugin's compiled-in default."""

    def to_gateway_rule(self) -> Dict[str, Any]:
        rule: Dict[str, Any] = {
            "sessionKeys": [
                key.model_dump(exclude_none=True) for key in self.sessionKeys
            ]
        }
        if self.enableOnPathSuffix is not None:
            rule["enableOnPathSuffix"] = self.enableOnPathSuffix
        if self.weight is not None:
            rule["weight"] = self.weight
        return rule


def session_affinity_from_meta(
    meta: Optional[Dict[str, Any]],
) -> Optional[SessionAffinityConfig]:
    section = (meta or {}).get("session-affinity")
    if not section:
        return None
    return SessionAffinityConfig.model_validate(section)
