"""Init-pass diff for the decision-service CR.

The capability convention splits the CR in two: init rewrites the static
half (url, phase, priority, the inert defaultConfig) and the reconcilers own
matchRules. The decision-service CR has a third field in play — the providers
catalogue inside defaultConfig, written by the ModelProvider-event sync — so
this diff additionally carries the live ``providers`` / ``activeProviderId``
keys of defaultConfig over, the same preserved-keys pattern the
model-router CR uses. The three writers never fight over one field.
"""

from typing import Callable, Optional

from gpustack.gateway.client.extensions_higress_io_v1_api import WasmPluginSpec

# defaultConfig keys owned by the providers-catalogue sync, not by init.
# ``activeProviderId`` is deliberately absent: the catalogue has no synthetic
# default entry any more, so a stale live value would name a missing provider
# and make the plugin's config parse fail -- init must let it be dropped.
_DYNAMIC_DEFAULT_CONFIG_KEYS = ("providers",)


def decision_static_spec_diff(
    expected_spec: WasmPluginSpec,
) -> Callable[[Optional[WasmPluginSpec]], Optional[WasmPluginSpec]]:
    def _diff(current: Optional[WasmPluginSpec]) -> Optional[WasmPluginSpec]:
        if current is None:
            return expected_spec
        current_dict = current.model_dump(exclude_none=True)
        merged = expected_spec.model_dump(exclude_none=True)
        if current_dict.get("matchRules"):
            merged["matchRules"] = current_dict["matchRules"]
        current_default = current_dict.get("defaultConfig") or {}
        for key in _DYNAMIC_DEFAULT_CONFIG_KEYS:
            if key in current_default:
                merged.setdefault("defaultConfig", {})[key] = current_default[key]
        return WasmPluginSpec.model_validate(merged)

    return _diff
