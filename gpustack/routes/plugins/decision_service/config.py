"""Typed payload of the decision-service plugin's route section.

The section configures the ``gpustack-lb-decision-service`` capability plugin
(AUTHN/328): per-request Jev model-selection over the candidate set the LB
context role publishes. The decision service itself is NOT configured here —
it comes from a ``type=gpustack-lb-typesafe`` ModelProvider whose
``provider-{id}`` catalogue entry the route selects via ``providerId``.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ModelSelectionConfig(BaseModel):
    """The injected ``model_selection`` choice question. ``criteria`` keys
    must equal the candidates' model names (the ModelRoute's target models);
    keys matching no candidate are inert. Descriptions are what the Jev
    service reasons over, so a name-only question is rejected here just as
    the plugin rejects it."""

    instructions: Optional[str] = None
    criteria: Dict[str, str] = Field(min_length=1)
    """Model name -> capability description. Required and non-empty: the
    descriptions are what the Jev service reasons over, so a name-only
    question is rejected here just as the plugin rejects it."""

    @model_validator(mode="after")
    def _descriptions_substantive(self):
        # min_length only guards the mapping's size; a blank key or an
        # empty description would persist a name-only question the Jev
        # service cannot reason over.
        for name, description in self.criteria.items():
            if not name.strip():
                raise ValueError("criteria keys are model names and must not be blank")
            if not (description or "").strip():
                raise ValueError(
                    f"criteria['{name}'] needs a capability description; "
                    "a name-only question carries nothing for the decision "
                    "service to reason over"
                )
        return self

    def to_gateway(self) -> Dict[str, Any]:
        question: Dict[str, Any] = {"criteria": self.criteria}
        if self.instructions is not None:
            question["instructions"] = self.instructions
        return question


class DecisionServiceRouteConfig(BaseModel):
    """The ``plugins["decision-service"]`` section: modelSelection plus the decision
    knobs that are per-route. The finisher weight (wasm ``rankWeight``) rides
    the shared capability ``weight`` column and the matchRule config."""

    enabled: bool = True
    providerId: int
    """The ModelProvider id of the decision service, rendered as the
    catalogue id ``provider-{id}`` (``activeProviderId`` on the rule).
    Required: the wasm plugin's implicit catalogue selection would bypass
    tenant isolation — an explicit reference is what gets ownership-checked
    at write time."""
    weight: Optional[float] = Field(default=None, gt=0)
    """Contribution weight in the finisher's L1-weighted sum, in (0, N] —
    the wasm plugin's ``rankWeight``. None uses the plugin's compiled-in
    default (10)."""
    decisionModel: Optional[str] = None
    """Route-level override of the decision-engine model (e.g.
    ``jev-latest`` / ``jev-preview``, discoverable via the service's
    ``/v1/models``). Precedence on the wire: this > the provider entry's
    ``model`` > omitted."""
    modelSelection: Optional[ModelSelectionConfig] = None

    def to_gateway_rule(self) -> Dict[str, Any]:
        # Imported here to keep the schemas -> gateway dependency direction
        # one-way (gateway imports schemas at module level already).
        from gpustack.gateway.utils import provider_registry_name

        rule: Dict[str, Any] = {
            "enabled": self.enabled,
            "activeProviderId": provider_registry_name(self.providerId),
        }
        if self.decisionModel is not None:
            rule["decisionModel"] = self.decisionModel
        if self.modelSelection is not None:
            rule["modelSelection"] = self.modelSelection.to_gateway()
        if self.weight is not None:
            rule["rankWeight"] = self.weight
        return rule
