"""Typed payload of the decision-service plugin's route section.

The section configures the ``gpustack-lb-decision-service`` capability plugin
(AUTHN/328): per-request Jev model-selection over the candidate set the LB
context role publishes. The decision service itself is NOT configured here —
it comes from a ``type=gpustack-lb-typesafe`` ModelProvider whose
``provider-{id}`` catalogue entry the route selects via ``providerId``.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    providerId: Optional[int] = None
    """The ModelProvider id of an external decision service, rendered as the
    catalogue id ``provider-{id}`` (``activeProviderId`` on the rule).
    Required when ``modelId`` is absent: the wasm plugin's implicit catalogue
    selection would bypass tenant isolation — an explicit reference is what
    gets ownership-checked at write time."""
    modelId: Optional[int] = None
    """The id of a locally deployed decision Model (``decision`` in
    ``categories``), rendered as the catalogue id ``gpustack-model-{id}``.
    Mutually exclusive with ``providerId``: the entry points straight at the
    deployment's running instance, so no provider row is involved."""
    weight: Optional[float] = Field(default=None, gt=0)
    """Contribution weight in the finisher's L1-weighted sum, in (0, N] —
    the wasm plugin's ``rankWeight``. None uses the plugin's compiled-in
    default (10)."""
    decisionModel: str = Field(min_length=1)
    """The decision-engine model for this route (e.g. ``jev-latest`` /
    ``jev-preview``, discoverable via the service's ``/v1/models``).
    Required: when blank the decision request ships without a ``model``
    field and the service does not perform a routing decision, so the
    policy would silently not run. For a local ``modelId`` target the
    model's own name is the natural default — local engines accept any
    alias."""
    modelSelection: Optional[ModelSelectionConfig] = None

    @model_validator(mode="after")
    def _exactly_one_decision_target(self):
        if (self.providerId is None) == (self.modelId is None):
            raise ValueError(
                "exactly one of providerId (external decision service) or "
                "modelId (locally deployed decision model) must be set"
            )
        return self

    @field_validator("decisionModel")
    @classmethod
    def _decision_model_substantive(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("decisionModel must not be blank")
        return stripped

    def to_gateway_rule(self) -> Dict[str, Any]:
        # Imported here to keep the schemas -> gateway dependency direction
        # one-way (gateway imports schemas at module level already).
        from gpustack.gateway.utils import (
            model_ai_proxy_provider_id,
            provider_registry_name,
        )

        catalogue_id = (
            provider_registry_name(self.providerId)
            if self.providerId is not None
            else model_ai_proxy_provider_id(self.modelId)
        )
        rule: Dict[str, Any] = {
            "enabled": self.enabled,
            "activeProviderId": catalogue_id,
            "decisionModel": self.decisionModel,
        }
        if self.modelSelection is not None:
            rule["modelSelection"] = self.modelSelection.to_gateway()
        if self.weight is not None:
            rule["rankWeight"] = self.weight
        return rule
