"""The decision-service capability plugin — Jev model-selection routing.

Configuration under ``plugins["decision-service"]``, stored in the shared
capability policy table (capability = ``decision-service``). Gateway presence is
one CR, ``gpustack-lb-decision-service`` (AUTHN/328): for each task request it asks
the configured Jev decision service which candidate model should serve and
publishes a rank entry the gpustack-lb finisher combines with the other
capability opinions. The decision services themselves are
``gpustack-lb-typesafe`` (TypeSafe) ModelProviders, rendered into the CR's
``providers`` catalogue by ``providers.sync_decision_service_providers``.
"""

from gpustack.routes.plugins.decision_service.plugin import (  # noqa: F401
    DecisionServicePlugin,
    decision_service_plugin,
)
