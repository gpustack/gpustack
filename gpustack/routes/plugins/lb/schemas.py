"""The LB route plugin defines no payload shapes of its own.

LB is a base capability: its route-level configuration lives in
``ModelRoute.meta["lb"]`` (validated through the ``plugins.lb``
section) and its per-target knob — ``max_running_requests`` — is a
first-class column on ``model_route_targets``, edited like any other
target field. The
separately shipped capability plugins (gpustack-lb-session-affinity,
gpustack-lb-least-load) are the real external-plugin pattern.
"""
