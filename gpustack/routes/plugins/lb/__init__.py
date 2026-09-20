"""The LB route plugin — the base routing capability.

Storage rides the route's own rows: the policy section in
``ModelRoute.meta["lb"]`` and the per-target knob
(``max_running_requests``) as a first-class
``model_route_targets`` column.

Gateway presence: two WasmPlugin CRs from the ``gpustack-lb`` module —
the context role replaces ``gpustack-model-mapper`` in place, the
finisher role is the new ``gpustack-lb`` CR (see gateway.py); per-route
candidates/modelMappers and the cluster_header EnvoyFilter are written
by reconciler.sync_model_route_lb from the route reconcile path. The
capability plugins (session-affinity, least-load) are separate route
plugins of their own; the band conventions they share with LB live in
capability.py.
"""

from gpustack.routes.plugins.lb.config import (  # noqa: F401
    LBHealthConfig,
    LBPolicyConfig,
    LBRejectConfig,
    lb_policy_from_meta,
)
from gpustack.routes.plugins.lb.plugin import LBPlugin, lb_plugin  # noqa: F401
