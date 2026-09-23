"""The least-load capability plugin — full external plugin pattern.

Configuration under ``plugins["least-load"]``, stored in
``route.meta["least-load"]``. Gateway presence is one CR,
``gpustack-lb-least-load`` (AUTHN/330): scores
``1/(1+inflight+penalty)`` over the candidate set the LB context role
publishes — in-flight and health state live in the LB plugin itself,
this one only reads what context publishes.
"""

from gpustack.routes.plugins.least_load.plugin import (  # noqa: F401
    LeastLoadPlugin,
    least_load_plugin,
)
