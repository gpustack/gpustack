"""The session-affinity capability plugin — the first "full external
plugin pattern" consumer of the route-plugin framework.

Its configuration lives under ``plugins["session-affinity"]`` on the
route payload and is stored in ``route.meta["session-affinity"]`` (the
plugin's own key — nothing writes it but this plugin). Its gateway
presence is one CR, ``gpustack-lb-session-affinity`` (AUTHN/780), whose
matchRule per route carries the session key chain — the plugin has no
safe default for it and does not inherit it, so every rule must carry
its own copy.
"""

from gpustack.routes.plugins.session_affinity.plugin import (  # noqa: F401
    SessionAffinityPlugin,
    session_affinity_plugin,
)
