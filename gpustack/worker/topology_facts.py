"""What a worker can say about where it is, from what its devices report.

One source, read on the worker and static for as long as nobody recables the
host: the **per-device hints** the runtime attached to each GPU
(`topology_hints`) naming the NVLink domain the card is in. Eight cards
agreeing on one domain is one fact about the host.

The result is keyed by topology label key and read *under* the worker's own
labels, so a hand-filled value always wins.

The fold is over whatever keys the hints carry, so a runtime that learns to
report another kind of domain needs nothing here beyond naming the key.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

NVIDIA_CLIQUE_KEY = "nvidia.com/gpu.clique"

_DOMAIN_KEYS = (NVIDIA_CLIQUE_KEY,)


def facts_from_devices(devices: Iterable) -> Dict[str, str]:
    """Fold per-device hints into host-level facts.

    A domain is claimed only when every card that reports one reports the
    same; disagreement means the host straddles domains, which the model does
    not represent, so nothing is claimed and the disagreement is logged.
    """
    hints: List[dict] = [
        getattr(d, "topology_hints", None) or {} for d in devices or []
    ]
    out: Dict[str, str] = {}

    for key in _DOMAIN_KEYS:
        values = {h[key] for h in hints if h.get(key)}
        if len(values) == 1:
            out[key] = values.pop()
        elif len(values) > 1:
            logger.warning(
                "Devices on this worker report different %s values (%s); "
                "not claiming a domain for the host.",
                key,
                ", ".join(sorted(values)),
            )

    return out
