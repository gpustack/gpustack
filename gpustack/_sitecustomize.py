"""
This module adds a sitecustomize hook to Ray to allow setting the current
placement group from an environment variable.
"""

import json
import os

try:
    from ray.util import (
        get_current_placement_group as original_get_current_placement_group,
        placement_group,
    )

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    original_get_current_placement_group = None
    placement_group = None

_cached_pg = None

pg_bundles_spec = os.environ.get("GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES")
pg_ready_timeout = float(
    os.environ.get("GPUSTACK_RAY_PLACEMENT_GROUP_READY_TIMEOUT", 30)
)


def bundles_env_aware_get_current_placement_group():
    if not RAY_AVAILABLE:
        return None

    global _cached_pg
    RAY_PG_BUNDLES_ENV = "GPUSTACK_RAY_PLACEMENT_GROUP_BUNDLES"
    if pg_bundles_spec:
        if _cached_pg is None:
            try:
                _cached_pg = placement_group(json.loads(pg_bundles_spec))
                ray.get(_cached_pg.ready(), timeout=pg_ready_timeout)
            except ray.exceptions.GetTimeoutError:
                raise ValueError(
                    f"Timeout while waiting for placement group {_cached_pg.id} to be ready. "
                    f"Run `ray get placement-groups {_cached_pg.id.hex()}` to check the status."
                )
            except Exception as e:
                raise ValueError(
                    f"Fail to create placement group from {RAY_PG_BUNDLES_ENV} environment variable: {e}"
                )

        return _cached_pg

    return original_get_current_placement_group()


if RAY_AVAILABLE:
    import ray.util

    ray.util.get_current_placement_group = bundles_env_aware_get_current_placement_group
