import json

import pytest
from pydantic import ValidationError

from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderL2Backend,
    CacheProviderL2Field,
    render_l2_adapter,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"type": "string"},
        {"type": "boolean", "env_name": "CACHE_OPTION"},
        {"type": "boolean", "metrics_target": True},
    ],
)
def test_adapter_options_require_a_boolean_json_field(settings):
    with pytest.raises(ValidationError, match="requires a boolean field"):
        CacheProviderL2Field(
            name="enabled",
            enabled_adapter_options={"option": {"policy": "LRU"}},
            **settings,
        )


@pytest.mark.parametrize(
    "value, enabled", [(True, True), (False, False), ("false", False)]
)
def test_enabled_adapter_options_remain_outside_backend_params(value, enabled):
    provider = CacheProvider(
        name="example",
        custom_version=True,
        l2_adapter_flag="--l2-adapter",
        l2_backends={
            "store": CacheProviderL2Backend(
                adapter_backend="POSIX",
                adapter_params={"path": "path"},
                fields=[
                    CacheProviderL2Field(name="path"),
                    CacheProviderL2Field(
                        name="auto_cleanup",
                        type="boolean",
                        enabled_adapter_options={
                            "eviction": {"eviction_policy": "LRU"}
                        },
                    ),
                ],
            )
        },
    )

    args, env = render_l2_adapter(
        provider, "store", {"path": "/data/l2", "auto_cleanup": value}
    )

    expected = {
        "type": "store",
        "backend": "POSIX",
        "backend_params": {"path": "/data/l2"},
    }
    if enabled:
        expected["eviction"] = {"eviction_policy": "LRU"}
    assert json.loads(args[1]) == expected
    assert env == {}
