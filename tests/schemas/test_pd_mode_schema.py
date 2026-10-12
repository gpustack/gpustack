import pytest
from pydantic import ValidationError

from gpustack.schemas.pd_modes import PDMode


@pytest.mark.parametrize("default", [None, "", "unknown"])
def test_connector_variants_require_a_declared_default(default):
    with pytest.raises(ValidationError, match="default_connector_variant"):
        PDMode(
            name="example",
            connector_variants={"standard": "Standard", "hybrid": "Hybrid"},
            default_connector_variant=default,
        )


@pytest.mark.parametrize("default", ["standard", "hybrid"])
def test_connector_variants_accept_a_declared_default(default):
    mode = PDMode(
        name="example",
        connector_variants={"standard": "Standard", "hybrid": "Hybrid"},
        default_connector_variant=default,
    )
    assert mode.connector_variants[mode.default_connector_variant]


@pytest.mark.parametrize(
    "declared,provider,role,reason",
    [
        (None, "Other", "decode", None),
        ([], "LMCache", "prefill", "does not support cache provider 'LMCache'"),
        (["LMCache"], "LMCache", "prefill", None),
        (["LMCache"], "LMCache", None, None),
        (["LMCache"], "LMCache", "decode", "does not support LMCache on role 'decode'"),
        (["LMCache"], "Other", "prefill", "does not support cache provider 'Other'"),
    ],
)
def test_cache_provider_and_role_eligibility(declared, provider, role, reason):
    mode = PDMode(
        name="example",
        roles={"prefill": {}, "decode": {}},
        cache_services=(
            [
                {"provider": name, "roles": {"prefill": {"kv_role": "kv_both"}}}
                for name in declared
            ]
            if declared is not None
            else None
        ),
    )

    actual = mode.cache_service_reason(provider, role)
    assert actual == (f"PD mode 'example' {reason}" if reason else None)


@pytest.mark.parametrize(
    "provider,role,kv_role",
    [
        ("LMCache", "prefill", "kv_both"),
        ("LMCache", "decode", None),
        ("LMCache", None, None),
        ("Other", "prefill", None),
    ],
)
def test_store_connector_settings_are_role_specific(provider, role, kv_role):
    mode = PDMode(
        name="example",
        roles={"prefill": {}, "decode": {}},
        cache_services=[
            {"provider": "LMCache", "roles": {"prefill": {"kv_role": "kv_both"}}}
        ],
    )

    settings = mode.cache_service_role(provider, role)
    assert (settings.kv_role if settings else None) == kv_role
