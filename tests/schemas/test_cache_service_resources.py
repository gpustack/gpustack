import pytest

from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.cache_service_resources import cache_service_resource_claim
from gpustack.schemas.cache_services import CacheService, CacheServiceConfig


def provider(**kwargs):
    return CacheProvider(
        name="cache",
        supported_modes=["managed"],
        default_version="v1",
        versions={"v1": {"image": "cache:1", "run_args": "--ram {{ram}}"}},
        fields=[{"name": "ram", "type": "number", "default": 4}],
        **kwargs,
    )


def service(**config):
    return CacheService(
        name="cache",
        provider_name="cache",
        cluster_id=1,
        config=CacheServiceConfig(**config),
    )


@pytest.mark.parametrize(
    "value,expected", [(0, 0), (1.5, 1610612736), ("0.000000001", 2)]
)
def test_ram_is_resolved_in_bytes_and_rounded_up(value, expected):
    assert cache_service_resource_claim(
        service(fields={"ram": value}),
        provider(resource_profile={"ram_gib": "{{ram}}"}),
        "",
    ) == {"ram": expected}


def test_default_capacity_and_undeclared_capacity_are_distinct():
    assert cache_service_resource_claim(service(), provider(), "") is None
    assert cache_service_resource_claim(
        service(), provider(resource_profile={"ram_gib": "{{ram}}"}), ""
    ) == {"ram": 4 * 1024**3}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "{{missing}}", ""])
def test_invalid_capacity_cannot_become_a_zero_reservation(value):
    with pytest.raises(ValueError):
        cache_service_resource_claim(
            service(fields={"ram": value}),
            provider(resource_profile={"ram_gib": "{{ram}}"}),
            "",
        )


@pytest.mark.parametrize("parameters", [["--ram", "16"], ["--ram=16"]])
def test_capacity_flags_cannot_override_the_declared_reservation(parameters):
    with pytest.raises(ValueError, match="extra parameters"):
        cache_service_resource_claim(
            service(parameters={"": parameters}),
            provider(resource_profile={"ram_gib": "{{ram}}"}),
            "",
        )


def test_components_do_not_inherit_the_provider_capacity():
    p = provider(
        resource_profile={"ram_gib": "100"},
        components={
            "master": {"topology": "replicas", "replicas": 1, "attach_endpoint": True},
            "store": {
                "topology": "per_node",
                "resource_profile": {"ram_gib": "{{ram}}"},
            },
        },
    )
    assert cache_service_resource_claim(service(), p, "master") is None
    assert cache_service_resource_claim(service(), p, "store") == {"ram": 4 * 1024**3}


@pytest.mark.parametrize("run_args", ["{{ram}}", "serve {{ram}}"])
def test_positional_capacity_does_not_turn_other_arguments_into_flags(run_args):
    p = provider(resource_profile={"ram_gib": "{{ram}}"})
    p.versions["v1"].run_args = run_args
    assert cache_service_resource_claim(
        service(parameters={"": ["--label", "serve", "--template", "{{ram}}"]}),
        p,
        "",
    ) == {"ram": 4 * 1024**3}


@pytest.mark.parametrize("run_args", ["--ram {{ram}}", "--ram={{ram}}"])
def test_named_capacity_flags_remain_protected(run_args):
    p = provider(resource_profile={"ram_gib": "{{ram}}"})
    p.versions["v1"].run_args = run_args
    with pytest.raises(ValueError, match="extra parameters"):
        cache_service_resource_claim(service(parameters={"": ["--ram=8"]}), p, "")


def test_invalid_launch_template_identifies_the_provider():
    p = provider(resource_profile={"ram_gib": "{{ram}}"})
    p.versions["v1"].run_args = '--ram "{{ram}}'
    with pytest.raises(
        ValueError, match="Invalid launch template for cache provider 'cache'"
    ):
        cache_service_resource_claim(service(), p, "")
