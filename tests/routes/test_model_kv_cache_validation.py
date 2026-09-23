"""Shared extended-KV-cache validation on model create/update.

``validate_shared_kv_cache`` gates a model's attachment to a cache
service: the service must exist in the model's Org (cross-tenant ids read
as missing), sit in the model's cluster, and have a provider compatible
with the model's backend; local mode must not carry a service id.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.routes import models as models_route
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderIntegration,
)
from gpustack.schemas.models import (
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    RoleSpec,
)

OWNER_PRINCIPAL = 42


def _fake_lookup(provider):
    """Stand in for the catalog lookup, which reads a table: a coroutine taking
    the session its caller holds."""

    async def lookup(_session, name=None):
        if name is None or provider is None:
            return provider
        return provider if name.lower() == provider.name.lower() else None

    return lookup


def _model_in(ext, backend=None, distributed=False, backend_version=None, roles=None):
    return SimpleNamespace(
        extended_kv_cache=ext,
        backend=backend,
        distributed_inference_across_workers=distributed,
        backend_version=backend_version,
        roles=roles,
    )


def _shared_ext(cache_service_id=9):
    return ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=cache_service_id
    )


def _service(owner_principal_id=OWNER_PRINCIPAL, cluster_id=1):
    return SimpleNamespace(
        id=9,
        deleted_at=None,
        owner_principal_id=owner_principal_id,
        cluster_id=cluster_id,
        provider_name="LMCache",
    )


def _worker(framework=None):
    devices = [SimpleNamespace(type=framework)] if framework else []
    return SimpleNamespace(status=SimpleNamespace(gpu_devices=devices))


def _patch_lookups(
    monkeypatch,
    service,
    provider_backends=("vLLM",),
    frameworks=None,
    workers=(),
    versions=None,
):

    monkeypatch.setattr(
        models_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    provider = CacheProvider(
        name="LMCache",
        custom_version=True,
        inference_backend_integrations=[
            CacheProviderIntegration(
                backend=b, frameworks=frameworks, versions=versions
            )
            for b in provider_backends
        ],
    )
    monkeypatch.setattr(models_route, "get_cache_provider", _fake_lookup(provider))
    monkeypatch.setattr(
        models_route.Worker, "all_by_fields", AsyncMock(return_value=list(workers))
    )


@pytest.mark.asyncio
async def test_disabled_config_passes():
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(ExtendedKVCacheConfig(enabled=False)),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_requires_cache_service_id():
    ext = ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.SHARED)
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
        )

    assert "cache_service_id is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_local_rejects_cache_service_id():
    ext = ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.LOCAL, cache_service_id=9
    )
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
        )

    assert "only valid when mode is 'shared'" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_denies_cross_org_service(monkeypatch):
    _patch_lookups(monkeypatch, _service(owner_principal_id=999))

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )


@pytest.mark.asyncio
async def test_shared_denies_missing_service(monkeypatch):
    _patch_lookups(monkeypatch, None)

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )


@pytest.mark.asyncio
async def test_shared_denies_cross_cluster_service(monkeypatch):
    _patch_lookups(monkeypatch, _service(cluster_id=2))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )

    assert "same cluster" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_denies_incompatible_backend(monkeypatch):
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(_shared_ext(), backend="SGLang"),
            OWNER_PRINCIPAL,
            1,
        )

    assert "not compatible" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_defaults_backend_to_vllm(monkeypatch):
    """A model without an explicit backend deploys on vLLM, so
    compatibility is checked against vLLM."""
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_happy_path_vllm(monkeypatch):
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), backend="vLLM"),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_skips_cluster_check_without_effective_cluster(monkeypatch):
    """No chosen cluster means default-cluster resolution runs later; the
    cluster match can't be validated here and must not false-positive."""
    _patch_lookups(monkeypatch, _service(cluster_id=2), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, None
    )


@pytest.mark.asyncio
async def test_shared_allows_distributed_permission_flag(monkeypatch):
    """distributed_inference_across_workers defaults to True for vLLM and
    SGLang as a permission, not a placement plan — creation must not
    reject it; the node-local incompatibility is decided at scheduling,
    where the real placement is known (see the injection resolver)."""
    _patch_lookups(monkeypatch, _service())
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), distributed=True),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_shared_rejects_cluster_without_supported_accelerator(monkeypatch):
    """A cluster whose accelerators are all outside the provider's
    framework-scoped integrations would degrade on every instance —
    caught at creation instead."""
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker("cann"), _worker()],
    )
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
        )

    assert "cann" in exc_info.value.message


@pytest.mark.asyncio
async def test_shared_passes_with_one_supported_accelerator(monkeypatch):
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker("cann"), _worker("cuda")],
    )
    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_skips_accelerator_precheck_without_gpu_workers(monkeypatch):
    """Accelerator-less clusters are left to scheduling — the pre-check
    only fires when the cluster has detectable accelerators."""
    _patch_lookups(
        monkeypatch,
        _service(),
        frameworks=["cuda"],
        workers=[_worker()],
    )
    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(_shared_ext()), OWNER_PRINCIPAL, 1
    )


@pytest.mark.asyncio
async def test_shared_rejects_pinned_version_below_floor(monkeypatch):
    """A pinned engine version below the integration's floor would crash
    on injected args the engine does not accept — rejected at creation."""
    _patch_lookups(monkeypatch, _service(), versions=">=0.25.0")
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(_shared_ext(), backend_version="0.24.1"),
            OWNER_PRINCIPAL,
            1,
        )

    assert ">=0.25.0" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_version",
    [
        "0.25.0",
        "0.26.1",
        # unparseable pins fail open — enforcement must never block an
        # exotic version string
        "custom-build",
        None,
    ],
)
async def test_shared_allows_version_in_range_or_unknown(monkeypatch, backend_version):
    _patch_lookups(monkeypatch, _service(), versions=">=0.25.0")
    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(_shared_ext(), backend_version=backend_version),
        OWNER_PRINCIPAL,
        1,
    )


# --- the same rules, written one level down ------------------------------- #
#
# `extended_kv_cache` is a per-role override, and until these tests existed
# every rule above applied to the Model's value alone. The identical
# configuration written under `roles[].extended_kv_cache` was accepted with a
# 200 in every case — measured across two e2e rounds, seven configurations,
# seven acceptances. That it was an omission rather than an exemption is not a
# guess: `_reject_cache_under_a_hand_written_mode`, the one role-aware check in
# this area, does refuse a role-level `enabled`.


def _role(name, ext=None, backend=None, backend_version=None):
    return RoleSpec(
        name=name,
        extended_kv_cache=ext,
        backend=backend,
        backend_version=backend_version,
    )


@pytest.mark.asyncio
async def test_a_role_naming_a_service_that_does_not_exist_is_refused(monkeypatch):
    _patch_lookups(monkeypatch, None)

    with pytest.raises(NotFoundException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(None, roles=[_role("prefill", _shared_ext()), _role("decode")]),
            OWNER_PRINCIPAL,
            1,
        )

    assert "prefill" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_role_asking_for_local_mode_with_a_service_id_is_refused():
    ext = ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.LOCAL, cache_service_id=9
    )
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(None, roles=[_role("decode", ext)]),
            OWNER_PRINCIPAL,
            1,
        )

    assert "only valid when mode is 'shared'" in exc_info.value.message
    assert "decode" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_role_asking_for_shared_mode_without_a_service_id_is_refused():
    ext = ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.SHARED)
    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(None, roles=[_role("prefill", ext)]),
            OWNER_PRINCIPAL,
            1,
        )

    assert "cache_service_id is required" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_role_reaching_into_another_cluster_is_refused(monkeypatch):
    _patch_lookups(monkeypatch, _service(cluster_id=2))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(None, roles=[_role("prefill", _shared_ext())]),
            OWNER_PRINCIPAL,
            1,
        )

    assert "same cluster" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_role_reaching_into_another_tenant_is_refused(monkeypatch):
    _patch_lookups(monkeypatch, _service(owner_principal_id=999))

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(None, roles=[_role("prefill", _shared_ext())]),
            OWNER_PRINCIPAL,
            1,
        )


@pytest.mark.asyncio
async def test_a_role_pinning_a_version_below_the_floor_is_refused(monkeypatch):
    """The version is a per-role override too, so the floor has to be applied
    to the role's pin and not only to the model's — a group whose model-level
    pin is fine still starts one member on the build that cannot read the
    injected arguments."""
    _patch_lookups(monkeypatch, _service(), versions=">=0.25.0")

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(
                _shared_ext(),
                backend_version="0.26.0",
                roles=[_role("prefill"), _role("decode", backend_version="0.24.1")],
            ),
            OWNER_PRINCIPAL,
            1,
        )

    assert "0.24.1" in exc_info.value.message
    assert "decode" in exc_info.value.message


@pytest.mark.asyncio
async def test_a_role_switching_to_an_unsupported_engine_is_refused(monkeypatch):
    """Only reachable under pd mode 'custom', which is the one mode that lets a
    group mix engines — and the one where the cache is refused outright. Pinned
    anyway because the provider is matched against whatever engine the member
    actually runs, not against the deployment's."""
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(
                _shared_ext(),
                backend="vLLM",
                roles=[_role("prefill"), _role("decode", backend="SGLang")],
            ),
            OWNER_PRINCIPAL,
            1,
        )

    assert "not compatible" in exc_info.value.message


@pytest.mark.asyncio
async def test_roles_inheriting_the_models_cache_are_checked_once(monkeypatch):
    """The common shape, and it must not start reporting refusals against a
    role name for a value written at the model level: three roles inheriting
    one configuration on one engine are one configuration."""
    one_by_id = AsyncMock(return_value=_service())
    _patch_lookups(monkeypatch, _service())
    monkeypatch.setattr(models_route.CacheService, "one_by_id", one_by_id)

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(
            _shared_ext(),
            backend="vLLM",
            roles=[_role("prefill"), _role("decode"), _role("router")],
        ),
        OWNER_PRINCIPAL,
        1,
    )

    assert one_by_id.await_count == 1


@pytest.mark.asyncio
async def test_a_role_level_cache_on_a_valid_service_passes(monkeypatch):
    _patch_lookups(monkeypatch, _service(), provider_backends=("vLLM",))

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(
            None,
            backend="vLLM",
            roles=[_role("prefill", _shared_ext()), _role("decode")],
        ),
        OWNER_PRINCIPAL,
        1,
    )


# --- one deployment, one pool --------------------------------------------- #


@pytest.mark.asyncio
async def test_two_roles_naming_two_cache_services_are_refused(monkeypatch):
    """The identity of the service is a property of the deployment, not of the
    member: two services are two pools, so a prefix prefill stored is never the
    one decode looks for, and the only symptom is a hit rate that never
    arrives."""
    _patch_lookups(monkeypatch, _service())

    with pytest.raises(BadRequestException) as exc_info:
        await models_route.validate_shared_kv_cache(
            MagicMock(),
            _model_in(
                None,
                roles=[
                    _role("prefill", _shared_ext(cache_service_id=9)),
                    _role("decode", _shared_ext(cache_service_id=10)),
                ],
            ),
            OWNER_PRINCIPAL,
            1,
        )

    assert "different cache services" in exc_info.value.message


@pytest.mark.asyncio
async def test_one_side_taking_a_cache_and_the_other_none_is_left_alone(monkeypatch):
    """Deliberately NOT refused here. Whether a one-sided cache is usable is
    a question about the connector's compatibility hash rather than about
    intent, it is being settled elsewhere, and this rule is written so it cannot
    pre-empt that answer: it needs two roles that are both shared-enabled and
    can never fire on a one-sided configuration."""
    _patch_lookups(monkeypatch, _service())

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(
            None,
            roles=[_role("prefill", _shared_ext()), _role("decode")],
        ),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_roles_naming_the_same_service_are_one_pool(monkeypatch):
    _patch_lookups(monkeypatch, _service())

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(
            None,
            roles=[
                _role("prefill", _shared_ext(cache_service_id=9)),
                _role("decode", _shared_ext(cache_service_id=9)),
            ],
        ),
        OWNER_PRINCIPAL,
        1,
    )


@pytest.mark.asyncio
async def test_a_shared_declaration_loses_its_local_sizing(monkeypatch):
    """`ram_size` / `ram_ratio` size the cache the engine offloads into host
    memory, and a shared deployment has none — the cache service holds it.
    `ram_ratio` defaults to 1.2, so leaving it set would book 1.2x the VRAM
    claim in host memory that no process would take."""
    _patch_lookups(monkeypatch, _service())
    ext = _shared_ext()
    assert ext.ram_ratio == 1.2

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
    )

    assert ext.ram_ratio is None
    assert ext.ram_size is None


@pytest.mark.asyncio
async def test_every_role_of_a_group_is_cleared(monkeypatch):
    """A role-level cache is a separate object from the model's, so clearing
    one says nothing about the others — and a group is where the booking
    multiplied."""
    _patch_lookups(monkeypatch, _service())
    prefill = _shared_ext(cache_service_id=9)
    decode = _shared_ext(cache_service_id=9)
    decode.ram_size = 16

    await models_route.validate_shared_kv_cache(
        MagicMock(),
        _model_in(None, roles=[_role("prefill", prefill), _role("decode", decode)]),
        OWNER_PRINCIPAL,
        1,
    )

    assert (prefill.ram_ratio, prefill.ram_size) == (None, None)
    assert (decode.ram_ratio, decode.ram_size) == (None, None)


@pytest.mark.asyncio
async def test_a_local_declaration_keeps_its_sizing(monkeypatch):
    """The knobs mean exactly what they say in local mode. Clearing them there
    would silently turn a sized offload into an unsized one."""
    ext = ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.LOCAL, ram_size=16, ram_ratio=2.0
    )

    await models_route.validate_shared_kv_cache(
        MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
    )

    assert ext.ram_size == 16
    assert ext.ram_ratio == 2.0


@pytest.mark.asyncio
async def test_clearing_happens_after_the_service_is_accepted(monkeypatch):
    """A rejected declaration is not normalized. The request never takes
    effect, and a half-edited body handed back to a caller retrying against a
    fixed service id would drop a value they had set."""
    _patch_lookups(monkeypatch, _service(owner_principal_id=999))
    ext = _shared_ext()

    with pytest.raises(NotFoundException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), _model_in(ext), OWNER_PRINCIPAL, 1
        )

    assert ext.ram_ratio == 1.2
