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
from gpustack.routes import (
    cache_services as cache_services_route,
    models as models_route,
)
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
        provider_version="0.5.5",
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
    provider_name="LMCache",
):

    monkeypatch.setattr(
        models_route.CacheService, "one_by_id", AsyncMock(return_value=service)
    )
    provider = CacheProvider(
        name=provider_name,
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
    """Without an effective model cluster, validation skips the cluster match
    rather than inferring placement from the cache service's cluster."""
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
    """Without PD, roles inheriting identical cache/backend settings share
    one validation lookup; routers do not inherit the engine cache."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,version",
    [
        ("vllm-nixl", "0.5.4"),
        ("vllm-nixl", "0.5.5"),
        ("vllm-mooncake", "0.5.5"),
        ("vllm-ascend-mooncake", "0.5.5"),
    ],
)
async def test_pd_decode_lmcache_provider_is_not_supported(monkeypatch, mode, version):
    from gpustack.schemas.models import DisaggregationSpec

    service = _service()
    service.provider_version = version
    _patch_lookups(monkeypatch, service)
    model = _model_in(
        None,
        backend="vLLM",
        backend_version="0.29.0",
        roles=[_role("decode", _shared_ext())],
    )
    model.disaggregation = DisaggregationSpec(mode=mode)
    with pytest.raises(BadRequestException):
        await models_route.validate_shared_kv_cache(
            MagicMock(), model, OWNER_PRINCIPAL, 1
        )


@pytest.mark.asyncio
async def test_inherited_pd_cache_cannot_attach_lmcache_to_decode(monkeypatch):
    from gpustack.schemas.models import DisaggregationSpec

    service = _service()
    service.provider_version = "0.5.4"
    _patch_lookups(monkeypatch, service)
    model = _model_in(
        _shared_ext(),
        backend="vLLM",
        backend_version="0.29.0",
        roles=[_role("prefill"), _role("decode")],
    )
    model.disaggregation = DisaggregationSpec(mode="vllm-nixl")
    with pytest.raises(BadRequestException) as error:
        await models_route.validate_shared_kv_cache(
            MagicMock(), model, OWNER_PRINCIPAL, 1
        )
    assert "decode" in error.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,provider,version",
    [
        ("vllm-nixl", "LMCache", "0.24.1"),
        ("vllm-nixl", "LMCache", "0.25.3"),
        ("vllm-mooncake", "Mooncake", "0.27.1"),
        ("vllm-ascend-mooncake", "Mooncake", "0.20.2"),
        ("vllm-nixl", "LMCache", "0.24.1-custom"),
    ],
)
@pytest.mark.parametrize("cache_mode", [KVCacheModeEnum.LOCAL, KVCacheModeEnum.SHARED])
async def test_pd_explicit_engine_versions_do_not_block_cache_admission(
    monkeypatch, mode, provider, version, cache_mode
):
    from gpustack.schemas.models import DisaggregationSpec

    service = _service()
    service.provider_name = provider
    _patch_lookups(monkeypatch, service, versions=">=0.29.0", provider_name=provider)
    cache = (
        _shared_ext()
        if cache_mode == KVCacheModeEnum.SHARED
        else ExtendedKVCacheConfig(enabled=True)
    )
    model = _model_in(
        None,
        backend="vLLM",
        backend_version=version,
        roles=[
            _role("prefill", cache),
            _role("decode", cache if cache_mode == KVCacheModeEnum.LOCAL else None),
        ],
    )
    model.disaggregation = DisaggregationSpec(mode=mode)

    await models_route.validate_shared_kv_cache(MagicMock(), model, OWNER_PRINCIPAL, 1)
    assert cache.enabled


def _stored_pd_model(mode="vllm-nixl", ext=None):
    from gpustack.schemas.models import Model, DisaggregationSpec, SourceEnum

    return Model(
        id=1,
        name="pd",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/model",
        backend="vLLM",
        cluster_id=1,
        owner_principal_id=OWNER_PRINCIPAL,
        roles=[
            RoleSpec(name="prefill"),
            RoleSpec(name="decode"),
            RoleSpec(name="router"),
        ],
        disaggregation=DisaggregationSpec(mode=mode),
        extended_kv_cache=ext,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cache", "mode", "roles"])
async def test_sparse_pd_updates_validate_effective_cache_policy(monkeypatch, change):
    from gpustack.schemas.models import ModelUpdate, DisaggregationSpec

    _patch_lookups(monkeypatch, _service())
    stored = _stored_pd_model("vllm-mooncake")
    changes = {"extended_kv_cache": _shared_ext()}
    if change == "mode":
        stored.disaggregation = DisaggregationSpec(mode="custom")
        stored.extended_kv_cache = _shared_ext()
        changes = {"disaggregation": DisaggregationSpec(mode="vllm-mooncake")}
    elif change == "roles":
        changes = {"roles": [RoleSpec(name="decode", extended_kv_cache=_shared_ext())]}
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        **changes,
    )
    with pytest.raises(BadRequestException) as error:
        await models_route.validate_shared_kv_cache(
            MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
        )
    assert "does not support" in error.value.message


@pytest.mark.asyncio
async def test_sparse_cache_validation_preserves_unsubmitted_fields(monkeypatch):
    from gpustack.schemas.models import ModelUpdate

    _patch_lookups(monkeypatch, _service())
    ext = _shared_ext()
    ext.ram_ratio = 2.0
    stored = _stored_pd_model("custom", ext)
    before = stored.model_dump()
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        description="updated",
    )
    fields = update.model_fields_set.copy()
    await models_route.validate_shared_kv_cache(
        MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
    )
    assert update.model_fields_set == fields
    assert stored.model_dump() == before
    assert update.extended_kv_cache is None


@pytest.mark.asyncio
async def test_explicit_cache_removal_does_not_validate_the_stored_cache(monkeypatch):
    from gpustack.schemas.models import ModelUpdate

    _patch_lookups(monkeypatch, _service())
    stored = _stored_pd_model("custom", _shared_ext())
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        extended_kv_cache=None,
    )
    await models_route.validate_shared_kv_cache(
        MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
    )
    models_route.CacheService.one_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_sparse_shared_cache_update_preserves_explicit_role_local_cache_sizes(
    monkeypatch,
):
    from gpustack.schemas.models import ModelUpdate

    _patch_lookups(monkeypatch, _service())
    local = ExtendedKVCacheConfig(enabled=True, ram_ratio=2.0, ram_size=16)
    stored = _stored_pd_model(ext=ExtendedKVCacheConfig(enabled=True))
    for role in stored.roles:
        if role.name in ("prefill", "decode"):
            role.extended_kv_cache = local.model_copy()
        else:
            role.extended_kv_cache = ExtendedKVCacheConfig(enabled=False)
    before = stored.model_dump()
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        extended_kv_cache=_shared_ext(),
    )
    fields = update.model_fields_set.copy()
    await models_route.validate_shared_kv_cache(
        MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
    )
    assert stored.model_dump() == before
    assert update.model_fields_set == fields
    assert update.roles is None
    assert update.extended_kv_cache.ram_ratio is None
    assert update.extended_kv_cache.ram_size is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,provider,backend",
    [("vllm-mooncake", "Mooncake", "vLLM"), ("sglang-mooncake", "LMCache", "SGLang")],
)
@pytest.mark.parametrize("action", ["create", "description", "stop"])
async def test_router_does_not_inherit_shared_cache_for_admission(
    monkeypatch, mode, provider, backend, action
):
    from gpustack.schemas.models import ModelUpdate

    service = _service()
    service.provider_name = provider
    _patch_lookups(
        monkeypatch, service, provider_backends=(backend,), provider_name=provider
    )
    stored = _stored_pd_model(mode, _shared_ext())
    stored.backend = backend
    if action == "create":
        model_in, previous = stored, None
    else:
        changes = (
            {"description": "updated"} if action == "description" else {"replicas": 0}
        )
        model_in = ModelUpdate(
            name=stored.name,
            source=stored.source,
            huggingface_repo_id=stored.huggingface_repo_id,
            **changes,
        )
        previous = stored
    await models_route.validate_shared_kv_cache(
        MagicMock(), model_in, OWNER_PRINCIPAL, 1, stored=previous
    )


@pytest.mark.asyncio
async def test_router_explicit_shared_cache_is_still_rejected(monkeypatch):
    from gpustack.schemas.models import DisaggregationSpec

    service = _service()
    service.provider_name = "Mooncake"
    _patch_lookups(monkeypatch, service, provider_name="Mooncake")
    model = _model_in(
        None, roles=[_role("prefill"), _role("decode"), _role("router", _shared_ext())]
    )
    model.disaggregation = DisaggregationSpec(mode="vllm-mooncake")
    with pytest.raises(BadRequestException) as refused:
        await models_route.validate_shared_kv_cache(
            MagicMock(), model, OWNER_PRINCIPAL, 1
        )
    assert "router" in refused.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_mode", ["disabled", "local", "shared"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("action", ["description", "stop"])
async def test_overridden_cache_can_be_deleted_without_blocking_updates(
    monkeypatch, cache_mode, full, action
):
    from gpustack.schemas.models import ModelUpdate

    stored = _stored_pd_model("vllm-mooncake", _shared_ext())
    override = (
        _shared_ext(cache_service_id=10)
        if cache_mode == "shared"
        else ExtendedKVCacheConfig(enabled=cache_mode == "local")
    )
    for role in stored.roles:
        if role.name != "router":
            role.extended_kv_cache = override.model_copy()
    monkeypatch.setattr(
        models_route.Model, "all_by_fields", AsyncMock(return_value=[stored])
    )
    assert (
        await cache_services_route._models_referencing_cache_service(
            MagicMock(), _service()
        )
        == []
    )

    service = _service()
    service.id = 10
    service.provider_name = "Mooncake"
    _patch_lookups(monkeypatch, service, provider_name="Mooncake")
    lookup = AsyncMock(side_effect=lambda session, id: service if id == 10 else None)
    monkeypatch.setattr(models_route.CacheService, "one_by_id", lookup)
    fields = (
        stored.model_dump()
        if full
        else {
            "name": stored.name,
            "source": stored.source,
            "huggingface_repo_id": stored.huggingface_repo_id,
        }
    )
    fields.update(
        {"description": "updated"} if action == "description" else {"replicas": 0}
    )
    update = ModelUpdate.model_validate(fields)
    before = stored.model_dump()
    submitted = update.model_fields_set.copy()

    await models_route.validate_shared_kv_cache(
        MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
    )

    assert stored.model_dump() == before
    assert update.model_fields_set == submitted
    assert all(call.args[1] == 10 for call in lookup.await_args_list)
    if cache_mode != "shared":
        lookup.assert_not_awaited()
    if full:
        assert update.extended_kv_cache.ram_ratio is None


@pytest.mark.asyncio
@pytest.mark.parametrize("inherit", [False, True])
async def test_active_cache_references_are_protected_and_validated(
    monkeypatch, inherit
):
    from gpustack.schemas.models import ModelUpdate

    stored = _stored_pd_model("vllm-mooncake", _shared_ext())
    if inherit:
        stored.roles[1].extended_kv_cache = ExtendedKVCacheConfig(enabled=False)
    else:
        stored.roles = None
        stored.disaggregation = None
    monkeypatch.setattr(
        models_route.Model, "all_by_fields", AsyncMock(return_value=[stored])
    )
    assert await cache_services_route._models_referencing_cache_service(
        MagicMock(), _service()
    ) == [stored]
    _patch_lookups(monkeypatch, None)
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        replicas=0,
    )
    with pytest.raises(NotFoundException) as error:
        await models_route.validate_shared_kv_cache(
            MagicMock(), update, OWNER_PRINCIPAL, 1, stored=stored
        )
    assert "Cache service not found" in error.value.message
    assert models_route.CacheService.one_by_id.await_args.args[1] == 9
