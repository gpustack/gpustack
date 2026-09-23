"""Compatibility reporting for hybrid-attention models and extended KV cache.

``evaluate_hybrid_model_kv_cache`` answers the check the UI runs before a
deploy. It is a warning the user may deploy through, so what it stays silent
about matters as much as what it reports: the local mode is refused because
no setting makes its connector serve a hybrid model, while an LMCache service
is reported only until it carries the settings a hybrid model needs.
"""

import types
import pytest

from gpustack.scheduler import evaluator
from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderComponent,
    CacheProviderIntegration,
)
from gpustack.schemas.cache_services import CacheServiceConfig
from gpustack.schemas.model_evaluations import ModelSpec
from gpustack.schemas.models import (
    BackendEnum,
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    SourceEnum,
)

HYBRID_CONFIG = types.SimpleNamespace(
    layer_types=["linear_attention", "full_attention"]
)
PLAIN_CONFIG = types.SimpleNamespace(architectures=["LlamaForCausalLM"])

OWNER = 42


def _model(ext, backend=BackendEnum.VLLM, gpu_ids=None, owner_principal_id=OWNER):
    return ModelSpec(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3.5-0.8B",
        name="qwen3-5",
        backend=backend,
        extended_kv_cache=ext,
        owner_principal_id=owner_principal_id,
        gpu_selector=({"gpu_ids": gpu_ids} if gpu_ids is not None else None),
    )


def _local():
    return ExtendedKVCacheConfig(enabled=True, ram_size=20)


def _shared(cache_service_id=9):
    return ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=cache_service_id
    )


def _worker(*vendors, name="worker-1"):
    return types.SimpleNamespace(
        name=name,
        status=types.SimpleNamespace(
            gpu_devices=[
                types.SimpleNamespace(
                    vendor=vendor,
                    arch_family=None,
                    type="cann" if vendor == "ascend" else "cuda",
                    index=index,
                )
                for index, vendor in enumerate(vendors)
            ]
        ),
    )


def _service(
    config=None, name="lmcache", provider_name="LMCache", owner_principal_id=OWNER
):
    return types.SimpleNamespace(
        id=9,
        name=name,
        deleted_at=None,
        owner_principal_id=owner_principal_id,
        provider_name=provider_name,
        config=config,
    )


def _provider(name="LMCache"):
    return CacheProvider(
        name=name,
        custom_version=True,
        components={"server": CacheProviderComponent(attach_endpoint=True)},
        inference_backend_integrations=[CacheProviderIntegration(backend="vLLM")],
    )


def _patch_config(monkeypatch, config=HYBRID_CONFIG, raises=None):
    async def fetch(model, workers=None, trust_remote_code=False):
        if raises is not None:
            raise raises
        return config

    monkeypatch.setattr(evaluator, "get_pretrained_config_with_workers", fetch)


def _patch_provider(monkeypatch, provider=None, missing=False):
    async def lookup(_session, name=None):
        if missing:
            return None
        return provider if provider is not None else _provider()

    monkeypatch.setattr(evaluator, "get_cache_provider", lookup)


def _patch_config_unreachable(monkeypatch):
    async def unreachable(*_args, **_kwargs):
        raise AssertionError("a deployment with no verdict to give must not read")

    monkeypatch.setattr(evaluator, "get_pretrained_config_with_workers", unreachable)


def _patch_query(monkeypatch, row):
    """Answer like the database: a row only comes back when it matches every
    condition the query carries, so a scope left out of the query is a scope
    the test does not see applied."""

    async def query(_session, fields, options=None):
        if row is None:
            return None
        if any(getattr(row, key, None) != value for key, value in fields.items()):
            return None
        return row

    monkeypatch.setattr(evaluator.CacheService, "one_by_fields", query)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ext",
    [None, ExtendedKVCacheConfig(enabled=False)],
    ids=["absent", "disabled"],
)
async def test_cache_not_enabled_reports_nothing(monkeypatch, ext):
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(ext), [_worker("nvidia")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_plain_model_reports_nothing(monkeypatch):
    _patch_config(monkeypatch, PLAIN_CONFIG)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local()), [_worker("nvidia")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_unreadable_config_reports_nothing(monkeypatch):
    """A gated repo or an offline hub must not turn into a claim about the
    model's architecture."""
    _patch_config(monkeypatch, raises=ValueError("no config"))

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local()), [_worker("nvidia")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_local_mode_is_refused(monkeypatch):
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local()), [_worker("nvidia")]
    )

    assert compatible is False
    assert "cannot run with the local extended KV cache" in messages[0]
    assert evaluator.HYBRID_MODEL_DOC_URL in messages[0]


@pytest.mark.asyncio
async def test_local_mode_on_ascend_only_candidates_reports_nothing(monkeypatch):
    """Ascend runs the connector vllm-ascend ships rather than LMCache, and
    that path was not what was observed failing."""
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local()), [_worker("ascend")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_local_mode_on_a_mixed_fleet_reports_nothing(monkeypatch):
    """The candidates are not the placement: the label, GPU-type and
    backend-framework filters run later, so one Ascend candidate is a
    placement this cannot rule out. Reporting a deployment that would have
    run costs more than staying quiet about one that will not."""
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local()), [_worker("ascend"), _worker("nvidia", name="worker-2")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_local_mode_without_candidate_gpus_reports_nothing(monkeypatch):
    """A pinned GPU whose worker was filtered out, or a fleet with no GPUs.
    Scheduling reports either accurately — and it never gets to, because a
    verdict here returns from the evaluation before scheduling runs."""
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local(), gpu_ids=["worker-down:cuda:0"]), [_worker("nvidia")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_local_mode_with_another_backend_reports_nothing(monkeypatch):
    _patch_config(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_local(), backend=BackendEnum.SGLANG), [_worker("nvidia")]
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_shared_mode_reports_an_unconfigured_service(monkeypatch):
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch)
    service = _service()

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], service
    )

    assert compatible is False
    assert "lmcache" in messages[0]
    assert evaluator.LMCACHE_HYBRID_SERVICE_NOTE in messages[0]
    assert evaluator.HYBRID_MODEL_DOC_URL in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        CacheServiceConfig(fields={"chunk_size": 544}),
        CacheServiceConfig(parameters={"server": ["--separate-object-groups"]}),
        # The flag belongs to the component engines attach to; on another
        # component it is not what the engine talks to.
        CacheServiceConfig(
            fields={"chunk_size": 544},
            parameters={"coordinator": ["--separate-object-groups"]},
        ),
    ],
    ids=["chunk_size_only", "flag_only", "flag_on_another_component"],
)
async def test_shared_mode_reports_a_half_configured_service(monkeypatch, config):
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch)
    service = _service(config)

    compatible, _ = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], service
    )

    assert compatible is False


@pytest.mark.asyncio
async def test_shared_mode_accepts_a_configured_service(monkeypatch):
    """The user already configured the service for a hybrid model; whether
    the chunk size is the right multiple is the engine's to answer."""
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch)
    service = _service(
        CacheServiceConfig(
            fields={"chunk_size": 544},
            parameters={"server": ["--separate-object-groups"]},
        )
    )

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], service
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_another_provider_reports_nothing(monkeypatch):
    """The constraint belongs to LMCache's connector. A provider nobody has
    run against a hybrid model is not guessed about."""
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch, _provider(name="Mooncake"))
    service = _service(provider_name="Mooncake")

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], service
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_shared_mode_with_another_backend_reports_nothing(monkeypatch):
    """SGLang attaches to the same server, but its adapter has not been run
    against a hybrid model."""
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch)
    service = _service()

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None,
        _model(_shared(), backend=BackendEnum.SGLANG),
        [_worker("nvidia")],
        service,
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_shared_mode_without_a_service_reports_nothing(monkeypatch):
    """No service attached, or none its Org can see: the deployment is
    rejected on create/update, and this has nothing to add."""
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], None
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_another_orgs_service_is_not_loaded(monkeypatch):
    """Evaluation runs no shared-cache validation of its own, so the Org has
    to be a condition on the query: the verdict carries the service's name,
    and answering at all would tell the caller whether the id exists."""
    _patch_query(monkeypatch, _service(owner_principal_id=OWNER + 1))

    assert await evaluator.visible_cache_service(None, _model(_shared())) is None


@pytest.mark.asyncio
async def test_the_orgs_own_service_is_loaded(monkeypatch):
    service = _service()
    _patch_query(monkeypatch, service)

    assert await evaluator.visible_cache_service(None, _model(_shared())) is service


@pytest.mark.asyncio
async def test_platform_admin_sees_the_service_by_id(monkeypatch):
    """An evaluation with no Org resolved is the platform admin's."""
    service = _service(owner_principal_id=OWNER + 1)
    _patch_query(monkeypatch, service)

    model = _model(_shared(), owner_principal_id=None)

    assert await evaluator.visible_cache_service(None, model) is service


@pytest.mark.asyncio
async def test_a_local_deployment_queries_no_service(monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise AssertionError("a deployment attaching to no service must not query")

    monkeypatch.setattr(evaluator.CacheService, "one_by_fields", unreachable)

    assert await evaluator.visible_cache_service(None, _model(_local())) is None


def test_the_cache_key_follows_what_the_verdict_reads():
    """Everything the message or the verdict reads belongs in the key, or a
    user who acts on a verdict gets it back unchanged until the entry
    expires."""
    base = _service(CacheServiceConfig(fields={"chunk_size": 544}))
    key = evaluator.cache_service_verdict_key(base)

    configured = _service(
        CacheServiceConfig(
            fields={"chunk_size": 544},
            parameters={"server": ["--separate-object-groups"]},
        )
    )
    renamed = _service(CacheServiceConfig(fields={"chunk_size": 544}), name="renamed")
    other_provider = _service(
        CacheServiceConfig(fields={"chunk_size": 544}), provider_name="Mooncake"
    )

    assert key != evaluator.cache_service_verdict_key(configured)
    assert key != evaluator.cache_service_verdict_key(renamed)
    assert key != evaluator.cache_service_verdict_key(other_provider)
    assert evaluator.cache_service_verdict_key(None) is None


def test_the_cache_key_is_stable_for_an_unchanged_service():
    service = _service(CacheServiceConfig(fields={"chunk_size": 544}))

    assert evaluator.cache_service_verdict_key(
        service
    ) == evaluator.cache_service_verdict_key(service)


@pytest.mark.asyncio
async def test_an_unknown_provider_reports_nothing(monkeypatch):
    """A catalog that no longer declares the provider leaves the component
    the parameters belong to unknown. Reading the wrong one would report a
    configured service as unconfigured — the one direction this check must
    not fail in."""
    _patch_config(monkeypatch)
    _patch_provider(monkeypatch, missing=True)
    service = _service(
        CacheServiceConfig(
            fields={"chunk_size": 544},
            parameters={"server": ["--separate-object-groups"]},
        )
    )

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, _model(_shared()), [_worker("nvidia")], service
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, service",
    [
        (_model(_local(), backend=BackendEnum.SGLANG), None),
        (_model(_shared(), backend=BackendEnum.SGLANG), _service()),
        (_model(_shared()), None),
        (_model(_shared()), _service(provider_name="Mooncake")),
        (_model(ExtendedKVCacheConfig(enabled=False)), None),
    ],
    ids=["sglang_local", "sglang_shared", "no_service", "other_provider", "disabled"],
)
async def test_a_deployment_with_no_verdict_to_give_reads_no_config(
    monkeypatch, model, service
):
    """The model's config can be a hub round-trip whenever the local cache is
    cold. A deployment these checks can only stay silent about is recognized
    from the spec and the resolved service, before paying for it."""
    _patch_config_unreachable(monkeypatch)
    _patch_provider(monkeypatch)

    compatible, messages = await evaluator.evaluate_hybrid_model_kv_cache(
        None, model, [_worker("nvidia")], service
    )

    assert (compatible, messages) == (True, [])


@pytest.mark.asyncio
async def test_a_failed_service_lookup_reports_the_spec_not_the_request(monkeypatch):
    """Specs are evaluated concurrently over one session, so a query raising
    out of this function would fail the whole request rather than the one
    spec it belongs to."""

    async def boom(*_args, **_kwargs):
        raise RuntimeError("this session is already running an operation")

    monkeypatch.setattr(evaluator.CacheService, "one_by_fields", boom)

    result = await evaluator.evaluate_model_with_cache(
        None, None, _model(_shared()), [], []
    )

    assert result.error is True
    assert "already running" in result.error_message
