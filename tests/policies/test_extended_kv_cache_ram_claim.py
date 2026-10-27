"""Host RAM a deployment claims for its extended KV cache.

The claim belongs to the in-process mode, where the cache sits in the
engine's own address space. A deployment attached to a cache service holds
no cache of its own — the worker skips the sizing and the memory is the
cache server's, under the service's capacity — so a claim there would
reserve RAM nothing uses, once for every deployment attached to the same
service.
"""

import types

import pytest

from gpustack.policies.utils import get_computed_ram_claim, get_model_ram_claim
from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum

GIB = 1024**3


def _model(extended_kv_cache):
    return types.SimpleNamespace(extended_kv_cache=extended_kv_cache)


def _local(**kwargs):
    return ExtendedKVCacheConfig(enabled=True, **kwargs)


def _shared(**kwargs):
    return ExtendedKVCacheConfig(
        enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=9, **kwargs
    )


def test_local_claims_its_configured_size():
    assert get_model_ram_claim(_model(_local(ram_size=20))) == 20 * GIB
    assert get_computed_ram_claim(_model(_local(ram_size=20)), {0: 8 * GIB}) == 20 * GIB


def test_local_claims_the_ratio_of_its_vram():
    claim = get_computed_ram_claim(_model(_local(ram_ratio=1.2)), {0: 10 * GIB})

    assert claim == int(12 * GIB)


@pytest.mark.parametrize(
    "extended_kv_cache",
    [_shared(ram_size=20), _shared(ram_ratio=1.2), _shared()],
    ids=["size", "ratio", "defaults"],
)
def test_shared_claims_nothing(extended_kv_cache):
    model = _model(extended_kv_cache)

    assert get_model_ram_claim(model) == 0
    assert get_computed_ram_claim(model, {0: 10 * GIB}) is None


@pytest.mark.parametrize(
    "extended_kv_cache",
    [None, ExtendedKVCacheConfig(enabled=False, ram_size=20)],
    ids=["absent", "disabled"],
)
def test_cache_not_enabled_claims_nothing(extended_kv_cache):
    model = _model(extended_kv_cache)

    assert get_model_ram_claim(model) == 0
    assert get_computed_ram_claim(model, {0: 10 * GIB}) is None


def test_a_measured_claim_still_wins():
    """An estimate the caller already made (the GGUF parser's) is the claim,
    whatever the cache configuration says."""
    assert (
        get_computed_ram_claim(_model(_shared(ram_size=20)), {0: 10 * GIB}, 4 * GIB)
        == 4 * GIB
    )
