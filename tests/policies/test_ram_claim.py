"""Who pays for an extended KV cache's host memory.

The distinction these pin down: ``ram_size`` / ``ram_ratio`` size the cache
the *engine process* offloads into host memory, which exists only in "local"
mode. A "shared" cache lives in a separate cache service, and both worker
backends return before they apply either knob -- ``vllm.py``'s
``_set_lmcache_env`` never sets ``LMCACHE_MAX_LOCAL_CPU_SIZE``, ``sglang.py``
never passes ``--hicache-ratio``. The scheduler booked it anyway, and since
``ram_ratio`` defaults to 1.2, merely picking a cache service made every
accelerator-bearing member reserve 1.2x its VRAM claim of memory that no
process would take.
"""

import pytest

from gpustack.policies.utils import get_computed_ram_claim, get_model_ram_claim
from gpustack.schemas.models import (
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    Model,
)

GIB = 1024**3


def _model(ext=None) -> Model:
    model = Model(name="m", source="huggingface", huggingface_repo_id="x/y")
    model.extended_kv_cache = ext
    return model


def _shared(**kwargs) -> ExtendedKVCacheConfig:
    return ExtendedKVCacheConfig(
        enabled=True,
        mode=KVCacheModeEnum.SHARED,
        cache_service_id=1,
        **kwargs,
    )


def _local(**kwargs) -> ExtendedKVCacheConfig:
    return ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL, **kwargs)


def test_a_shared_cache_books_no_host_ram_off_the_default_ratio():
    """The regression, and the default is the whole of it: a deployment that
    only ever picked a cache service carries ``ram_ratio=1.2`` without anyone
    having chosen it."""
    ext = _shared()
    assert ext.ram_ratio == 1.2

    assert get_computed_ram_claim(_model(ext), {0: 40 * GIB}) is None


def test_a_shared_cache_books_no_host_ram_off_an_explicit_size():
    """Same rule for the knob a user *did* write. It sizes a local offload,
    and a shared deployment has none -- the cache service holds the memory."""
    assert get_computed_ram_claim(_model(_shared(ram_size=16)), {0: 40 * GIB}) is None
    assert get_model_ram_claim(_model(_shared(ram_size=16))) == 0


@pytest.mark.parametrize(
    "ext, vram, expected",
    [
        (_local(), {0: 40 * GIB}, int(40 * GIB * 1.2)),
        (_local(ram_ratio=2.0), {0: 10 * GIB}, 20 * GIB),
        (_local(ram_size=16), {0: 40 * GIB}, 16 * GIB),
        # Several cards under one member: the ratio applies to the whole claim.
        (_local(ram_ratio=1.0), {0: 8 * GIB, 1: 8 * GIB}, 16 * GIB),
    ],
)
def test_a_local_cache_still_books_what_it_always_did(ext, vram, expected):
    assert get_computed_ram_claim(_model(ext), vram) == expected


def test_a_local_size_still_pre_checks():
    """`get_model_ram_claim` is the worker-level pre-check; a local cache with
    a declared size is the one case it can answer before a GPU is picked."""
    assert get_model_ram_claim(_model(_local(ram_size=16))) == 16 * GIB


def test_a_local_ratio_has_no_pre_check():
    """Not a gap this change closes, and worth stating so it is not mistaken
    for one: the ratio needs the VRAM claim, which does not exist until a card
    is chosen, so the pre-check cannot see it."""
    assert get_model_ram_claim(_model(_local(ram_ratio=1.2))) == 0


def test_a_disabled_or_absent_cache_books_nothing():
    assert get_computed_ram_claim(_model(None), {0: 40 * GIB}) is None
    assert (
        get_computed_ram_claim(
            _model(ExtendedKVCacheConfig(enabled=False, ram_ratio=1.2)), {0: 40 * GIB}
        )
        is None
    )


def test_an_explicit_static_ram_outranks_every_mode():
    """The caller-supplied figure is the router's declared memory, and it is
    the answer whatever the cache is doing."""
    assert get_computed_ram_claim(_model(_shared()), {}, static_ram=2 * GIB) == 2 * GIB
    assert get_computed_ram_claim(_model(_local()), {}, static_ram=2 * GIB) == 2 * GIB
