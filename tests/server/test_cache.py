import asyncio
import pytest
from aiocache import Cache

from gpustack.logging import setup_logging
from gpustack.server.cache import (
    build_cache_key,
    delete_cache_by_key,
    locked_cached,
    class_key,
    cache as global_cache,
)

setup_logging()


def make_cache():
    return Cache(Cache.MEMORY)


# ---------------------------------------------------------------------------
# build_cache_key
# ---------------------------------------------------------------------------


class TestBuildCacheKey:
    def test_positional_and_keyword_produce_same_key(self):
        async def my_func(name: str):
            pass

        assert build_cache_key(my_func, "foo") == build_cache_key(my_func, name="foo")

    def test_different_args_produce_different_keys(self):
        async def my_func(name: str):
            pass

        assert build_cache_key(my_func, "foo") != build_cache_key(my_func, "bar")

    def test_multiple_params_all_equivalent_forms(self):
        async def my_func(a: int, b: str):
            pass

        key_all_pos = build_cache_key(my_func, 1, "x")
        key_all_kw = build_cache_key(my_func, a=1, b="x")
        key_mixed = build_cache_key(my_func, 1, b="x")
        assert key_all_pos == key_all_kw == key_mixed

    def test_key_includes_function_qualname(self):
        async def my_func(x: int):
            pass

        assert "my_func" in build_cache_key(my_func, 42)

    def test_default_args_treated_as_explicit(self):
        async def my_func(x: int, y: int = 10):
            pass

        assert build_cache_key(my_func, 1) == build_cache_key(my_func, 1, 10)

    def test_unbound_method_strips_self(self):
        """Unbound method (with self in sig) called without self arg produces
        same key as bound method called with same arg."""

        class MyService:
            async def fetch(self, name: str):
                pass

        svc = MyService()
        key_unbound = build_cache_key(MyService.fetch, "foo")
        key_bound = build_cache_key(svc.fetch, "foo")
        assert key_unbound == key_bound

    def test_kwarg_ordering_is_stable(self):
        """Keys are stable regardless of the order keyword arguments are passed,
        because bound.arguments follows declaration order not caller order."""

        async def my_func(_a: int, _b: str, _c: float):
            pass

        key1 = build_cache_key(my_func, _a=1, _b="x", _c=3.0)
        key2 = build_cache_key(my_func, _c=3.0, _a=1, _b="x")
        key3 = build_cache_key(my_func, _b="x", _c=3.0, _a=1)
        assert key1 == key2 == key3

    def test_fallback_for_signature_mismatch(self):
        """When args don't match the function signature (e.g. manual key construction
        with extra args), fall back to old-style string concatenation without crashing.
        This covers the pre-existing ModelUsageService.update() call pattern."""

        async def my_func(_fields: dict):
            pass

        # Passing 3 args to a 1-param function triggers the fallback
        key = build_cache_key(my_func, 1, 2, 3)
        assert "my_func" in key

    def test_fallback_kwargs_are_sorted(self):
        """Fallback path (signature mismatch) sorts kwargs for stable keys."""

        async def my_func(_fields: dict):
            pass

        # Wrong kwarg names trigger the fallback; order should not matter
        key1 = build_cache_key(my_func, z=3, a=1, m=2)
        key2 = build_cache_key(my_func, a=1, m=2, z=3)
        assert key1 == key2


# ---------------------------------------------------------------------------
# locked_cached decorator
# ---------------------------------------------------------------------------


class TestLockedCached:
    @pytest.mark.asyncio
    async def test_result_is_cached_on_second_call(self):
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        r1 = await svc.fetch("foo")
        r2 = await svc.fetch("foo")

        assert r1 == r2 == "result-foo"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_args_have_separate_cache_entries(self):
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        await svc.fetch("foo")
        await svc.fetch("bar")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_none_result_is_not_cached(self):
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return None

        svc = MyService()
        await svc.fetch("foo")
        await svc.fetch("foo")

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_positional_and_keyword_hit_same_cache_entry(self):
        """Regression: before the inspect.signature fix, keyword-arg calls generated
        a different cache key than positional-arg calls, so cache was never reused."""
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        r1 = await svc.fetch("foo")
        r2 = await svc.fetch(name="foo")

        assert r1 == r2
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_key_matches_delete_cache_by_key(self):
        """Regression: deleting via positional arg must invalidate an entry that was
        populated via keyword arg (the bug in token.py before fix-5168)."""
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        await svc.fetch(name="foo")
        assert call_count == 1

        # Simulate what services.py update()/delete() does: positional arg, bound method
        key = build_cache_key(svc.fetch, "foo")
        await test_cache.delete(key)

        await svc.fetch(name="foo")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_delete_cache_by_key_invalidates_entry(self):
        """delete_cache_by_key correctly evicts a cached result (uses global cache)."""
        call_count = 0

        class MyService:
            @locked_cached()
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        await svc.fetch("foo")
        assert call_count == 1

        await delete_cache_by_key(svc.fetch, "foo")

        await svc.fetch("foo")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_delete_cache_by_key_positional_invalidates_keyword_call(self):
        """delete_cache_by_key with positional args invalidates entry cached via
        keyword args (the actual bug scenario in services.py update/delete)."""
        call_count = 0

        class MyService:
            @locked_cached()
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        await svc.fetch(name="foo")
        assert call_count == 1

        await delete_cache_by_key(svc.fetch, "foo")

        await svc.fetch(name="foo")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_calls_execute_function_once(self):
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.05)
                return f"result-{name}"

        svc = MyService()
        results = await asyncio.gather(*[svc.fetch("foo") for _ in range(5)])

        assert all(r == "result-foo" for r in results)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_custom_static_key(self):
        call_count = 0
        test_cache = make_cache()

        class MyService:
            @locked_cached(cache=test_cache, key="fixed-key")
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        await svc.fetch("foo")
        await svc.fetch("bar")  # different arg, same fixed key → cache hit

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_custom_callable_key(self):
        call_count = 0
        test_cache = make_cache()

        def my_key(f, *args, **kwargs):
            return f"custom:{args[1]}"  # args[0] is self

        class MyService:
            @locked_cached(cache=test_cache, key=my_key)
            async def fetch(self, name: str):
                nonlocal call_count
                call_count += 1
                return f"result-{name}"

        svc = MyService()
        r1 = await svc.fetch("foo")
        r2 = await svc.fetch("foo")

        assert r1 == r2
        assert call_count == 1


# ---------------------------------------------------------------------------
# class_key helper
# ---------------------------------------------------------------------------


class TestClassKey:
    def test_key_format_is_classname_dot_suffix(self):
        async def dummy():
            pass

        kb = class_key("all_cached")

        class MyModel:
            pass

        assert kb(dummy, MyModel) == "MyModel.all_cached"

    def test_different_classes_produce_different_keys(self):
        async def dummy():
            pass

        kb = class_key("all_cached")

        class A:
            pass

        class B:
            pass

        assert kb(dummy, A) != kb(dummy, B)


# ---------------------------------------------------------------------------
# delete_cache_by_key
# ---------------------------------------------------------------------------


class TestDeleteCacheByKey:
    @pytest.mark.asyncio
    async def test_delete_by_explicit_key(self):
        await global_cache.set("my-key", "my-value")
        assert await global_cache.get("my-key") == "my-value"

        await delete_cache_by_key(_key="my-key")

        assert await global_cache.get("my-key") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key_is_safe(self):
        await delete_cache_by_key(_key="nonexistent-key")

    @pytest.mark.asyncio
    async def test_raises_if_neither_func_nor_key(self):
        with pytest.raises(ValueError):
            await delete_cache_by_key()

    @pytest.mark.asyncio
    async def test_delete_by_func_and_args(self):
        call_count = 0

        class MyService:
            @locked_cached()
            async def lookup(self, item_id: int):
                nonlocal call_count
                call_count += 1
                return f"item-{item_id}"

        svc = MyService()
        await svc.lookup(42)
        assert call_count == 1

        await delete_cache_by_key(svc.lookup, 42)
        await svc.lookup(42)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_delete_only_removes_matching_key(self):
        call_count = {"a": 0, "b": 0}

        class MyService:
            @locked_cached()
            async def lookup(self, name: str):
                call_count[name] += 1
                return f"result-{name}"

        svc = MyService()
        await svc.lookup("a")
        await svc.lookup("b")

        await delete_cache_by_key(svc.lookup, "a")

        await svc.lookup("a")
        await svc.lookup("b")

        assert call_count["a"] == 2
        assert call_count["b"] == 1
