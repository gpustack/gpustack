"""The worker's copy of the cache-provider catalog: when it is fetched, when a
miss is worth a round trip, and what a worker that never read it knows."""

from typing import Any, Dict, List

import pytest

from gpustack.worker import cache_provider_manager as manager_module
from gpustack.worker.cache_provider_manager import CacheProviderManager


def _declaration(name: str, version: str = "v1") -> Dict[str, Any]:
    return {
        "name": name,
        "display_name": name,
        "description": f"{name} for tests.",
        "topology": "per_node",
        "default_version": version,
        "default_image": f"{name.lower()}:{version}",
        "versions": {version: {}},
        "default_run_command": f"{name.lower()} --port {{{{port}}}}",
    }


class _Response:
    def __init__(self, items: List[Dict[str, Any]], error: Exception = None):
        self._items = items
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return {"items": self._items}


class _HttpxClient:
    def __init__(self):
        self.responses: List[_Response] = []
        self.calls: List[Dict[str, Any]] = []

    def get(self, path, params=None):
        self.calls.append({"path": path, "params": params})
        return self.responses.pop(0)


class _ClientSet:
    def __init__(self, httpx_client):
        self.http_client = self
        self._httpx_client = httpx_client

    def get_httpx_client(self):
        return self._httpx_client


@pytest.fixture
def catalog(monkeypatch):
    """A manager wired to a scripted API, with the miss throttle disabled so a
    test controls when a refresh may run."""
    httpx_client = _HttpxClient()
    monkeypatch.setattr(manager_module, "_REFRESH_MIN_INTERVAL_SECONDS", 0)
    return CacheProviderManager(lambda: _ClientSet(httpx_client)), httpx_client


def test_a_provider_is_served_from_the_fetched_catalog(catalog):
    provider_manager, http = catalog
    http.responses.append(_Response([_declaration("Demo")]))

    provider = provider_manager.get("Demo")

    assert provider is not None and provider.default_image == "demo:v1"
    # Every provider in one response: a catalog is a handful of declarations.
    assert http.calls == [{"path": "/cache-providers", "params": {"page": 0}}]


def test_a_name_matches_whatever_case_it_is_asked_in(catalog):
    provider_manager, http = catalog
    http.responses.append(_Response([_declaration("LMCache")]))

    assert provider_manager.get("lmcache") is not None


def test_a_cached_provider_costs_no_round_trip(catalog):
    provider_manager, http = catalog
    http.responses.append(_Response([_declaration("Demo")]))

    provider_manager.get("Demo")
    provider_manager.get("Demo")

    assert len(http.calls) == 1


def test_a_miss_refreshes_once_and_then_serves_what_arrived(catalog):
    """A provider added to the catalog after the last pass can have an instance
    scheduled here before the next one, so a miss is worth a fetch."""
    provider_manager, http = catalog
    http.responses.append(_Response([_declaration("Demo")]))
    provider_manager.sync()
    assert provider_manager.get("Added") is None

    http.responses.append(_Response([_declaration("Demo"), _declaration("Added")]))
    assert provider_manager.get("Added") is not None


def test_a_miss_is_throttled_against_a_catalog_that_lacks_the_provider(monkeypatch):
    """Repeated lookups for a provider the catalog genuinely does not carry must
    not hammer the API."""
    http = _HttpxClient()
    provider_manager = CacheProviderManager(lambda: _ClientSet(http))
    http.responses.append(_Response([_declaration("Demo")]))

    assert provider_manager.get("Missing") is None
    assert provider_manager.get("Missing") is None

    assert len(http.calls) == 1


def test_a_worker_that_never_read_the_catalog_says_so(catalog):
    """Told apart from a catalog that simply carries no such provider: one is a
    connectivity problem, the other a configuration one."""
    provider_manager, http = catalog
    http.responses.append(_Response([], error=RuntimeError("connection refused")))

    assert provider_manager.get("Demo") is None
    assert provider_manager.loaded is False

    http.responses.append(_Response([_declaration("Demo")]))
    provider_manager.sync()
    assert provider_manager.loaded is True


def test_a_miss_says_whether_the_catalog_could_be_read(catalog):
    """The two things a miss can mean: a catalog read that does not carry the
    provider (its declaration is gone), and a catalog this worker could not
    read (the provider may well exist). Reporting the second as the first sends
    whoever reads the instance after a declaration that is probably there."""
    provider_manager, http = catalog
    http.responses.append(_Response([], error=RuntimeError("connection refused")))

    provider, catalog_read = provider_manager.lookup("Demo")
    assert provider is None and catalog_read is False

    http.responses.append(_Response([_declaration("Other")]))
    provider, catalog_read = provider_manager.lookup("Demo")
    assert provider is None and catalog_read is True


def test_a_throttled_miss_answers_from_the_copy_the_throttle_protects(monkeypatch):
    """The throttle exists so a provider nothing declares is not looked up on
    every probe. The copy it protects was fetched seconds ago, which is recent
    enough to answer: calling that a catalog this worker could not read would
    blame connectivity for a provider that genuinely is not declared."""
    http = _HttpxClient()
    provider_manager = CacheProviderManager(lambda: _ClientSet(http))
    http.responses.append(_Response([_declaration("Demo")]))
    provider_manager.sync()

    provider, catalog_read = provider_manager.lookup("Added")

    assert provider is None and catalog_read is True
    # Nothing was fetched: the periodic pass had just run.
    assert len(http.calls) == 1


def test_a_miss_before_any_successful_read_answers_nothing(monkeypatch):
    """The other side of it: a worker that has never read the catalog knows
    nothing about what it does or does not carry."""
    http = _HttpxClient()
    provider_manager = CacheProviderManager(lambda: _ClientSet(http))
    http.responses.append(_Response([], error=RuntimeError("connection refused")))

    provider, catalog_read = provider_manager.lookup("Demo")

    assert provider is None and catalog_read is False


def test_a_failed_refresh_keeps_serving_what_was_fetched(catalog):
    """A blip on the API must not empty the catalog a running worker launches
    from."""
    provider_manager, http = catalog
    http.responses.append(_Response([_declaration("Demo")]))
    provider_manager.sync()

    http.responses.append(_Response([], error=RuntimeError("boom")))
    provider_manager.sync()

    assert provider_manager.get("Demo") is not None


def test_the_periodic_pass_refreshes_whatever_the_throttle_says():
    """The loop's own cadence is its throttle; the miss throttle must not skip
    its passes."""
    http = _HttpxClient()
    provider_manager = CacheProviderManager(lambda: _ClientSet(http))
    http.responses.append(_Response([_declaration("Demo")]))
    http.responses.append(_Response([_declaration("Demo"), _declaration("Added")]))

    provider_manager.sync()
    provider_manager.sync()

    assert len(http.calls) == 2
    assert provider_manager.get("Added") is not None
