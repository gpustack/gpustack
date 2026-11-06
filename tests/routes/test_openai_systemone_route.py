"""The server-side surface of the decision API: the ``/systemone`` prefix
registers ``POST /v1/systemone`` on the main router (model-name resolution
applies, body ``model`` is the route name) while the frozen legacy
``/v1-openai`` router stays untouched."""

from gpustack.routes.openai import get_api_router, get_legacy_api_router


def _paths(router):
    return {route.path for route in router.routes}


def test_systemone_registered_on_main_router():
    assert "/systemone" in _paths(get_api_router())


def test_systemone_absent_from_legacy_router():
    # The legacy subset is frozen; new endpoints go to /v1 only.
    assert "/systemone" not in _paths(get_legacy_api_router())
