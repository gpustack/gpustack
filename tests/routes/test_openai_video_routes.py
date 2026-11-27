"""The Server-side split of the video family.

Both spellings ride the model ingress on the gateway (see
``openai_model_prefixes``). On the Server they differ: the asynchronous
``/videos`` belongs to the job router, which qualifies the id with the
creating instance, and must not be shadowed by the generic model proxy's
auto-registration; the synchronous ``/videos/sync`` is an ordinary model
request and keeps the generic path. The legacy router stays frozen.
"""

from gpustack.routes.openai import get_api_router, get_legacy_api_router


def _paths(router):
    return {route.path for route in router.routes}


def test_async_create_is_left_to_the_job_router():
    assert "/videos" not in _paths(get_api_router())


def test_sync_generation_rides_the_generic_model_proxy():
    assert "/videos/sync" in _paths(get_api_router())


def test_video_stays_off_the_frozen_legacy_router():
    paths = _paths(get_legacy_api_router())
    assert "/videos" not in paths
    assert "/videos/sync" not in paths
