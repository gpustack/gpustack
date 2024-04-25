from gpustack.utils import normalize_route_path


def test_normalize_route_path():
    assert (
        normalize_route_path("/api/v1") == "/api/v1"
    ), "Failed on already correct path"

    assert normalize_route_path("api/v1") == "/api/v1", "Failed to add leading slash"

    assert normalize_route_path("") == "/", "Failed on empty string input"

    assert normalize_route_path("/") == "/", "Failed on root path"
