import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import debug


def test_memory_profile_explains_how_to_enable_tracing(monkeypatch):
    monkeypatch.setattr(debug.tracemalloc, "is_tracing", lambda: False)
    with pytest.raises(BadRequestException) as exc:
        debug.get_memory_profile()
    assert "GPUSTACK_DEBUG_ENABLE_TRACEMALLOC=true" in exc.value.message
    assert "restart" in exc.value.message
