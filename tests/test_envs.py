"""How the GPUSTACK_* knobs in ``gpustack.envs`` read their environment.

The module resolves every value once at import, so varying the environment
means re-executing it.
"""

import importlib

import pytest

from gpustack import envs


@pytest.fixture
def envs_with(monkeypatch):
    """Re-read ``gpustack.envs`` under a modified environment.

    ``reload`` rebinds the attributes of the one shared module object, so the
    teardown reload restores what every importer of it sees.
    """

    def _reload(**environment):
        for name, value in environment.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return importlib.reload(envs)

    yield _reload

    monkeypatch.undo()
    importlib.reload(envs)


@pytest.mark.parametrize(
    "value, enabled",
    [
        (None, False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        # Only "true" and "1" count. Presence alone does not enable it, so an
        # operator who writes a falsy value gets certificate verification --
        # and "yes" / "on", which the backend-parameter parser does accept,
        # are not a way to turn TLS verification off by accident.
        ("false", False),
        ("0", False),
        ("yes", False),
    ],
)
def test_insecure_tls(envs_with, value, enabled):
    assert envs_with(GPUSTACK_INSECURE_TLS=value).INSECURE_TLS is enabled


@pytest.mark.parametrize(
    "value, enabled",
    [(None, False), ("false", False), ("true", True), ("TRUE", True)],
)
def test_debug_enable_tracemalloc(envs_with, value, enabled):
    assert (
        envs_with(GPUSTACK_DEBUG_ENABLE_TRACEMALLOC=value).DEBUG_ENABLE_TRACEMALLOC
        is enabled
    )


def test_gpu_instance_finalizing_unreachable_tolerance(envs_with):
    """The finalizing tolerance defaults to 30 minutes and reads the environment."""
    assert (
        envs_with(
            GPUSTACK_GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE=None
        ).GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE
        == 1800
    )
    assert (
        envs_with(
            GPUSTACK_GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE="60"
        ).GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE
        == 60
    )
    # 0 means "give up at once" -- the same polarity as the PV finalizer knob,
    # and the opposite of GPU_INSTANCE_UNREADABLE_CR_TOLERANCE's 0.
    assert (
        envs_with(
            GPUSTACK_GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE="0"
        ).GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE
        == 0
    )
