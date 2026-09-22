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
