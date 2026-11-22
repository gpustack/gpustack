import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpustack.utils.model_env import (
    NON_MODEL_MIN_RAM_BYTES,
    get_non_model_cpu_claim,
    get_non_model_ram_claim,
)


@pytest.mark.parametrize("value", [None, "67108864", "4294967296"])
def test_non_model_ram_declaration_accepts_minimum_and_larger_limits(value):
    model = SimpleNamespace(env={"GPUSTACK_MODEL_RAM_CLAIM": value})
    expected = int(value) if value is not None else None
    assert get_non_model_ram_claim(model) == expected


@pytest.mark.parametrize(
    "value", ["", "bad", "0", "-1", "1048576", str(NON_MODEL_MIN_RAM_BYTES - 1), "1.5"]
)
def test_non_model_ram_declaration_rejects_invalid_or_too_small_limits(value):
    model = SimpleNamespace(env={"GPUSTACK_MODEL_RAM_CLAIM": value})
    with pytest.raises(ValueError, match="GPUSTACK_MODEL_RAM_CLAIM.*64 MiB"):
        get_non_model_ram_claim(model)


@pytest.mark.parametrize("value", [None, "0.25", "1.5", "2"])
def test_cpu_declaration_accepts_optional_fractional_cores(value):
    model = SimpleNamespace(env={"GPUSTACK_MODEL_CPU_CLAIM": value})
    expected = float(value) if value is not None else None
    assert get_non_model_cpu_claim(model) == expected


@pytest.mark.parametrize("value", ["", "bad", "0", "-0.5", "nan", "inf", "-inf"])
def test_cpu_declaration_rejects_invalid_or_nonpositive_cores(value):
    model = SimpleNamespace(env={"GPUSTACK_MODEL_CPU_CLAIM": value})
    with pytest.raises(ValueError, match="GPUSTACK_MODEL_CPU_CLAIM"):
        get_non_model_cpu_claim(model)


def test_resource_environment_parsing_does_not_load_server_or_scheduler():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder

class BlockServerImports(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith((
            'gpustack.policies', 'gpustack.server', 'gpustack.scheduler',
            'gpustack.worker',
        )):
            raise ImportError(fullname)

sys.meta_path.insert(0, BlockServerImports())
from gpustack.utils.model_env import (
    get_non_model_cpu_claim,
    get_non_model_ram_claim,
)
from types import SimpleNamespace

model = SimpleNamespace(env={})
assert get_non_model_cpu_claim(model) is None
assert get_non_model_ram_claim(model) is None
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
