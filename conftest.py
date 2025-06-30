import shutil
import tempfile
import pytest
from gpustack.config.config import Config, set_global_config


@pytest.fixture(scope="module", autouse=True)
def temp_dir():
    tmp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {tmp_dir}")
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module", autouse=True)
def config(temp_dir):
    cfg = Config(
        token="test", jwt_secret_key="test", data_dir=temp_dir, enable_ray=True
    )
    set_global_config(cfg)
    return cfg
