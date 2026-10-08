import pytest

from gpustack.routes.worker import filesystem


@pytest.mark.asyncio
async def test_model_weight_size_accepts_a_single_file(tmp_path):
    weight = tmp_path / "model.ninfer"
    weight.write_bytes(b"weights")

    assert await filesystem.get_model_weight_size(str(weight)) == {"size": 7}
