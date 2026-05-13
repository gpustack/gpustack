import pytest

from gpustack.gpu_instances import get_builtin_templates


@pytest.mark.asyncio
async def test_get_builtin_templates():

    templates = await get_builtin_templates()
    assert len(templates) > 0
    for template in templates:
        assert template.name is not None
        assert template.manufacturer is not None
