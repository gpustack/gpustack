import pytest

from gpustack.gpu_instances import get_builtin_templates
from gpustack.schemas.gpu_instance_templates import GPUInstanceSpecTemplate


@pytest.mark.asyncio
async def test_get_builtin_templates():

    templates = await get_builtin_templates()
    assert len(templates) > 0
    for template in templates:
        assert template.name is not None
        assert template.manufacturer is not None


@pytest.mark.asyncio
async def test_no_builtin_template_disables_auth():
    # Regression guard for gpustack/gpustack#5757: a built-in template must
    # never launch its service with authentication explicitly disabled.
    templates = await get_builtin_templates()
    for template in templates:
        # Table-model init skips validation, so the YAML-loaded spec is a raw
        # dict; validate it into the schema the DB column deserializes to.
        spec = GPUInstanceSpecTemplate.model_validate(template.spec)
        for item in spec.command or []:
            assert "--ServerApp.token=''" not in item
            assert "--ServerApp.password=''" not in item


@pytest.mark.asyncio
async def test_jupyter_templates_carry_generated_token():
    templates = await get_builtin_templates()
    jupyter = [t for t in templates if t.name.startswith("jupyter-lab-")]
    assert len(jupyter) == 2
    for template in jupyter:
        spec = GPUInstanceSpecTemplate.model_validate(template.spec)
        assert "--ServerApp.token={{generated_token}}" in spec.command
        port = next(p for p in spec.ports if p.name == "JUPYTER")
        assert port.access_params == {"token": "{{generated_token}}"}
