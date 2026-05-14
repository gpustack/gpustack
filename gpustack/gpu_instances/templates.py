import asyncio
from typing import List

import aiofiles
import yaml

from gpustack.utils.compat_importlib import pkg_resources
from gpustack.schemas.common import ItemList
from gpustack.schemas import GPUInstanceTemplate
from sqlmodel.ext.asyncio.session import AsyncSession


async def get_builtin_templates() -> List[GPUInstanceTemplate]:
    """
    Retrieves the list of built-in GPU instance templates from a YAML file.
    """

    assets_dir = pkg_resources.files("gpustack.assets")
    file_path = assets_dir.joinpath("gpu-instance-templates.yaml")

    async with aiofiles.open(str(file_path), "r") as f:
        content = await f.read()

    data = await asyncio.to_thread(yaml.safe_load, content)
    page = ItemList[GPUInstanceTemplate](**data)
    return page.items


async def sync_builtin_templates_to_db(session: AsyncSession):
    """
    Syncs the built-in GPU instance templates to the database.

    Only creates templates that do not already exist in the database,
    ensuring that built-in templates are always available without overwriting any user-defined templates.
    """

    templates = await get_builtin_templates()
    for template in reversed(templates):
        # Skip templates without a name.
        if not template.name:
            continue

        existed = await GPUInstanceTemplate.exist_by_fields(
            session=session,
            fields={
                "owner_principal_id": None,
                "name": template.name,
            },
        )
        if existed:
            continue

        # Ensure the owner_principal_id is set to None for built-in templates.
        template.owner_principal_id = None
        await GPUInstanceTemplate.create(
            session=session,
            source=template,
        )
