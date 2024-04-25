from fastapi import APIRouter

from . import probes, ui, users, models, nodes, openai


resource_router = APIRouter()
resource_router.include_router(users.router, prefix="/users", tags=["users"])
resource_router.include_router(models.router, prefix="/models", tags=["models"])
resource_router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])


api_router = APIRouter()
api_router.include_router(probes.router, tags=["probes"])
api_router.include_router(openai.router, tags=["openai"])
api_router.include_router(ui.router)
api_router.include_router(resource_router, prefix="/v1")
