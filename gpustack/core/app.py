from fastapi import FastAPI


from ..routes.routes import api_router


app = FastAPI(
    title="GPUStack",
    response_model_exclude_unset=True,
)


app.include_router(api_router)
