from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    # TODO serve the UI.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>GPUStack</title>
    </head>
    <body>
    <h1>Welcome to GPUStack!</h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
