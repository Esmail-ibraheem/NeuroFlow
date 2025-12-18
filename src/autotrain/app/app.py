import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from autotrain import __version__, logger
from autotrain.app.api_routes import api_router
from autotrain.app.oauth import attach_oauth
from autotrain.app.ui_routes import ui_router


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Starting AutoTrain...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if "SPACE_ID" in os.environ:
    attach_oauth(app)

app.include_router(ui_router, prefix="/ui", include_in_schema=False)
app.include_router(api_router, prefix="/api")
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
logger.info(f"AutoTrain version: {__version__}")
logger.info("AutoTrain started successfully")


@app.get("/")
async def forward_to_ui(request: Request):
    query_params = request.query_params
    url = "/ui/"
    if query_params:
        url += f"?{query_params}"
    return RedirectResponse(url=url)
