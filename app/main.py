from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import settings
from app.db.database import init_db
from app.api.routes_predict import router as predict_router
from app.api.routes_dataset import router as dataset_router
from app.api.routes_samples import router as samples_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API para detección de estado de madurez de flor de cannabis usando YOLOv8 + Roboflow",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount(
    "/images/processed",
    StaticFiles(directory=str(settings.PROCESSED_DIR)),
    name="processed",
)
app.mount(
    "/images/raw",
    StaticFiles(directory=str(settings.UPLOAD_DIR)),
    name="raw",
)

templates = Jinja2Templates(directory="frontend/templates")

# Routers
app.include_router(predict_router)
app.include_router(dataset_router)
app.include_router(samples_router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": settings.APP_VERSION}
