from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from app.db.models import MaturityStage


# ── Prediction ────────────────────────────────────────────────
class TrichomeResult(BaseModel):
    counts: dict
    amber_pct: float
    milky_pct: float
    clear_pct: float
    inference_ms: float

class StigmaResult(BaseModel):
    orange_pct: float
    white_pct: float
    inference_ms: float

class PredictionResponse(BaseModel):
    id: str
    sample_id: str
    maturity_stage: MaturityStage
    confidence: float
    trichomes: TrichomeResult
    stigmas: StigmaResult
    annotated_image_url: Optional[str]
    total_inference_ms: float
    predicted_at: datetime

    class Config:
        from_attributes = True


# ── Sample ────────────────────────────────────────────────────
class SampleCreate(BaseModel):
    plant_id: Optional[str] = None
    strain: Optional[str] = None
    week_of_flower: Optional[int] = None
    camera_model: Optional[str] = None
    magnification: Optional[float] = None
    lighting: Optional[str] = None

class SampleResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    plant_id: Optional[str]
    strain: Optional[str]
    week_of_flower: Optional[int]
    has_annotation: bool = False
    prediction_count: int = 0

    class Config:
        from_attributes = True


# ── Dataset ───────────────────────────────────────────────────
class DatasetVersionResponse(BaseModel):
    id: str
    task_type: str
    roboflow_project: str
    roboflow_version: int
    total_images: Optional[int]
    num_classes: Optional[int]
    class_names: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Experiment ────────────────────────────────────────────────
class TrainRequest(BaseModel):
    task: str = Field(..., pattern="^(trichome|stigma)$")
    roboflow_version: int = 1
    model_arch: str = "yolov8n"
    epochs: int = Field(default=100, ge=1, le=500)
    imgsz: int = Field(default=640, ge=320, le=1280)
    batch: int = Field(default=16, ge=1, le=64)
    experiment_name: Optional[str] = None

class ExperimentResponse(BaseModel):
    id: str
    name: str
    task_type: str
    status: str
    model_arch: str
    epochs: int
    map50: Optional[float]
    map50_95: Optional[float]
    started_at: datetime
    finished_at: Optional[datetime]
    mlflow_run_id: Optional[str]

    class Config:
        from_attributes = True


# ── Stats ─────────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_samples: int
    annotated_samples: int
    total_predictions: int
    stage_distribution: dict
    avg_confidence: float
    last_prediction_at: Optional[datetime]
