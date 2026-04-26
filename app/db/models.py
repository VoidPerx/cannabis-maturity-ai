"""
Sustainable data architecture:
- Sample: una imagen de flor con metadata de captura
- Annotation: estado de madurez asignado (ground truth)
- Prediction: resultado de inferencia (trichomes + stigmas)
- DatasetVersion: versión de Roboflow vinculada a cada experimento
- Experiment: run de entrenamiento trackeado en MLflow
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, ForeignKey, JSON, Text, Enum
)
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


class MaturityStage(str, enum.Enum):
    EARLY = "early"           # Tricomas transparentes, estigmas blancos
    MID = "mid"               # Tricomas lechosos, estigmas virando
    LATE = "late"             # Tricomas ámbar, estigmas naranjos >70%
    HARVEST_READY = "harvest_ready"  # Peak madurez


class Sample(Base):
    __tablename__ = "samples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    captured_at = Column(DateTime, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Metadata de captura
    camera_model = Column(String, nullable=True)
    magnification = Column(Float, nullable=True)   # Para microscopio/lupa
    lighting = Column(String, nullable=True)       # natural, UV, etc.
    plant_id = Column(String, nullable=True)       # ID de la planta
    strain = Column(String, nullable=True)         # Variedad/genotipo
    week_of_flower = Column(Integer, nullable=True)

    # Relaciones
    annotation = relationship("Annotation", back_populates="sample", uselist=False)
    predictions = relationship("Prediction", back_populates="sample")


class Annotation(Base):
    """Ground truth etiquetado manualmente (ej: desde Roboflow)."""
    __tablename__ = "annotations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False, unique=True)
    maturity_stage = Column(Enum(MaturityStage), nullable=False)
    annotator = Column(String, nullable=True)
    annotated_at = Column(DateTime, default=datetime.utcnow)

    # Métricas visuales manuales (extraídas de Roboflow)
    trichome_boxes = Column(JSON, nullable=True)    # [{x,y,w,h,class,conf}]
    stigma_masks = Column(JSON, nullable=True)      # [{polygon, class, conf}]

    # Features calculadas
    amber_trichome_pct = Column(Float, nullable=True)
    milky_trichome_pct = Column(Float, nullable=True)
    orange_stigma_pct = Column(Float, nullable=True)
    white_stigma_pct = Column(Float, nullable=True)

    notes = Column(Text, nullable=True)
    roboflow_image_id = Column(String, nullable=True)  # ID en Roboflow

    sample = relationship("Sample", back_populates="annotation")


class Prediction(Base):
    """Resultado de inferencia del modelo."""
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False)
    model_version = Column(String, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    # Resultado principal
    predicted_stage = Column(Enum(MaturityStage), nullable=False)
    confidence = Column(Float, nullable=False)

    # Detección de trichomas (YOLO detect)
    trichome_detections = Column(JSON, nullable=True)
    trichome_amber_count = Column(Integer, default=0)
    trichome_milky_count = Column(Integer, default=0)
    trichome_clear_count = Column(Integer, default=0)

    # Segmentación de estigmas (YOLO segment)
    stigma_masks = Column(JSON, nullable=True)
    orange_stigma_pct = Column(Float, nullable=True)
    white_stigma_pct = Column(Float, nullable=True)

    # Output visual
    annotated_image_path = Column(String, nullable=True)
    inference_time_ms = Column(Float, nullable=True)

    # Correcto vs ground truth (si existe annotation)
    is_correct = Column(Boolean, nullable=True)

    sample = relationship("Sample", back_populates="predictions")


class DatasetVersion(Base):
    """Versiones del dataset en Roboflow — trazabilidad completa."""
    __tablename__ = "dataset_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_type = Column(String, nullable=False)  # "detect" | "segment"
    roboflow_project = Column(String, nullable=False)
    roboflow_version = Column(Integer, nullable=False)
    download_url = Column(String, nullable=True)
    format = Column(String, default="yolov8")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Stats
    total_images = Column(Integer, nullable=True)
    train_images = Column(Integer, nullable=True)
    val_images = Column(Integer, nullable=True)
    test_images = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)
    class_names = Column(JSON, nullable=True)

    experiments = relationship("Experiment", back_populates="dataset_version")


class Experiment(Base):
    """Run de entrenamiento trackeado con MLflow."""
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)  # "detect" | "segment"
    dataset_version_id = Column(String, ForeignKey("dataset_versions.id"))
    mlflow_run_id = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default="running")  # running | done | failed

    # Hiperparámetros
    model_arch = Column(String, default="yolov8n")
    epochs = Column(Integer, default=100)
    imgsz = Column(Integer, default=640)
    batch = Column(Integer, default=16)
    hyperparams = Column(JSON, nullable=True)

    # Métricas finales
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)

    model_path = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    dataset_version = relationship("DatasetVersion", back_populates="experiments")
