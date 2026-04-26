from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import DatasetVersion
from app.services.roboflow_service import roboflow_service
from app.models.schemas import DatasetVersionResponse, TrainRequest, ExperimentResponse
from app.db.models import Experiment
import shutil, uuid
from pathlib import Path
from app.core.config import settings
from datetime import datetime

router = APIRouter(prefix="/dataset", tags=["Dataset & Training"])


@router.post("/upload")
async def upload_to_roboflow(
    file: UploadFile = File(...),
    task: str = Form("trichome"),
    split: str = Form("train"),
    db: Session = Depends(get_db),
):
    """Sube imagen a Roboflow para anotación."""
    tmp = settings.UPLOAD_DIR / f"rf_{uuid.uuid4().hex}{Path(file.filename).suffix}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = roboflow_service.upload_image(str(tmp), task=task, split=split)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"Error subiendo a Roboflow: {e}")

    tmp.unlink(missing_ok=True)
    return {"status": "uploaded", "roboflow_response": result}


@router.post("/download")
def download_dataset(
    task: str = "trichome",
    version: int = 1,
    db: Session = Depends(get_db),
):
    """Descarga versión de dataset desde Roboflow y registra en DB."""
    try:
        stats = roboflow_service.get_dataset_stats(task=task, version=version)
        path = roboflow_service.download_dataset(task=task, version=version)
    except Exception as e:
        raise HTTPException(500, f"Error descargando dataset: {e}")

    dv = DatasetVersion(
        task_type=task,
        roboflow_project=stats["project"],
        roboflow_version=version,
        total_images=stats.get("images"),
        class_names=list(stats.get("classes", {}).keys()),
        num_classes=len(stats.get("classes", {})),
    )
    db.add(dv)
    db.commit()
    db.refresh(dv)
    return {"dataset_version_id": dv.id, "path": str(path), "stats": stats}


@router.get("/versions/{task}")
def list_versions(task: str, db: Session = Depends(get_db)):
    """Lista versiones disponibles en Roboflow."""
    try:
        return roboflow_service.list_versions(task=task)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/train", response_model=ExperimentResponse)
def start_training(req: TrainRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Lanza entrenamiento YOLOv8 en background con tracking MLflow."""
    from app.services.training_service import train_model

    exp = Experiment(
        name=req.experiment_name or f"{req.task}_v{req.roboflow_version}",
        task_type=req.task,
        model_arch=req.model_arch,
        epochs=req.epochs,
        imgsz=req.imgsz,
        batch=req.batch,
        status="running",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    def run_training(exp_id: str):
        from app.db.database import SessionLocal
        session = SessionLocal()
        try:
            result = train_model(
                task=req.task,
                roboflow_version=req.roboflow_version,
                model_arch=req.model_arch,
                epochs=req.epochs,
                imgsz=req.imgsz,
                batch=req.batch,
                experiment_name=req.experiment_name,
            )
            experiment = session.query(Experiment).filter(Experiment.id == exp_id).first()
            if experiment:
                experiment.status = "done"
                experiment.mlflow_run_id = result["run_id"]
                experiment.model_path = result["model_path"]
                experiment.finished_at = datetime.utcnow()
                metrics = result.get("metrics", {})
                experiment.map50 = metrics.get("metrics/mAP50(B)")
                experiment.map50_95 = metrics.get("metrics/mAP50-95(B)")
                session.commit()
        except Exception as e:
            experiment = session.query(Experiment).filter(Experiment.id == exp_id).first()
            if experiment:
                experiment.status = "failed"
                experiment.notes = str(e)
                session.commit()
        finally:
            session.close()

    background_tasks.add_task(run_training, exp.id)
    return exp


@router.get("/experiments", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).order_by(Experiment.started_at.desc()).limit(50).all()
