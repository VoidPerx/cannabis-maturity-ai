"""
Entrenamiento YOLOv8 con tracking MLflow (opcional).
Descarga dataset de Roboflow, entrena, registra métricas.
"""
from datetime import datetime
from pathlib import Path
from app.core.config import settings
from app.services.roboflow_service import roboflow_service


def train_model(
    task: str = "trichome",
    roboflow_version: int = 1,
    model_arch: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    experiment_name: str = None,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Ultralytics no instalado: pip install ultralytics")

    try:
        import mlflow as _mlflow
        _mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        _mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        use_mlflow = True
    except ImportError:
        _mlflow = None
        use_mlflow = False

    run_name = experiment_name or f"{task}_v{roboflow_version}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    dataset_path = roboflow_service.download_dataset(task=task, version=roboflow_version)
    data_yaml = dataset_path / "data.yaml"

    suffix = "" if task == "trichome" else "-seg"
    base_model = f"{model_arch}{suffix}.pt"

    run_id = "no-mlflow"
    ctx = _mlflow.start_run(run_name=run_name) if use_mlflow else None

    try:
        if use_mlflow and ctx:
            ctx.__enter__()
            _mlflow.log_params({
                "task": task, "roboflow_version": roboflow_version,
                "model_arch": model_arch, "epochs": epochs,
                "imgsz": imgsz, "batch": batch,
            })

        model = YOLO(base_model)
        results = model.train(
            data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
            project=str(settings.MODELS_DIR / task), name=run_name, verbose=True,
        )

        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        elif hasattr(results, "maps"):
            metrics = {"mAP50": float(results.maps[0]), "mAP50-95": float(results.maps[1])}

        if use_mlflow and ctx:
            _mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            run_id = ctx.info.run_id

        best_model_path = Path(settings.MODELS_DIR) / task / run_name / "weights" / "best.pt"
        if use_mlflow and ctx and best_model_path.exists():
            _mlflow.log_artifact(str(best_model_path), artifact_path="model")

    finally:
        if use_mlflow and ctx:
            ctx.__exit__(None, None, None)

    return {
        "run_id": run_id,
        "run_name": run_name,
        "task": task,
        "model_path": str(best_model_path),
        "metrics": metrics,
    }
