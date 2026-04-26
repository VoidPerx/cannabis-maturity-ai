import uuid, shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Sample, Prediction
from app.services.inference_service import inference_service
from app.models.schemas import PredictionResponse, SampleCreate
from app.core.config import settings
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post("/", response_model=PredictionResponse)
async def predict_maturity(
    file: UploadFile = File(...),
    plant_id: str = Form(None),
    strain: str = Form(None),
    week_of_flower: int = Form(None),
    camera_model: str = Form(None),
    db: Session = Depends(get_db),
):
    """
    Sube una imagen y retorna el estado de madurez detectado.
    Ejecuta trichome detection + stigma segmentation en paralelo.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se aceptan imágenes (jpg, png, webp)")

    # Guardar imagen
    sample_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".jpg"
    save_path = settings.UPLOAD_DIR / f"{sample_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Registro en DB
    sample = Sample(
        id=sample_id,
        filename=file.filename,
        filepath=str(save_path),
        plant_id=plant_id,
        strain=strain,
        week_of_flower=week_of_flower,
        camera_model=camera_model,
    )
    db.add(sample)
    db.commit()

    # Inferencia
    try:
        result = inference_service.run_full_pipeline(str(save_path))
    except Exception as e:
        db.delete(sample)
        db.commit()
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error en inferencia: {str(e)}")

    # Guardar predicción
    pred = Prediction(
        sample_id=sample_id,
        model_version=str(settings.TRICHOME_MODEL_PATH),
        predicted_stage=result["maturity_stage"],
        confidence=result["confidence"],
        trichome_detections=result["trichomes"].get("detections"),
        trichome_amber_count=result["trichomes"]["counts"].get("amber", 0),
        trichome_milky_count=result["trichomes"]["counts"].get("milky", 0),
        trichome_clear_count=result["trichomes"]["counts"].get("clear", 0),
        stigma_masks=result["stigmas"].get("masks"),
        orange_stigma_pct=result["stigmas"]["orange_pct"],
        white_stigma_pct=result["stigmas"]["white_pct"],
        annotated_image_path=result["annotated_image"],
        inference_time_ms=result["total_inference_ms"],
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return PredictionResponse(
        id=pred.id,
        sample_id=sample_id,
        maturity_stage=result["maturity_stage"],
        confidence=result["confidence"],
        trichomes=result["trichomes"],
        stigmas=result["stigmas"],
        annotated_image_url=f"/images/processed/{Path(result['annotated_image']).name}",
        total_inference_ms=result["total_inference_ms"],
        predicted_at=pred.predicted_at,
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(404, "Predicción no encontrada")
    return pred
