from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.database import get_db
from app.db.models import Sample, Prediction, MaturityStage
from app.models.schemas import SampleResponse, DashboardStats
from datetime import datetime

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.get("/", response_model=list[SampleResponse])
def list_samples(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    samples = db.query(Sample).offset(skip).limit(limit).all()
    result = []
    for s in samples:
        result.append(SampleResponse(
            id=s.id,
            filename=s.filename,
            uploaded_at=s.uploaded_at,
            plant_id=s.plant_id,
            strain=s.strain,
            week_of_flower=s.week_of_flower,
            has_annotation=s.annotation is not None,
            prediction_count=len(s.predictions),
        ))
    return result


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    total_samples = db.query(func.count(Sample.id)).scalar()
    annotated = db.query(func.count(Sample.id)).filter(
        Sample.annotation != None  # noqa
    ).scalar()
    total_preds = db.query(func.count(Prediction.id)).scalar()

    # Distribución por etapa
    stage_rows = db.query(
        Prediction.predicted_stage, func.count(Prediction.id)
    ).group_by(Prediction.predicted_stage).all()
    stage_dist = {row[0].value if row[0] else "unknown": row[1] for row in stage_rows}

    avg_conf = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
    last_pred = db.query(func.max(Prediction.predicted_at)).scalar()

    return DashboardStats(
        total_samples=total_samples,
        annotated_samples=annotated,
        total_predictions=total_preds,
        stage_distribution=stage_dist,
        avg_confidence=round(float(avg_conf), 3),
        last_prediction_at=last_pred,
    )


@router.delete("/{sample_id}")
def delete_sample(sample_id: str, db: Session = Depends(get_db)):
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(404, "Muestra no encontrada")
    db.delete(sample)
    db.commit()
    return {"deleted": sample_id}
