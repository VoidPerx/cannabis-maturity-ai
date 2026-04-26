"""
Inferencia dual:
- Trichome detector   → YOLOv8 detect  → bounding boxes (clear/milky/amber)
- Stigma segmentor    → YOLOv8 segment → máscaras COCO (white/orange stigma)
- MaturityClassifier  → reglas + features → MaturityStage final
"""
import time
import uuid
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from app.core.config import settings
from app.db.models import MaturityStage


TRICHOME_CLASSES = {0: "clear", 1: "milky", 2: "amber"}
STIGMA_CLASSES = {0: "white_stigma", 1: "orange_stigma"}


class InferenceService:
    def __init__(self):
        self._trichome_model: Optional[YOLO] = None
        self._stigma_model: Optional[YOLO] = None

    def _get_yolo(self):
        try:
            from ultralytics import YOLO
            return YOLO
        except ImportError:
            raise RuntimeError("Ultralytics no instalado. Ejecuta: pip install ultralytics")

    def _load_trichome_model(self):
        if self._trichome_model is None:
            YOLO = self._get_yolo()
            path = settings.TRICHOME_MODEL_PATH
            if Path(path).exists():
                self._trichome_model = YOLO(str(path))
            else:
                self._trichome_model = YOLO("yolov8n.pt")
        return self._trichome_model

    def _load_stigma_model(self):
        if self._stigma_model is None:
            YOLO = self._get_yolo()
            path = settings.STIGMA_MODEL_PATH
            if Path(path).exists():
                self._stigma_model = YOLO(str(path))
            else:
                self._stigma_model = YOLO("yolov8n-seg.pt")
        return self._stigma_model

    def reload_models(self):
        self._trichome_model = None
        self._stigma_model = None

    def detect_trichomes(self, image_path: str) -> dict:
        """Detección de trichomas: bounding boxes por clase."""
        model = self._load_trichome_model()
        t0 = time.perf_counter()
        results = model.predict(
            source=image_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        detections = []
        counts = {"clear": 0, "milky": 0, "amber": 0}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = TRICHOME_CLASSES.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })
                if cls_name in counts:
                    counts[cls_name] += 1

        total = sum(counts.values()) or 1
        return {
            "detections": detections,
            "counts": counts,
            "amber_pct": round(counts["amber"] / total * 100, 1),
            "milky_pct": round(counts["milky"] / total * 100, 1),
            "clear_pct": round(counts["clear"] / total * 100, 1),
            "inference_ms": round(elapsed, 1),
        }

    def segment_stigmas(self, image_path: str) -> dict:
        """Segmentación de estigmas: máscaras por clase."""
        model = self._load_stigma_model()
        t0 = time.perf_counter()
        results = model.predict(
            source=image_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        masks_data = []
        pixel_counts = {"white_stigma": 0, "orange_stigma": 0}

        for r in results:
            if r.masks is None:
                continue
            for i, mask in enumerate(r.masks):
                cls_id = int(r.boxes.cls[i])
                cls_name = STIGMA_CLASSES.get(cls_id, f"class_{cls_id}")
                conf = float(r.boxes.conf[i])
                # Polígono como lista de puntos
                polygon = mask.xy[0].tolist() if mask.xy else []
                pixel_area = int(mask.data[0].sum().item())
                masks_data.append({
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "polygon": polygon,
                    "pixel_area": pixel_area,
                })
                if cls_name in pixel_counts:
                    pixel_counts[cls_name] += pixel_area

        total_pixels = sum(pixel_counts.values()) or 1
        return {
            "masks": masks_data,
            "pixel_counts": pixel_counts,
            "orange_pct": round(pixel_counts["orange_stigma"] / total_pixels * 100, 1),
            "white_pct": round(pixel_counts["white_stigma"] / total_pixels * 100, 1),
            "inference_ms": round(elapsed, 1),
        }

    def classify_maturity(self, trichome_result: dict, stigma_result: dict) -> tuple[MaturityStage, float]:
        """
        Reglas basadas en el paper:
        - Estigmas naranjos >70%  → HARVEST_READY
        - Estigmas naranjos 50-70% + tricomas ámbar >30% → LATE
        - Estigmas naranjos 20-50% → MID
        - Estigmas naranjos <20%  → EARLY

        Los estigmas son mejor predictor que tricomas (Lara et al.).
        """
        orange_pct = stigma_result.get("orange_pct", 0)
        amber_pct = trichome_result.get("amber_pct", 0)

        if orange_pct >= 70:
            stage = MaturityStage.HARVEST_READY
            confidence = min(0.95, 0.70 + orange_pct / 1000)
        elif orange_pct >= 50 and amber_pct >= 30:
            stage = MaturityStage.LATE
            confidence = 0.80
        elif orange_pct >= 20:
            stage = MaturityStage.MID
            confidence = 0.72
        else:
            stage = MaturityStage.EARLY
            confidence = 0.65 + (20 - orange_pct) / 100

        return stage, round(min(confidence, 0.99), 3)

    def annotate_image(
        self,
        image_path: str,
        trichome_result: dict,
        stigma_result: dict,
        maturity_stage: MaturityStage,
        confidence: float,
    ) -> str:
        """Genera imagen anotada con bboxes, máscaras y label de madurez."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer imagen: {image_path}")

        h, w = img.shape[:2]
        overlay = img.copy()

        # Colores por clase
        STIGMA_COLORS = {"white_stigma": (200, 200, 200), "orange_stigma": (0, 140, 255)}
        TRICHOME_COLORS = {"clear": (200, 255, 200), "milky": (255, 255, 150), "amber": (0, 165, 255)}
        STAGE_COLORS = {
            MaturityStage.EARLY: (0, 200, 0),
            MaturityStage.MID: (0, 200, 200),
            MaturityStage.LATE: (0, 165, 255),
            MaturityStage.HARVEST_READY: (0, 0, 255),
        }

        # Dibujar máscaras de estigmas
        for m in stigma_result.get("masks", []):
            pts = np.array(m["polygon"], dtype=np.int32)
            if pts.size > 0:
                color = STIGMA_COLORS.get(m["class"], (128, 128, 128))
                cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # Dibujar bounding boxes de trichomas
        for det in trichome_result.get("detections", []):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = TRICHOME_COLORS.get(det["class"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Banner de resultado
        color = STAGE_COLORS.get(maturity_stage, (255, 255, 255))
        label = f"{maturity_stage.value.upper().replace('_', ' ')} | {confidence:.0%}"
        cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Stats sidebar
        stats = [
            f"Stigmas naranja: {stigma_result.get('orange_pct', 0):.1f}%",
            f"Tricomas ambar: {trichome_result.get('amber_pct', 0):.1f}%",
            f"Tricomas lechoso: {trichome_result.get('milky_pct', 0):.1f}%",
        ]
        for i, s in enumerate(stats):
            cv2.putText(img, s, (10, h - 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        out_path = str(settings.PROCESSED_DIR / f"annotated_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(out_path, img)
        return out_path

    def run_full_pipeline(self, image_path: str) -> dict:
        """Ejecuta trichome detection + stigma segmentation + clasificación + anotación."""
        trichome_result = self.detect_trichomes(image_path)
        stigma_result = self.segment_stigmas(image_path)
        stage, confidence = self.classify_maturity(trichome_result, stigma_result)
        annotated_path = self.annotate_image(image_path, trichome_result, stigma_result, stage, confidence)

        return {
            "maturity_stage": stage,
            "confidence": confidence,
            "trichomes": trichome_result,
            "stigmas": stigma_result,
            "annotated_image": annotated_path,
            "total_inference_ms": trichome_result["inference_ms"] + stigma_result["inference_ms"],
        }


inference_service = InferenceService()
