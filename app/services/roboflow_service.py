"""
Roboflow integration:
- Subir imágenes para anotación
- Descargar versiones de dataset (detect/segment)
- Disparar augmentación y re-entrenamiento hosted
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from app.core.config import settings


class RoboflowService:
    def __init__(self):
        self._rf: Optional[Roboflow] = None

    def _get_client(self):
        if not self._rf:
            try:
                from roboflow import Roboflow
            except ImportError:
                raise RuntimeError("Roboflow no instalado. Ejecuta: pip install roboflow")
            if not settings.ROBOFLOW_API_KEY:
                raise ValueError("ROBOFLOW_API_KEY no configurada en .env")
            self._rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
        return self._rf

    def get_trichome_project(self):
        rf = self._get_client()
        workspace = rf.workspace(settings.ROBOFLOW_WORKSPACE)
        return workspace.project(settings.ROBOFLOW_TRICHOME_PROJECT)

    def get_stigma_project(self):
        rf = self._get_client()
        workspace = rf.workspace(settings.ROBOFLOW_WORKSPACE)
        return workspace.project(settings.ROBOFLOW_STIGMA_PROJECT)

    def upload_image(
        self,
        image_path: str,
        task: str = "trichome",  # "trichome" | "stigma"
        split: str = "train",
        annotation_path: Optional[str] = None,
    ) -> dict:
        """Sube imagen a Roboflow para anotación."""
        project = self.get_trichome_project() if task == "trichome" else self.get_stigma_project()
        result = project.upload(
            image_path=image_path,
            annotation_path=annotation_path,
            split=split,
            num_retry_uploads=3,
            batch_name="auto-upload",
            tag_names=[task, split],
            is_prediction=False,
        )
        return result

    def download_dataset(
        self,
        task: str = "trichome",
        version: int = 1,
        fmt: str = "yolov8",
        location: Optional[str] = None,
    ) -> Path:
        """Descarga dataset versionado listo para YOLOv8."""
        project = self.get_trichome_project() if task == "trichome" else self.get_stigma_project()
        dataset_version = project.version(version)

        dest = Path(location) if location else settings.PROCESSED_DIR / task / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)

        dataset = dataset_version.download(fmt, location=str(dest), overwrite=True)
        return Path(dataset.location)

    def get_dataset_stats(self, task: str = "trichome", version: int = 1) -> dict:
        """Retorna stats de una versión del dataset."""
        project = self.get_trichome_project() if task == "trichome" else self.get_stigma_project()
        v = project.version(version)
        return {
            "project": project.name,
            "version": version,
            "created": str(v.created),
            "images": v.images,
            "splits": v.splits,
            "classes": v.classes,
            "preprocessing": v.preprocessing,
            "augmentation": v.augmentation,
        }

    def list_versions(self, task: str = "trichome") -> list[dict]:
        project = self.get_trichome_project() if task == "trichome" else self.get_stigma_project()
        versions = project.get_version_information()
        return [
            {
                "version": v.get("id", "").split("/")[-1],
                "created": v.get("created"),
                "images": v.get("images"),
                "classes": v.get("classes"),
            }
            for v in versions
        ]


roboflow_service = RoboflowService()
