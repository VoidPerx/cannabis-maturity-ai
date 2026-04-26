from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Cannabis Maturity AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Roboflow
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_WORKSPACE: str = ""
    ROBOFLOW_TRICHOME_PROJECT: str = "cannabis-trichomes"
    ROBOFLOW_STIGMA_PROJECT: str = "cannabis-stigmas"

    # Database
    DATABASE_URL: str = "sqlite:///./data/cannabis_maturity.db"

    # Storage paths
    UPLOAD_DIR: Path = Path("./data/raw")
    PROCESSED_DIR: Path = Path("./data/processed")
    MODELS_DIR: Path = Path("./data/models")
    EXPORTS_DIR: Path = Path("./data/exports")

    # MLflow
    MLFLOW_TRACKING_URI: str = "./data/mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "cannabis-maturity-detection"

    # Model
    TRICHOME_MODEL_PATH: Path = Path("./data/models/trichome_detect.pt")
    STIGMA_MODEL_PATH: Path = Path("./data/models/stigma_seg.pt")
    CONFIDENCE_THRESHOLD: float = 0.45
    IOU_THRESHOLD: float = 0.45

    # External datasets
    HUGGINGFACE_DATASET: str = "siccan/tricomas-semillero-cannabis"

    class Config:
        env_file = ".env"
        extra = "ignore"

    def ensure_dirs(self):
        for d in [self.UPLOAD_DIR, self.PROCESSED_DIR, self.MODELS_DIR, self.EXPORTS_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
