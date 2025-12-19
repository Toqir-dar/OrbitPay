from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # -------------------------
    # API Settings
    # -------------------------
    APP_NAME: str = "OrbitPay Credit Score Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # -------------------------
    # Database
    # -------------------------
    DATABASE_URL: str = "sqlite:///./predictions.db"

    # -------------------------
    # ML Model Paths (PRODUCTION ONLY)
    # -------------------------
    ML_BASE_DIR: Path = Path("ml_models")
    PRODUCTION_DIR: Path = ML_BASE_DIR / "artifacts" / "production"

    BEST_MODEL_PATH: Path = PRODUCTION_DIR / "xgbosst_tuned.pkl"
    SCALER_PATH: Path = PRODUCTION_DIR / "scaler.pkl"
    ENCODER_PATH: Path = PRODUCTION_DIR / "encoder.pkl"
    TARGET_MAPPING_PATH: Path = PRODUCTION_DIR / "target_mapping.pkl"

    # -------------------------
    # API Limits
    # -------------------------
    MAX_PREDICTIONS_PER_REQUEST: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
