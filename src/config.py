"""
FastAPI configuration for the Titanic ML prediction service.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Settings
    api_title: str = "Titanic ML API"
    api_version: str = "1.0.0"
    api_description: str = "API for Titanic survival prediction using ML models"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Model Settings
    model_path: Path = Field(
        default=Path(__file__).parent.parent / "models" / "titanic_model.pkl",
        env="MODEL_PATH"
    )
    preprocessor_path: Path = Field(
        default=Path(__file__).parent.parent / "models" / "preprocessor.pkl",
        env="PREPROCESSOR_PATH"
    )
    
    # Data Settings
    data_dir: Path = Field(
        default=Path(__file__).parent.parent / "data",
        env="DATA_DIR"
    )
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "json"
    
    # CORS Settings
    cors_origins: list[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # Request/Response Settings
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: int = 30
    
    # Model Configuration
    model_name: str = "RandomForestClassifier"
    model_version: str = "1.0"
    
    # Feature Configuration
    expected_features: list[str] = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
    ]
    
    # Feature mappings
    sex_mapping: dict = {"male": 0, "female": 1}
    embarked_mapping: dict = {"S": 0, "C": 1, "Q": 2}
    title_mapping: dict = {
        "Mr": 0,
        "Miss": 1,
        "Mrs": 2,
        "Master": 3,
        "Rare": 4,
        "Unknown": 4,
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Create global settings instance
settings = Settings()
