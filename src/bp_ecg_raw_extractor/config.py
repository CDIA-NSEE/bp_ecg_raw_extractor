"""Configuration module for bp_ecg_raw_extractor.

Reads all settings from environment variables and optional .env file using
pydantic-settings. No secrets are hardcoded here.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class RegionConfig(BaseModel):
    """Relative coordinate region within an image (0.0 to 1.0 per axis)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables or .env file."""

    # MINIO / S3-compatible object storage
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str

    # Bucket names
    bucket_images: str = "bp-ecg-dev-images"
    bucket_intake: str = "bp-ecg-dev-intake"
    bucket_lake: str = "bp-ecg-dev-lake"

    # Iceberg catalog via Nessie REST
    nessie_uri: str = "http://localhost:19120/iceberg"
    iceberg_table_name: str = "bp_ecg.ecg_records_raw"

    # Processing concurrency
    ocr_workers: int = 2
    async_concurrency: int = 10

    # PaddleOCR
    paddle_use_gpu: bool = False
    paddle_lang: str = "pt"

    # Image regions (relative coords, 0.0–1.0)
    crop_region: RegionConfig = RegionConfig(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    ocr_region: RegionConfig = RegionConfig(x_min=0.0, y_min=0.0, x_max=1.0, y_max=0.15)

    # Prometheus metrics server port
    metrics_port: int = 8000

    # Logging and environment
    log_level: str = "INFO"
    environment: str = "dev"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
