"""Application configuration management."""

from typing import Any
from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "MXMAP-X Backend"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database Configuration
    DATABASE_URL: PostgresDsn
    DATABASE_ECHO: bool = False

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # ML Model Configuration
    MODEL_CONFIG_PATH: str = "config/model_config.yaml"
    MODEL_CACHE_DIR: str = "models/cache"

    # Logging
    LOG_LEVEL: str = "INFO"

    @property
    def database_url_str(self) -> str:
        """Get database URL as string."""
        return str(self.DATABASE_URL)


# Global settings instance
settings = Settings()
