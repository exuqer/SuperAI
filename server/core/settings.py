"""Application settings and configuration."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_path: str = ".superai/state.sqlite"

    # API
    api_prefix: str = "/api/v2"
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # App
    app_title: str = "SuperAI Cloud / Space / Placement API"
    app_version: str = "2.0.0"
    debug: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_mode(cls, value: object) -> object:
        """Accept the conventional deployment labels used by local shells."""
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
