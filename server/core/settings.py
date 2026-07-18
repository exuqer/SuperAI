"""Application settings and configuration."""

from __future__ import annotations

from pydantic import Field
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
    app_title: str = "SuperAI Role-Free Event Graph API"
    app_version: str = "2.7.0"
    debug: bool = False
    load_demo_knowledge: bool = False
    load_domain_pack: str | None = None
    allow_manual_seeds: bool = False
    retrieval_weights: dict[str, float] = Field(default_factory=lambda: {
        "predicate_compatibility": 0.24,
        "known_node_compatibility": 0.22,
        "required_component_compatibility": 0.14,
        "gap_slot_compatibility": 0.14,
        "relation_compatibility": 0.08,
        "polarity_compatibility": 0.06,
        "context_compatibility": 0.04,
        "evidence_confidence": 0.08,
    })
    exact_retrieval_stop_threshold: float = 0.85
    concept_retrieval_stop_threshold: float = 0.75
    retrieval_stage_limit: int = 64

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
