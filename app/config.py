"""
Centralized configuration via environment variables.
Uses pydantic-settings to validate and type-check all config values.
"""

from __future__ import annotations

import json
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────
    PORT: int = 8080
    ENVIRONMENT: str = "development"  # "development" | "production"

    # ── ONNX Model Paths ────────────────────────────────────────────────
    ROAST_MODEL_PATH: str = "checkpoints/roast_model.onnx"
    DEFECT_MODEL_PATH: str = "checkpoints/defect_model.onnx"

    # ── Supabase PostgreSQL ─────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///data/predictions.db"

    # ── Supabase Storage ────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_BUCKET: str = "coffee-uploads"

    # ── Security ────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:8080"]
    RATE_LIMIT: str = "30/minute"

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Accept both JSON strings and Python lists."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
