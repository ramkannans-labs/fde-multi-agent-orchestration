"""Configuration management for multi-agent orchestration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude")
    llm_model: str = Field(
        default="anthropic:claude-3-5-sonnet-20241022",
        description="PydanticAI model string (e.g., anthropic:claude-3-5-sonnet-20241022)",
    )

    # Agent Configuration
    max_turns: int = Field(default=10, description="Maximum conversation turns per task")
    enable_human_in_loop: bool = Field(default=True, description="Pause for human approval at checkpoints")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")


settings = Settings()


def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings