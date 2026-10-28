"""Configuration loading.

Two sources of configuration:

* ``.env``          -> secrets and runtime settings (API keys, webhooks, paths).
* ``criteria.yaml`` -> *what* to look for: search queries, the Ideal Customer
                       Profile (ICP), the scoring rubric and thresholds.

Keeping the ICP in YAML means you can tune targeting without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


class Settings(BaseModel):
    """Runtime settings and secrets, read from environment variables."""

    # --- Search ---------------------------------------------------------
    search_provider: Literal["duckduckgo", "serpapi"] = "duckduckgo"
    serpapi_api_key: str = ""
    fetch_pages: bool = True
    page_char_limit: int = 6000
    request_timeout: int = 15

    # --- LLM ------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    llm_model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_host: str = "http://localhost:11434"

    # --- Notifications --------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    slack_webhook_url: str = ""

    # --- Storage --------------------------------------------------------
    database_path: Path = BASE_DIR / "leads.db"
    criteria_path: Path = BASE_DIR / "criteria.yaml"

    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-sonnet-5-5",
            "openai": "gpt-4o-mini",
            "ollama": "llama3.1",
        }[self.llm_provider]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            search_provider=_env("SEARCH_PROVIDER", "duckduckgo").lower(),
            serpapi_api_key=_env("SERPAPI_API_KEY"),
            fetch_pages=_env("FETCH_PAGES", "true").lower() in {"1", "true", "yes"},
            page_char_limit=_env_int("PAGE_CHAR_LIMIT", 6000),
            request_timeout=_env_int("REQUEST_TIMEOUT", 15),
            llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
            llm_model=_env("LLM_MODEL"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
            discord_webhook_url=_env("DISCORD_WEBHOOK_URL"),
            slack_webhook_url=_env("SLACK_WEBHOOK_URL"),
            database_path=Path(_env("DATABASE_PATH") or BASE_DIR / "leads.db"),
            criteria_path=Path(_env("CRITERIA_PATH") or BASE_DIR / "criteria.yaml"),
        )


class RubricItem(BaseModel):
    """One scoring dimension, e.g. 'pain_points' with a weight and guidance."""

    name: str
    weight: float = Field(gt=0)
    description: str


class Criteria(BaseModel):
    """Targeting criteria loaded from ``criteria.yaml``."""

    campaign_name: str = "default"
    offer: str
    queries: list[str] = Field(min_length=1)
    results_per_query: int = Field(default=10, ge=1, le=50)
    region: str = "wt-wt"
    icp: str
    rubric: list[RubricItem] = Field(min_length=1)
    min_score_to_save: int = Field(default=5, ge=1, le=10)
    notify_threshold: int = Field(default=8, ge=1, le=10)
    blocked_domains: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)

    @field_validator("blocked_domains", "exclude_keywords")
    @classmethod
    def _lowercase(cls, values: list[str]) -> list[str]:
        return [v.lower().strip() for v in values if v.strip()]

    @classmethod
    def load(cls, path: Path) -> "Criteria":
        if not path.exists():
            raise FileNotFoundError(
                f"Criteria file not found: {path}. "
                "Copy criteria.example.yaml to criteria.yaml and edit it."
            )
        with path.open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
