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
from pydantic import BaseModel, Field, field_validator, model_validator

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


class ConfigError(Exception):
    """Raised for configuration problems the user must fix before running."""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value else default
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {value!r}") from None


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    try:
        return float(value) if value else default
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}") from None


def _env_path(name: str, default: str) -> Path:
    """Relative paths are resolved against the project dir, so cron jobs work."""
    path = Path(_env(name) or default).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


class Settings(BaseModel):
    """Runtime settings and secrets, read from environment variables."""

    # --- Search ---------------------------------------------------------
    search_provider: Literal["duckduckgo", "brave", "serpapi"] = "duckduckgo"
    serpapi_api_key: str = ""
    brave_api_key: str = ""
    search_delay: float = 2.0  # seconds between queries, to avoid rate limits
    fetch_pages: bool = True
    page_char_limit: int = 6000
    request_timeout: int = 15

    # --- LLM ------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    llm_model: str = ""
    llm_max_retries: int = 4
    anthropic_api_key: str = ""
    anthropic_api_url: str = "https://api.anthropic.com"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_host: str = "http://localhost:11434"

    # --- Notifications --------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_api_url: str = "https://api.telegram.org"
    discord_webhook_url: str = ""
    slack_webhook_url: str = ""

    # --- Storage --------------------------------------------------------
    database_path: Path = BASE_DIR / "leads.db"
    criteria_path: Path = BASE_DIR / "criteria.yaml"

    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    def problems(self) -> list[str]:
        """Settings that will make a pipeline run fail, as human-readable hints."""
        issues = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            issues.append("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        if self.llm_provider == "openai" and not self.openai_api_key:
            issues.append("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
        if self.search_provider == "serpapi" and not self.serpapi_api_key:
            issues.append("SEARCH_PROVIDER=serpapi but SERPAPI_API_KEY is empty")
        if self.search_provider == "brave" and not self.brave_api_key:
            issues.append("SEARCH_PROVIDER=brave but BRAVE_API_KEY is empty")
        if bool(self.telegram_bot_token) != bool(self.telegram_chat_id):
            issues.append("Telegram needs both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return issues

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            return cls(
                search_provider=_env("SEARCH_PROVIDER", "duckduckgo").lower(),
                serpapi_api_key=_env("SERPAPI_API_KEY"),
            brave_api_key=_env("BRAVE_API_KEY"),
                search_delay=_env_float("SEARCH_DELAY", 2.0),
                fetch_pages=_env("FETCH_PAGES", "true").lower() in {"1", "true", "yes", "on"},
                page_char_limit=_env_int("PAGE_CHAR_LIMIT", 6000),
                request_timeout=_env_int("REQUEST_TIMEOUT", 15),
                llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
                llm_model=_env("LLM_MODEL"),
                llm_max_retries=_env_int("LLM_MAX_RETRIES", 4),
                anthropic_api_key=_env("ANTHROPIC_API_KEY"),
                anthropic_api_url=_env("ANTHROPIC_API_URL", "https://api.anthropic.com"),
                openai_api_key=_env("OPENAI_API_KEY"),
                openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
                telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
                telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
                telegram_api_url=_env("TELEGRAM_API_URL", "https://api.telegram.org"),
                discord_webhook_url=_env("DISCORD_WEBHOOK_URL"),
                slack_webhook_url=_env("SLACK_WEBHOOK_URL"),
                database_path=_env_path("DATABASE_PATH", "leads.db"),
                criteria_path=_env_path("CRITERIA_PATH", "criteria.yaml"),
            )
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            raise ConfigError(f"Invalid .env setting:\n{exc}") from None


class RubricItem(BaseModel):
    """One scoring dimension, e.g. 'pain_points' with a weight and guidance."""

    name: str = Field(min_length=1)
    weight: float = Field(gt=0)
    description: str

    @field_validator("name")
    @classmethod
    def _key(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


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
    # "domain": one lead per website (best for businesses).
    # "url": every page is its own lead (best for forums / job posts).
    # Shared platforms (reddit.com, linkedin.com, ...) always use "url".
    dedupe_scope: Literal["domain", "url"] = "domain"
    max_llm_calls_per_run: int = Field(default=100, ge=1)
    blocked_domains: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)

    @field_validator("queries")
    @classmethod
    def _strip_queries(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty query is required")
        return list(dict.fromkeys(cleaned))  # drop duplicate queries, keep order

    @field_validator("blocked_domains", "exclude_keywords")
    @classmethod
    def _lowercase(cls, values: list[str]) -> list[str]:
        return [v.lower().strip().removeprefix("www.") for v in values if v and v.strip()]

    @model_validator(mode="after")
    def _check(self) -> "Criteria":
        names = [r.name for r in self.rubric]
        if len(names) != len(set(names)):
            raise ValueError(f"rubric names must be unique, got {names}")
        if self.notify_threshold < self.min_score_to_save:
            raise ValueError("notify_threshold must be >= min_score_to_save")
        return self

    @classmethod
    def load(cls, path: Path) -> "Criteria":
        if not path.exists():
            raise ConfigError(
                f"Criteria file not found: {path}\n"
                "Create it with:  cp criteria.example.yaml criteria.yaml"
            )
        try:
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return cls.model_validate(data)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path.name} is not valid YAML:\n{exc}") from None
        except ValueError as exc:
            raise ConfigError(f"{path.name} has invalid settings:\n{exc}") from None
