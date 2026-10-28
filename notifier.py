"""The "ding": push high-score leads to Telegram, Discord and/or Slack."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import requests

from config import Settings

log = logging.getLogger(__name__)


@dataclass
class LeadAlert:
    name: str
    url: str
    score: int
    reasoning: str
    query: str
    pain_points: list[str] = field(default_factory=list)
    budget_signals: list[str] = field(default_factory=list)
    contact_hints: list[str] = field(default_factory=list)
    suggested_opener: str = ""

    @classmethod
    def from_row(cls, row) -> "LeadAlert":
        return cls(
            name=row["name"] or row["domain"],
            url=row["url"],
            score=row["score"],
            reasoning=row["reasoning"] or "",
            query=row["query"] or "",
            pain_points=json.loads(row["pain_points"] or "[]"),
            budget_signals=json.loads(row["budget_signals"] or "[]"),
            contact_hints=json.loads(row["contact_hints"] or "[]"),
            suggested_opener=row["suggested_opener"] or "",
        )

    def to_text(self) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"  • {i}" for i in items[:5]) or "  • (none found)"

        lines = [
            f"🔔 New lead: {self.name} — score {self.score}/10",
            self.url,
            "",
            f"Why: {self.reasoning}",
            "",
            "Pain points:",
            bullets(self.pain_points),
            "Budget signals:",
            bullets(self.budget_signals),
        ]
        if self.contact_hints:
            lines += ["Contact:", bullets(self.contact_hints)]
        if self.suggested_opener:
            lines += ["", f"Opener: {self.suggested_opener}"]
        lines += ["", f"Found via: “{self.query}”"]
        return "\n".join(lines)


class Notifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.timeout = settings.request_timeout

    @property
    def channels(self) -> list[str]:
        s = self.settings
        enabled = []
        if s.telegram_bot_token and s.telegram_chat_id:
            enabled.append("telegram")
        if s.discord_webhook_url:
            enabled.append("discord")
        if s.slack_webhook_url:
            enabled.append("slack")
        return enabled

    def ding(self, alert: LeadAlert) -> bool:
        """Send to every configured channel. True if at least one succeeded."""
        if not self.channels:
            log.warning("No notification channel configured; printing alert instead.")
            print("\n" + alert.to_text() + "\n")
            return False

        text = alert.to_text()
        delivered = False
        for channel in self.channels:
            try:
                getattr(self, f"_send_{channel}")(text)
                delivered = True
                log.info("Dinged %s via %s", alert.name, channel)
            except requests.RequestException as exc:
                log.error("%s notification failed: %s", channel, exc)
        return delivered

    def _send_telegram(self, text: str) -> None:
        resp = self.session.post(
            f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage",
            json={
                "chat_id": self.settings.telegram_chat_id,
                "text": text[:4096],
                "disable_web_page_preview": True,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def _send_discord(self, text: str) -> None:
        resp = self.session.post(
            self.settings.discord_webhook_url,
            json={"content": text[:2000]},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def _send_slack(self, text: str) -> None:
        resp = self.session.post(
            self.settings.slack_webhook_url,
            json={"text": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
