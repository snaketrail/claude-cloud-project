"""The "ding": push high-score leads to Telegram, Discord and/or Slack."""

from __future__ import annotations

import json
import logging
import time
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
    lead_id: int | None = None
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
            lead_id=row["id"],
            pain_points=json.loads(row["pain_points"] or "[]"),
            budget_signals=json.loads(row["budget_signals"] or "[]"),
            contact_hints=json.loads(row["contact_hints"] or "[]"),
            suggested_opener=row["suggested_opener"] or "",
        )

    def to_text(self) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"  • {_truncate(i, 200)}" for i in items[:5]) or "  • (none found)"

        lines = [
            f"🔔 New lead: {self.name} — score {self.score}/10",
            self.url,
            "",
            f"Why: {_truncate(self.reasoning, 600)}",
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
        if self.lead_id is not None:
            lines.append(f"Lead #{self.lead_id} · mark done: python main.py contacted {self.lead_id}")
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
            print("\n" + alert.to_text() + "\n", flush=True)
            return False

        text = alert.to_text()
        delivered = False
        for channel in self.channels:
            try:
                getattr(self, f"_send_{channel}")(text)
                delivered = True
                log.info("Dinged '%s' via %s", alert.name, channel)
            except requests.RequestException as exc:
                log.error("%s notification failed: %s", channel, _redact(str(exc), self.settings))
        return delivered

    def _post(self, url: str, payload: dict) -> requests.Response:
        """POST, honouring one rate-limit (429) retry."""
        for attempt in range(2):
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 429 and attempt == 0:
                wait = _retry_after(resp)
                log.warning("Rate limited by notification API; waiting %.0fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp)
            return resp
        raise requests.HTTPError("rate limited twice")  # pragma: no cover

    def _send_telegram(self, text: str) -> None:
        self._post(
            f"{self.settings.telegram_api_url.rstrip('/')}/bot{self.settings.telegram_bot_token}/sendMessage",
            {
                "chat_id": self.settings.telegram_chat_id,
                "text": _truncate(text, 4000),  # limit is 4096 UTF-16 units; emoji count double
                "disable_web_page_preview": True,
            },
        )

    def _send_discord(self, text: str) -> None:
        self._post(
            self.settings.discord_webhook_url,
            # Scraped text could contain "@everyone"; never let it ping anyone.
            {"content": _truncate(text, 1990), "allowed_mentions": {"parse": []}},
        )

    def _send_slack(self, text: str) -> None:
        # Escape Slack control sequences so scraped text can't trigger <!channel> etc.
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._post(self.settings.slack_webhook_url, {"text": _truncate(safe, 39000)})


def _retry_after(resp: requests.Response) -> float:
    value = resp.headers.get("retry-after")
    if not value:
        try:  # Telegram puts it in the body; Discord also sends it in JSON
            body = resp.json()
            value = body.get("parameters", {}).get("retry_after") or body.get("retry_after")
        except ValueError:
            value = None
    try:
        return max(1.0, min(30.0, float(value)))
    except (TypeError, ValueError):
        return 5.0


def _redact(message: str, settings: Settings) -> str:
    """Keep bot tokens and webhook secrets out of logs."""
    for secret in (settings.telegram_bot_token, settings.discord_webhook_url, settings.slack_webhook_url):
        if secret:
            message = message.replace(secret, "***")
    return message
