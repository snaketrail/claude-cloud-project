"""Lightweight SQLite storage with duplicate blocking by URL, domain and name.

Every prospect that gets scored is stored -- including rejected ones -- so the
pipeline never pays to score the same page twice and never dings you twice.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

from analyzer import LeadScore
from searcher import Prospect, domain_of

STATUSES = ("rejected", "duplicate", "qualified", "notified", "contacted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url              TEXT NOT NULL,
    url_key          TEXT NOT NULL UNIQUE,
    domain           TEXT NOT NULL,
    name             TEXT,
    name_key         TEXT,
    query            TEXT,
    score            INTEGER,
    status           TEXT NOT NULL,
    reasoning        TEXT,
    pain_points      TEXT,  -- JSON list
    budget_signals   TEXT,  -- JSON list
    contact_hints    TEXT,  -- JSON list
    suggested_opener TEXT,
    created_at       TEXT NOT NULL,
    notified_at      TEXT,
    contacted_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_leads_name_key ON leads(name_key);
CREATE INDEX IF NOT EXISTS idx_leads_domain   ON leads(domain);
CREATE INDEX IF NOT EXISTS idx_leads_status   ON leads(status);
"""

# Sites where many unrelated prospects share one domain; never dedupe by domain.
PLATFORM_DOMAINS = {
    "reddit.com", "linkedin.com", "x.com", "twitter.com", "facebook.com", "instagram.com",
    "youtube.com", "medium.com", "substack.com", "quora.com", "github.com",
    "stackoverflow.com", "stackexchange.com", "news.ycombinator.com", "producthunt.com",
    "indeed.com", "glassdoor.com", "wellfound.com", "angel.co", "upwork.com",
    "fiverr.com", "craigslist.org", "google.com", "sites.google.com", "wordpress.com",
    "wixsite.com", "blogspot.com", "tumblr.com", "discord.com", "t.me",
}

_TRACKING_PARAM = re.compile(r"^(utm_.*|fbclid|gclid|msclkid|mc_[a-z]+|ref|ref_src|_ga|igshid)$", re.I)
_NAME_NOISE = re.compile(
    r"\b(inc|llc|l\.l\.c|ltd|limited|co|corp|corporation|company|gmbh|plc|pty|the)\b\.?", re.I
)


def normalize_url(url: str) -> str:
    """Canonical form used for duplicate detection."""
    parsed = urlparse(url.strip())
    host = domain_of(url)
    path = re.sub(r"/(index|default)\.(html?|php|aspx?)$", "", parsed.path, flags=re.I)
    path = path.rstrip("/")
    params = sorted(
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not _TRACKING_PARAM.match(k)
    )
    query = urlencode(params)
    return f"{host}{path}" + (f"?{query}" if query else "")


def normalize_name(name: str) -> str:
    cleaned = _NAME_NOISE.sub(" ", name.lower().replace("&", " and "))
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def is_platform(domain: str) -> bool:
    return any(domain == p or domain.endswith("." + p) for p in PLATFORM_DOMAINS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LeadDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LeadDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------- duplicate check
    def is_known(self, url: str, scope: str = "domain") -> bool:
        """Has this prospect (by URL, or by website when scope='domain') been seen?"""
        if self._exists("url_key = ?", normalize_url(url)):
            return True
        domain = domain_of(url)
        if scope == "domain" and not is_platform(domain):
            return self._exists("domain = ?", domain)
        return False

    def is_known_name(self, name: str) -> bool:
        key = normalize_name(name)
        if len(key) < 4:  # too short to be a meaningful identity
            return False
        return self._exists("name_key = ? AND status IN ('qualified', 'notified', 'contacted')", key)

    def _exists(self, where: str, value: str) -> bool:
        return self.conn.execute(f"SELECT 1 FROM leads WHERE {where} LIMIT 1", (value,)).fetchone() is not None

    # ------------------------------------------------------------------ writes
    def save(self, prospect: Prospect, result: LeadScore, status: str) -> int | None:
        """Insert a lead. Returns its id, or None if the URL was already stored."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        name = result.lead_name or prospect.title or prospect.domain
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO leads
               (url, url_key, domain, name, name_key, query, score, status, reasoning,
                pain_points, budget_signals, contact_hints, suggested_opener, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prospect.url,
                normalize_url(prospect.url),
                prospect.domain,
                name,
                normalize_name(name),
                prospect.query,
                result.score,
                status,
                result.reasoning,
                json.dumps(result.pain_points),
                json.dumps(result.budget_signals),
                json.dumps(result.contact_hints),
                result.suggested_opener,
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid if cur.rowcount else None

    def mark_notified(self, lead_id: int) -> None:
        self.conn.execute(
            "UPDATE leads SET status = 'notified', notified_at = ? WHERE id = ?",
            (_now(), lead_id),
        )
        self.conn.commit()

    def mark_contacted(self, lead_id: int) -> bool:
        cur = self.conn.execute(
            "UPDATE leads SET status = 'contacted', contacted_at = ? WHERE id = ?",
            (_now(), lead_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------- reads
    def get(self, lead_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()

    def pending_notifications(self, threshold: int) -> list[sqlite3.Row]:
        """Qualified leads above threshold whose ding failed or never ran."""
        return self.conn.execute(
            "SELECT * FROM leads WHERE status = 'qualified' AND score >= ? ORDER BY score DESC, id",
            (threshold,),
        ).fetchall()

    def list_leads(self, min_score: int = 1, status: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM leads WHERE score >= ?"
        params: list = [min_score]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY score DESC, id DESC"
        return self.conn.execute(sql, params).fetchall()

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM leads GROUP BY status").fetchall()
        return {status: n for status, n in rows}

    def export_csv(self, out_path: Path, min_score: int = 1, status: str | None = None) -> int:
        rows = self.list_leads(min_score=min_score, status=status)
        columns = [
            "id", "name", "url", "score", "status", "reasoning", "pain_points",
            "budget_signals", "contact_hints", "suggested_opener", "query",
            "created_at", "notified_at", "contacted_at",
        ]
        list_columns = {"pain_points", "budget_signals", "contact_hints"}
        with Path(out_path).open("w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel-friendly
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(
                    ["; ".join(json.loads(row[c] or "[]")) if c in list_columns else row[c]
                     for c in columns]
                )
        return len(rows)
