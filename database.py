"""Lightweight SQLite storage with duplicate blocking by URL and by name.

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
from urllib.parse import urlparse

from analyzer import LeadScore
from searcher import Prospect

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
    status           TEXT NOT NULL DEFAULT 'new',  -- rejected | qualified | notified | contacted
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

_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|ref$|source$)")
_NAME_NOISE = re.compile(
    r"\b(inc|llc|ltd|limited|co|corp|corporation|company|gmbh|plc|the)\b\.?", re.I
)


def normalize_url(url: str) -> str:
    """Canonical form used for duplicate detection."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or ""
    query = "&".join(
        p for p in sorted(parsed.query.split("&")) if p and not _TRACKING_PARAMS.match(p)
    )
    return f"{host}{path}" + (f"?{query}" if query else "")


def normalize_name(name: str) -> str:
    cleaned = _NAME_NOISE.sub(" ", name.lower())
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LeadDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------- duplicate check
    def is_known_url(self, url: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM leads WHERE url_key = ?", (normalize_url(url),)
        ).fetchone()
        return row is not None

    def is_known_name(self, name: str) -> bool:
        key = normalize_name(name)
        if len(key) < 3:  # too short to be a meaningful identity
            return False
        row = self.conn.execute(
            "SELECT 1 FROM leads WHERE name_key = ? AND status != 'rejected'", (key,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------ writes
    def save(self, prospect: Prospect, result: LeadScore, status: str) -> int:
        name = result.lead_name or prospect.title
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
        return cur.lastrowid

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
    def pending_notifications(self, threshold: int) -> list[sqlite3.Row]:
        """Qualified leads above threshold whose ding failed or never ran."""
        return self.conn.execute(
            "SELECT * FROM leads WHERE status = 'qualified' AND score >= ? ORDER BY score DESC",
            (threshold,),
        ).fetchall()

    def list_leads(self, min_score: int = 1, status: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM leads WHERE score >= ?"
        params: list = [min_score]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY score DESC, created_at DESC"
        return self.conn.execute(sql, params).fetchall()

    def export_csv(self, out_path: Path, min_score: int = 1) -> int:
        rows = self.list_leads(min_score=min_score)
        columns = [
            "id", "name", "url", "score", "status", "reasoning", "pain_points",
            "budget_signals", "contact_hints", "suggested_opener", "query",
            "created_at", "notified_at", "contacted_at",
        ]
        with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[c] for c in columns])
        return len(rows)
