"""Offline tests: no network, no API keys. Search and LLM are stubbed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import Analyzer, parse_score  # noqa: E402
from config import Criteria, Settings  # noqa: E402
from database import LeadDatabase, normalize_name, normalize_url  # noqa: E402
from main import Pipeline  # noqa: E402
from searcher import Prospect, extract_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def criteria() -> Criteria:
    return Criteria.load(ROOT / "criteria.example.yaml")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_path=tmp_path / "leads.db", fetch_pages=False)


def test_example_criteria_is_valid(criteria):
    assert criteria.queries
    assert criteria.notify_threshold >= criteria.min_score_to_save


def test_normalize_url_dedupes_variants():
    a = normalize_url("https://www.acme.com/contact/?utm_source=x")
    b = normalize_url("http://acme.com/contact")
    assert a == b == "acme.com/contact"


def test_normalize_name_strips_suffixes():
    assert normalize_name("Acme Plumbing, LLC") == normalize_name("The ACME plumbing")


def test_parse_score_handles_fenced_json_and_clamps():
    raw = '```json\n{"score": 12, "rubric_scores": {"relevance": 0}, "lead_name": "X"}\n```'
    result = parse_score(raw)
    assert result.score == 10
    assert result.rubric_scores["relevance"] == 1


def test_weighted_score(settings, criteria):
    analyzer = Analyzer(settings, criteria)
    score = analyzer.weighted_score(
        {"relevance": 10, "pain_points": 10, "budget_signals": 4, "reachability": 4}
    )
    # (30 + 30 + 8 + 4) / 9 = 8
    assert score == 8


def test_extract_text_flags_missing_viewport():
    text = extract_text("<html><title>Joe's</title><body>Call us <script>x</script></body></html>", 500)
    assert "no mobile viewport" in text
    assert "Call us" in text and "x" not in text.split("Call us")[1]


def test_pipeline_end_to_end(monkeypatch, settings, criteria):
    criteria.queries = ["q"]
    prospects = [
        Prospect("Acme Plumbing", "https://acmeplumbing.com", "family owned", "q"),
        Prospect("Acme Plumbing LLC", "https://acme-plumbing-austin.com", "dup name", "q"),
        Prospect("Big Chain", "https://bigchain.com", "national", "q"),
    ]
    replies = {
        "https://acmeplumbing.com": {"lead_name": "Acme Plumbing", "score": 9,
                                     "rubric_scores": {"relevance": 9, "pain_points": 9,
                                                       "budget_signals": 8, "reachability": 9}},
        "https://acme-plumbing-austin.com": {"lead_name": "Acme Plumbing LLC", "score": 9},
        "https://bigchain.com": {"lead_name": "Big Chain", "score": 2, "is_relevant": False},
    }
    dings = []

    pipeline = Pipeline(settings, criteria)
    monkeypatch.setattr(pipeline.searcher, "search", lambda q: [Prospect(**vars(p)) for p in prospects])
    monkeypatch.setattr(
        pipeline.analyzer, "_complete",
        lambda prompt: json.dumps(next(v for k, v in replies.items() if k in prompt)),
    )
    monkeypatch.setattr(pipeline.notifier, "ding", lambda alert: dings.append(alert) or True)

    stats = pipeline.run_once()
    assert stats.dinged == 1 and stats.rejected == 1 and stats.skipped_duplicates == 1
    assert dings[0].name == "Acme Plumbing"

    # Second run: everything is a known URL -> no new scoring, no new dings.
    stats = pipeline.run_once()
    assert stats.scored == 0 and stats.skipped_duplicates == 3 and len(dings) == 1

    db = LeadDatabase(settings.database_path)
    assert sorted(r["status"] for r in db.list_leads(min_score=8)) == ["notified", "rejected"]
