"""Full pipeline runs with search and LLM stubbed."""

import json

import pytest

from analyzer import LLMConfigError
from database import LeadDatabase
from main import Pipeline, main
from searcher import Prospect


def _wire(monkeypatch, pipeline, prospects, replies, dings):
    monkeypatch.setattr(pipeline.searcher, "search",
                        lambda q, attempts=3: [Prospect(**vars(p)) for p in prospects])

    def complete(prompt):
        for url, reply in replies.items():
            if f"URL: {url}\n" in prompt:
                return reply if isinstance(reply, str) else json.dumps(reply)
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(pipeline.analyzer, "_complete", complete)
    monkeypatch.setattr(pipeline.notifier, "ding", lambda alert: dings.append(alert) or True)
    monkeypatch.setattr(type(pipeline.notifier), "channels", property(lambda self: ["fake"]))


PROSPECTS = [
    Prospect("Acme Plumbing", "https://acmeplumbing.com", "family owned", "q"),
    Prospect("Acme Plumbing - About", "https://acmeplumbing.com/about", "same site", "q"),
    Prospect("Acme Plumbing LLC", "https://acme-plumbing-austin.com", "same biz, other domain", "q"),
    Prospect("Big Chain", "https://bigchain.com", "national", "q"),
    Prospect("Flaky", "https://flaky.com", "llm returns junk", "q"),
]
REPLIES = {
    "https://acmeplumbing.com": {"lead_name": "Acme Plumbing", "score": 9,
                                 "rubric_scores": {"relevance": 9, "pain_points": 9,
                                                   "budget_signals": 8, "reachability": 9}},
    "https://acme-plumbing-austin.com": {"lead_name": "Acme Plumbing LLC", "score": 9},
    "https://bigchain.com": {"lead_name": "Big Chain", "score": 2, "is_relevant": False},
    "https://flaky.com": "sorry, I can't",
}


def test_pipeline_end_to_end(monkeypatch, settings, criteria):
    criteria.queries = ["q1", "q2"]  # same results twice -> in-run dedupe
    dings = []
    pipeline = Pipeline(settings, criteria)
    _wire(monkeypatch, pipeline, PROSPECTS, REPLIES, dings)

    stats = pipeline.run_once()
    assert (stats.dinged, stats.rejected, stats.duplicates, stats.llm_errors) == (1, 1, 1, 1)
    assert stats.scored == 3 and stats.skipped_known == 6  # /about + all 5 again for q2
    assert dings[0].name == "Acme Plumbing" and dings[0].lead_id

    # Second run: only the flaky one (never stored) is retried. No new dings.
    stats = pipeline.run_once()
    assert stats.scored == 0 and stats.llm_errors == 1 and len(dings) == 1

    with LeadDatabase(settings.database_path) as db:
        assert db.counts() == {"notified": 1, "duplicate": 1, "rejected": 1}


def test_failed_ding_is_retried_next_run(monkeypatch, settings, criteria):
    criteria.queries = ["q"]
    pipeline = Pipeline(settings, criteria)
    dings = []
    _wire(monkeypatch, pipeline, PROSPECTS[:1], REPLIES, dings)
    monkeypatch.setattr(pipeline.notifier, "ding", lambda alert: False)
    assert pipeline.run_once().dinged == 0

    monkeypatch.setattr(pipeline.notifier, "ding", lambda alert: dings.append(alert) or True)
    assert pipeline.run_once().dinged == 1
    assert pipeline.run_once().dinged == 0  # not dinged twice


def test_dry_run_stores_nothing(monkeypatch, settings, criteria):
    criteria.queries = ["q"]
    pipeline = Pipeline(settings, criteria, dry_run=True)
    dings = []
    _wire(monkeypatch, pipeline, PROSPECTS, REPLIES, dings)
    stats = pipeline.run_once()
    assert stats.qualified == 2 and not dings
    with LeadDatabase(settings.database_path) as db:
        assert db.counts() == {}


def test_llm_call_cap(monkeypatch, settings, criteria):
    criteria.queries, criteria.max_llm_calls_per_run = ["q"], 1
    pipeline = Pipeline(settings, criteria)
    _wire(monkeypatch, pipeline, PROSPECTS, REPLIES, [])
    assert pipeline.run_once().scored == 1


def test_config_error_aborts_run(monkeypatch, settings, criteria):
    criteria.queries = ["q"]
    pipeline = Pipeline(settings, criteria)
    _wire(monkeypatch, pipeline, PROSPECTS, REPLIES, [])

    def bad(prompt):
        raise LLMConfigError("bad key")

    monkeypatch.setattr(pipeline.analyzer, "_complete", bad)
    with pytest.raises(LLMConfigError):
        pipeline.run_once()


def test_cli_without_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert main(["run"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_list_export_contacted(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "l.db"))
    from analyzer import parse_score
    with LeadDatabase(tmp_path / "l.db") as db:
        db.save(Prospect("A", "https://a.com", "", "q"), parse_score('{"score": 9}'), "notified")
    assert main(["contacted", "1"]) == 0
    assert main(["list"]) == 0
    assert "contacted" in capsys.readouterr().out
    assert main(["export", str(tmp_path / "o.csv")]) == 0
    assert main(["contacted", "99"]) == 1
