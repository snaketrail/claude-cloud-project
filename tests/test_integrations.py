"""HTTP-level tests against a local mock server using each API's real response shape."""

import json

import pytest

from analyzer import Analyzer, LLMConfigError
from notifier import LeadAlert, Notifier
from searcher import Prospect

GOOD = {"is_relevant": True, "lead_name": "Joe's Plumbing", "score": 9,
        "rubric_scores": {"relevance": 9, "pain_points": 9, "budget_signals": 8, "reachability": 9},
        "pain_points": ["no mobile site"], "reasoning": "fits"}
PROSPECT = Prospect("Joe's Plumbing", "https://joes.example", "family owned", "plumber austin")


def _anthropic(settings, mock_server):
    settings.llm_provider, settings.anthropic_api_url = "anthropic", mock_server.url
    return "/v1/messages", {"content": [{"type": "text", "text": json.dumps(GOOD)}]}


def _openai(settings, mock_server):
    settings.llm_provider, settings.openai_base_url = "openai", mock_server.url + "/v1"
    return "/v1/chat/completions", {"choices": [{"message": {"content": json.dumps(GOOD)}}]}


def _ollama(settings, mock_server):
    settings.llm_provider, settings.ollama_host = "ollama", mock_server.url
    return "/api/chat", {"message": {"role": "assistant", "content": json.dumps(GOOD)}}


@pytest.mark.parametrize("setup", [_anthropic, _openai, _ollama])
def test_llm_providers_retry_and_parse(setup, settings, criteria, mock_server):
    path, body = setup(settings, mock_server)
    mock_server.add(path, 429, {"error": "rate"}, {"retry-after": "1"})
    mock_server.add(path, 529 if settings.llm_provider == "anthropic" else 503, {"error": "overloaded"})
    mock_server.add(path, 200, body)

    result = Analyzer(settings, criteria).analyze(PROSPECT)
    assert result is not None and result.score == 9 and result.lead_name == "Joe's Plumbing"
    assert len(mock_server.requests) == 3

    req = mock_server.requests[-1]
    assert req["json"]["model"] == settings.resolved_model
    if settings.llm_provider == "anthropic":
        assert req["headers"]["x-api-key"] == "test-key"
        assert req["headers"]["anthropic-version"] == "2023-06-01"
        assert req["json"]["system"] and req["json"]["messages"][0]["role"] == "user"
        assert req["json"]["max_tokens"] > 0
    elif settings.llm_provider == "openai":
        assert req["headers"]["authorization"] == "Bearer test-key"
        assert req["json"]["response_format"] == {"type": "json_object"}
    else:
        assert req["json"]["format"] == "json" and req["json"]["stream"] is False


@pytest.mark.parametrize("setup", [_anthropic, _openai])
def test_bad_key_raises_config_error(setup, settings, criteria, mock_server):
    path, _ = setup(settings, mock_server)
    mock_server.add(path, 401, {"error": {"message": "invalid x-api-key"}})
    with pytest.raises(LLMConfigError, match="HTTP 401"):
        Analyzer(settings, criteria).analyze(PROSPECT)
    assert len(mock_server.requests) == 1  # no pointless retries


def test_persistent_5xx_returns_none(settings, criteria, mock_server):
    path, _ = _anthropic(settings, mock_server)
    for _ in range(3):
        mock_server.add(path, 500, {"error": "boom"})
    assert Analyzer(settings, criteria).analyze(PROSPECT) is None
    assert len(mock_server.requests) == 3


def test_ollama_not_running(settings, criteria):
    settings.llm_provider, settings.ollama_host = "ollama", "http://127.0.0.1:9"
    with pytest.raises(LLMConfigError, match="ollama serve"):
        Analyzer(settings, criteria).analyze(PROSPECT)


def test_all_notifiers(settings, mock_server):
    settings.telegram_bot_token, settings.telegram_chat_id = "123:ABC", "42"
    settings.telegram_api_url = mock_server.url
    settings.discord_webhook_url = mock_server.url + "/discord"
    settings.slack_webhook_url = mock_server.url + "/slack"
    mock_server.add("/bot123:ABC/sendMessage", 429,
                    {"ok": False, "parameters": {"retry_after": 1}})
    mock_server.add("/bot123:ABC/sendMessage", 200, {"ok": True})
    mock_server.add("/discord", 204, "")
    mock_server.add("/slack", 200, "ok", {"Content-Type": "text/plain"})

    alert = LeadAlert("Joe <!channel> @everyone", "https://joes.example", 9, "fits", "q",
                      lead_id=7, pain_points=["x" * 5000])
    assert Notifier(settings).ding(alert)

    by_path = {r["path"]: r["json"] for r in mock_server.requests}
    tg = by_path["/bot123:ABC/sendMessage"]
    assert tg["chat_id"] == "42" and len(tg["text"]) <= 4096 and "Lead #7" in tg["text"]
    assert by_path["/discord"]["allowed_mentions"] == {"parse": []}
    assert len(by_path["/discord"]["content"]) <= 2000
    assert "<!channel>" not in by_path["/slack"]["text"] and "&lt;!channel&gt;" in by_path["/slack"]["text"]
    assert len(mock_server.requests) == 4  # telegram retried once after 429


def test_notifier_partial_failure_and_redaction(settings, mock_server, caplog):
    settings.discord_webhook_url = mock_server.url + "/discord/secret-token"
    settings.slack_webhook_url = mock_server.url + "/slack"
    mock_server.add("/discord/secret-token", 404, {"message": "Unknown Webhook"})
    mock_server.add("/slack", 200, "ok")
    assert Notifier(settings).ding(LeadAlert("A", "https://a", 9, "r", "q"))
    assert "secret-token" not in caplog.text


def test_notifier_all_fail(settings, mock_server):
    settings.slack_webhook_url = mock_server.url + "/slack"
    mock_server.add("/slack", 500, "err")
    assert not Notifier(settings).ding(LeadAlert("A", "https://a", 9, "r", "q"))
