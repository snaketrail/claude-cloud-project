# Lead Generation & Pipeline Agent

A small, modular Python agent that **searches the web for prospects**, has an **LLM score each one against your Ideal Customer Profile**, stores the results in **SQLite** (so nothing is scored or alerted twice), and **dings you** on Telegram, Discord or Slack when a high-potential lead turns up.

```
criteria.yaml ─┐
               ▼
  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐   ┌──────────┐
  │ searcher │──▶│ dedupe check │──▶│  analyzer   │──▶│ database  │──▶│ notifier │
  │ DDG/Serp │   │  (URL known?)│   │ LLM + ICP   │   │  SQLite   │   │  "ding"  │
  │ + fetch  │   └──────────────┘   │ rubric 1-10 │   │ dedupe by │   │ score ≥ 8│
  └──────────┘                      └─────────────┘   │ URL & name│   └──────────┘
                                                      └───────────┘
```

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | CLI and pipeline orchestration (`run`, `list`, `export`, `contacted`, `test-ding`) |
| `searcher.py` | Runs queries (DuckDuckGo, or SerpAPI for Google results), filters blocked domains and keywords, fetches page text |
| `analyzer.py` | Builds the ICP prompt, calls Anthropic, OpenAI (or any OpenAI-compatible API), or Ollama, then parses and validates the JSON score |
| `database.py` | SQLite storage, URL and name normalization for duplicate blocking, CSV export |
| `notifier.py` | Formats and sends alerts to Telegram, Discord and Slack |
| `config.py` | Loads `.env` (secrets) and `criteria.yaml` (targeting), both validated with pydantic |
| `criteria.example.yaml` | Template for queries, ICP, rubric and thresholds |
| `.env.example` | Template for API keys and settings |
| `tests/` | Offline tests (no network or keys needed) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

```bash
cp criteria.example.yaml criteria.yaml
```

Edit `.env` (set at least an LLM key) and `criteria.yaml`. Then:

```bash
python main.py test-ding        # check notifications reach you
```

```bash
python main.py run --dry-run    # search and score, but store nothing and send nothing
```

```bash
python main.py run              # real run
```

```bash
python main.py run --loop 120   # keep running, one pass every 2 hours
```

## Configuring the criteria (`criteria.yaml`)

This file decides **who** you're hunting. You can tune it without touching any code.

- **`offer`**: what you sell. The LLM uses it to judge fit and to write a personalised `suggested_opener`.
- **`queries`**: the "cold search" queries. Each run executes every one. Good queries pair a niche and a location with a pain signal, for example:
  - Local businesses with weak websites: `"roofing contractor Denver \"call for a free estimate\""`
  - Companies hiring for a role you can replace or augment: `"\"we're hiring\" \"marketing coordinator\" startup Berlin"`
  - Forum users asking for a solution: `"site:reddit.com \"any recommendations\" CRM for real estate agents"`
- **`icp`**: a plain-English description of your ideal buyer. Say who is in **and** who is out (franchises, directories, competitors). Explicit exclusions are the biggest lever on quality.
- **`rubric`**: the scoring dimensions. The LLM scores each one from 1 to 10, and the agent recomputes the overall score as the weighted average, so your weights hold even if the model's arithmetic slips. Add, remove or rename dimensions freely.
- **`min_score_to_save`**: below this, a prospect is stored as `rejected`. It's still kept, so it is never re-scored and never costs another LLM call.
- **`notify_threshold`**: a score at or above this triggers a ding (default 8).
- **`blocked_domains` / `exclude_keywords`**: cheap pre-filters that run *before* the LLM to cut cost. Use them to block directories (Yelp, BBB…) and listicles ("top 10…").

To run several campaigns, keep several criteria files and switch between them:

```bash
CRITERIA_PATH=campaigns/saas-founders.yaml DATABASE_PATH=saas.db python main.py run
```

## Configuring providers (`.env`)

### Search
- `SEARCH_PROVIDER=duckduckgo`: free and needs no key. Good for getting started, but heavy use gets rate-limited.
- `SEARCH_PROVIDER=serpapi`: Google results through [SerpAPI](https://serpapi.com). Set `SERPAPI_API_KEY`.
- `FETCH_PAGES=true` downloads each result and passes up to `PAGE_CHAR_LIMIT` characters of visible text to the LLM. The agent also adds hints such as the CMS `generator` tag and whether the page has a mobile viewport. These are strong signals for "outdated website" campaigns.

### LLM
| `LLM_PROVIDER` | Needs | Default model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `openai` | `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` for Groq, Together, LM Studio, vLLM…) | `gpt-4o-mini` |
| `ollama` | Ollama running locally (`ollama pull llama3.1`) | `llama3.1` |

Set `LLM_MODEL` to override the default model.

### Notifications (the "ding")
Configure one or more of these. Every configured channel gets each alert. If none are set, alerts are printed to the console.

- **Telegram**: message [@BotFather](https://t.me/BotFather) with `/newbot` to get `TELEGRAM_BOT_TOKEN`. Send your bot any message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id` into `TELEGRAM_CHAT_ID`.
- **Discord**: Channel settings → Integrations → Webhooks → New Webhook → copy the URL into `DISCORD_WEBHOOK_URL`.
- **Slack**: create an app → Incoming Webhooks → add it to a channel → put the URL in `SLACK_WEBHOOK_URL`.

An alert contains the lead name, score, URL, the model's reasoning, pain points, budget signals, any contact hints, and a suggested opening line.

## Tracking leads

```bash
python main.py list --min-score 7             # view stored leads
```

```bash
python main.py list --status notified         # leads you've been dinged about
```

```bash
python main.py contacted 12                   # mark lead #12 as contacted
```

```bash
python main.py export leads.csv --min-score 5 # spreadsheet export
```

Lead lifecycle: `rejected` · `qualified` (saved, score below the ding threshold, or the ding failed) → `notified` → `contacted`.

**Duplicate blocking** works at two levels:
1. **URL**: URLs are normalized (scheme, `www.`, trailing slash and `utm_*`/`gclid` tracking parameters are removed) and checked *before* scoring, so a known page never costs an LLM call.
2. **Name**: business names are normalized (`"Acme Plumbing, LLC"` matches `"The ACME Plumbing"`), so the same business found under a different URL isn't alerted twice.

If a ding fails (for example the webhook is down), the lead stays `qualified` and the alert is retried at the start of the next run.

## Scheduling

`--loop` is the simplest option. For something sturdier, use cron:

```cron
0 */3 * * * cd /path/to/project && .venv/bin/python main.py run >> pipeline.log 2>&1
```

## Tests

```bash
python -m pytest -q
```

The tests stub out search and the LLM. They cover URL and name normalization, JSON parsing and score clamping, weighted scoring, and a full two-run pipeline that checks a duplicate is never dinged twice.

## Extending

- **New search source** (Playwright for JavaScript-heavy sites, Google Maps, a job board API): add a `_search_<name>` method in `searcher.py` that returns `Prospect` objects, then add it to the `search_provider` choice in `config.py`.
- **New notification channel**: add `_send_<name>` in `notifier.py` and add it to `Notifier.channels`.
- **New lead fields**: add them to `LeadScore` and to the JSON schema in `SYSTEM_PROMPT` (`analyzer.py`), then add a column in `database.py`.

## Responsible use

Respect each site's terms of service and `robots.txt`, keep request volumes modest, and follow the outreach laws that apply to you (CAN-SPAM, GDPR, CASL). This tool finds and ranks *public* information. It deliberately does not send outreach automatically, so a human reviews every lead before contacting anyone.
