"""AI filtering & scoring: ask an LLM to grade each prospect against the ICP."""

from __future__ import annotations

import json
import logging
import re

import requests
from pydantic import BaseModel, Field, ValidationError

from config import Criteria, Settings
from searcher import Prospect

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meticulous B2B lead qualification analyst.
You receive one prospect found via web search plus a seller's Ideal Customer
Profile (ICP) and a weighted scoring rubric. Judge ONLY from the evidence given;
do not invent facts. If evidence is thin, score conservatively.

Respond with a single JSON object and nothing else, using this schema:
{
  "is_relevant": bool,              // false if this is not a real prospect (directory, news article, competitor, job board aggregator, etc.)
  "lead_name": str,                 // business / person / organisation name
  "rubric_scores": {"<rubric name>": int 1-10, ...},
  "score": int 1-10,                // overall, consistent with the weighted rubric
  "pain_points": [str],             // concrete problems the prospect appears to have
  "budget_signals": [str],          // evidence they can pay (size, funding, hiring, pricing pages...)
  "contact_hints": [str],           // emails, phone numbers, contact page URLs, usernames seen in the text
  "reasoning": str,                 // 1-3 sentences
  "suggested_opener": str           // one-sentence personalised outreach opener
}"""


class LeadScore(BaseModel):
    is_relevant: bool = True
    lead_name: str = ""
    rubric_scores: dict[str, int] = Field(default_factory=dict)
    score: int = Field(ge=1, le=10)
    pain_points: list[str] = Field(default_factory=list)
    budget_signals: list[str] = Field(default_factory=list)
    contact_hints: list[str] = Field(default_factory=list)
    reasoning: str = ""
    suggested_opener: str = ""


class Analyzer:
    def __init__(self, settings: Settings, criteria: Criteria):
        self.settings = settings
        self.criteria = criteria
        self.session = requests.Session()

    def analyze(self, prospect: Prospect) -> LeadScore | None:
        prompt = self.build_prompt(prospect)
        try:
            raw = self._complete(prompt)
            result = parse_score(raw)
        except (requests.RequestException, ValueError, ValidationError) as exc:
            log.error("LLM scoring failed for %s: %s", prospect.url, exc)
            return None

        if not result.is_relevant:
            result.score = 1
        elif result.rubric_scores:
            # Recompute the overall score from the rubric so weights are honoured
            # even if the model's own arithmetic drifts.
            result.score = self.weighted_score(result.rubric_scores) or result.score
        return result

    # ----------------------------------------------------------------- prompts
    def build_prompt(self, prospect: Prospect) -> str:
        rubric = "\n".join(
            f"- {r.name} (weight {r.weight}): {r.description}" for r in self.criteria.rubric
        )
        page = prospect.page_text or "(page not fetched)"
        return f"""## What the seller offers
{self.criteria.offer}

## Ideal Customer Profile
{self.criteria.icp}

## Scoring rubric (score each 1-10; use these exact names as keys)
{rubric}

## Prospect
Search query: {prospect.query}
Title: {prospect.title}
URL: {prospect.url}
Snippet: {prospect.snippet}

Page content (truncated):
\"\"\"
{page}
\"\"\"

Return the JSON object now."""

    def weighted_score(self, rubric_scores: dict[str, int]) -> int | None:
        total = weight_sum = 0.0
        normalized = {k.strip().lower(): v for k, v in rubric_scores.items()}
        for item in self.criteria.rubric:
            value = normalized.get(item.name.lower())
            if value is None:
                continue
            total += max(1, min(10, int(value))) * item.weight
            weight_sum += item.weight
        if not weight_sum:
            return None
        return max(1, min(10, round(total / weight_sum)))

    # --------------------------------------------------------------- providers
    def _complete(self, prompt: str) -> str:
        provider = self.settings.llm_provider
        if provider == "anthropic":
            return self._anthropic(prompt)
        if provider == "openai":
            return self._openai(prompt)
        return self._ollama(prompt)

    def _anthropic(self, prompt: str) -> str:
        if not self.settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        resp = self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.settings.resolved_model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json()["content"] if b["type"] == "text")

    def _openai(self, prompt: str) -> str:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        resp = self.session.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.resolved_model,
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _ollama(self, prompt: str) -> str:
        resp = self.session.post(
            f"{self.settings.ollama_host.rstrip('/')}/api/chat",
            json={
                "model": self.settings.resolved_model,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def parse_score(raw: str) -> LeadScore:
    """Extract the first JSON object from an LLM reply and validate it."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object in LLM reply: {raw[:200]!r}")
        text = text[start : end + 1]
    data = json.loads(text)
    # Clamp scores defensively: small models sometimes return 0 or 11.
    if "score" in data:
        data["score"] = max(1, min(10, int(data["score"])))
    if isinstance(data.get("rubric_scores"), dict):
        data["rubric_scores"] = {
            k: max(1, min(10, int(v))) for k, v in data["rubric_scores"].items()
        }
    return LeadScore.model_validate(data)
