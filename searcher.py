"""Target discovery: run "cold search" queries and fetch the pages they point to."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import Criteria, Settings

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass
class Prospect:
    """A raw search hit, optionally enriched with the page's visible text."""

    title: str
    url: str
    snippet: str
    query: str
    page_text: str = ""

    @property
    def domain(self) -> str:
        host = urlparse(self.url).netloc.lower()
        return host.removeprefix("www.")


class Searcher:
    def __init__(self, settings: Settings, criteria: Criteria):
        self.settings = settings
        self.criteria = criteria
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    # ------------------------------------------------------------------ search
    def search(self, query: str) -> list[Prospect]:
        limit = self.criteria.results_per_query
        try:
            if self.settings.search_provider == "serpapi":
                raw = self._search_serpapi(query, limit)
            else:
                raw = self._search_duckduckgo(query, limit)
        except Exception as exc:  # network errors, rate limits, etc.
            log.error("Search failed for %r: %s", query, exc)
            return []

        prospects = [p for p in raw if p.url and not self._is_blocked(p)]
        log.info("Query %r -> %d results (%d after filters)", query, len(raw), len(prospects))
        return prospects

    def _search_duckduckgo(self, query: str, limit: int) -> list[Prospect]:
        from ddgs import DDGS

        hits = DDGS().text(query, region=self.criteria.region, max_results=limit) or []
        return [
            Prospect(
                title=h.get("title", ""),
                url=h.get("href", ""),
                snippet=h.get("body", ""),
                query=query,
            )
            for h in hits
        ]

    def _search_serpapi(self, query: str, limit: int) -> list[Prospect]:
        if not self.settings.serpapi_api_key:
            raise RuntimeError("SEARCH_PROVIDER=serpapi but SERPAPI_API_KEY is empty")
        resp = self.session.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "num": limit,
                "api_key": self.settings.serpapi_api_key,
            },
            timeout=self.settings.request_timeout,
        )
        resp.raise_for_status()
        return [
            Prospect(
                title=h.get("title", ""),
                url=h.get("link", ""),
                snippet=h.get("snippet", ""),
                query=query,
            )
            for h in resp.json().get("organic_results", [])[:limit]
        ]

    # ----------------------------------------------------------------- filters
    def _is_blocked(self, prospect: Prospect) -> bool:
        domain = prospect.domain
        if any(domain == d or domain.endswith("." + d) for d in self.criteria.blocked_domains):
            return True
        haystack = f"{prospect.title} {prospect.snippet}".lower()
        return any(kw in haystack for kw in self.criteria.exclude_keywords)

    # ------------------------------------------------------------------ enrich
    def enrich(self, prospect: Prospect) -> Prospect:
        """Fetch the target page and attach its visible text (best effort)."""
        if not self.settings.fetch_pages:
            return prospect
        try:
            resp = self.session.get(prospect.url, timeout=self.settings.request_timeout)
            resp.raise_for_status()
            if "html" not in resp.headers.get("Content-Type", "html"):
                return prospect
            prospect.page_text = extract_text(resp.text, self.settings.page_char_limit)
        except requests.RequestException as exc:
            log.debug("Could not fetch %s: %s", prospect.url, exc)
        return prospect


def extract_text(html: str, char_limit: int) -> str:
    """Strip scripts/styles/navigation and collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    meta = []
    if soup.title and soup.title.string:
        meta.append(f"<title>{soup.title.string.strip()}</title>")
    generator = soup.find("meta", attrs={"name": "generator"})
    if generator and generator.get("content"):
        # Useful signal for "outdated website" style campaigns.
        meta.append(f"<generator>{generator['content']}</generator>")
    if not soup.find("meta", attrs={"name": "viewport"}):
        meta.append("<note>page has no mobile viewport meta tag</note>")

    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return ("\n".join(meta) + "\n" + text)[:char_limit]
