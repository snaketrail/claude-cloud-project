"""Target discovery: run "cold search" queries and fetch the pages they point to."""

from __future__ import annotations

import logging
import re
import time
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
MAX_PAGE_BYTES = 2_000_000
SKIP_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip",
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3",
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
        return domain_of(self.url)


def domain_of(url: str) -> str:
    host = urlparse(url.strip()).netloc.lower().split("@")[-1].split(":")[0]
    return host.removeprefix("www.")


class Searcher:
    def __init__(self, settings: Settings, criteria: Criteria):
        self.settings = settings
        self.criteria = criteria
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        )

    # ------------------------------------------------------------------ search
    def search(self, query: str, attempts: int = 3) -> list[Prospect]:
        limit = self.criteria.results_per_query
        raw: list[Prospect] = []
        for attempt in range(1, attempts + 1):
            try:
                raw = self._dispatch(query, limit)
                break
            except Exception as exc:  # rate limits, timeouts, provider errors
                # ddgs reports throttling as "No results found", so retry that too.
                if attempt == attempts:
                    hint = ""
                    if self.settings.search_provider == "duckduckgo":
                        hint = (" (DuckDuckGo may be rate-limiting you: raise SEARCH_DELAY, "
                                "or use SEARCH_PROVIDER=brave/serpapi)")
                    log.warning("No results for %r after %d attempts: %s%s", query, attempts, exc, hint)
                    return []
                wait = 5 * 2 ** (attempt - 1)
                log.info("Search for %r failed (%s); retrying in %ds", query, exc, wait)
                time.sleep(wait)

        seen: set[str] = set()
        prospects = []
        for p in raw:
            if not p.url.startswith(("http://", "https://")) or p.url in seen:
                continue
            seen.add(p.url)
            if not self._is_blocked(p):
                prospects.append(p)
        log.info("Query %r -> %d results (%d after filters)", query, len(raw), len(prospects))
        return prospects

    def _dispatch(self, query: str, limit: int) -> list[Prospect]:
        provider = self.settings.search_provider
        if provider == "serpapi":
            return self._search_serpapi(query, limit)
        if provider == "brave":
            return self._search_brave(query, limit)
        return self._search_duckduckgo(query, limit)

    def _search_duckduckgo(self, query: str, limit: int) -> list[Prospect]:
        from ddgs import DDGS

        hits = DDGS(timeout=self.settings.request_timeout).text(
            query, region=self.criteria.region, max_results=limit
        ) or []
        return [
            Prospect(
                title=(h.get("title") or "").strip(),
                url=(h.get("href") or "").strip(),
                snippet=(h.get("body") or "").strip(),
                query=query,
            )
            for h in hits[:limit]
        ]

    def _search_serpapi(self, query: str, limit: int) -> list[Prospect]:
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
        data = resp.json()
        if data.get("error"):
            if "hasn't returned any results" in data["error"]:
                return []
            raise RuntimeError(data["error"])
        return [
            Prospect(
                title=h.get("title", ""),
                url=h.get("link", ""),
                snippet=h.get("snippet", ""),
                query=query,
            )
            for h in data.get("organic_results", [])[:limit]
        ]

    def _search_brave(self, query: str, limit: int) -> list[Prospect]:
        results: list[Prospect] = []
        offset = 0
        while len(results) < limit and offset <= 9:  # Brave: max 20 per page, offset <= 9
            resp = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(20, limit), "offset": offset},
                headers={"X-Subscription-Token": self.settings.brave_api_key,
                         "Accept": "application/json"},
                timeout=self.settings.request_timeout,
            )
            resp.raise_for_status()
            page = resp.json().get("web", {}).get("results", [])
            results += [
                Prospect(
                    title=h.get("title", ""),
                    url=h.get("url", ""),
                    snippet=re.sub(r"<[^>]+>", "", h.get("description", "")),
                    query=query,
                )
                for h in page
            ]
            if len(page) < min(20, limit):
                break
            offset += 1
        return results[:limit]

    # ----------------------------------------------------------------- filters
    def _is_blocked(self, prospect: Prospect) -> bool:
        domain = prospect.domain
        if any(domain == d or domain.endswith("." + d) for d in self.criteria.blocked_domains):
            log.debug("Blocked domain: %s", prospect.url)
            return True
        haystack = f"{prospect.title} {prospect.snippet}".lower()
        if any(kw in haystack for kw in self.criteria.exclude_keywords):
            log.debug("Excluded keyword: %s", prospect.url)
            return True
        return False

    # ------------------------------------------------------------------ enrich
    def enrich(self, prospect: Prospect) -> Prospect:
        """Fetch the target page and attach its visible text (best effort)."""
        if not self.settings.fetch_pages:
            return prospect
        if urlparse(prospect.url).path.lower().endswith(SKIP_EXTENSIONS):
            return prospect
        try:
            with self.session.get(
                prospect.url, timeout=self.settings.request_timeout, stream=True
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "text/html").lower()
                if "html" not in content_type and "text/plain" not in content_type:
                    return prospect
                body = b""
                for chunk in resp.iter_content(64 * 1024):
                    body += chunk
                    if len(body) >= MAX_PAGE_BYTES:
                        break
                encoding = resp.encoding if "charset" in content_type else None
            html = body.decode(encoding or "utf-8", errors="replace")
            prospect.page_text = extract_text(html, self.settings.page_char_limit)
        except (requests.RequestException, LookupError) as exc:
            log.debug("Could not fetch %s: %s", prospect.url, exc)
        return prospect


def extract_text(html: str, char_limit: int) -> str:
    """Strip scripts/styles and collapse whitespace; prepend useful page signals."""
    soup = BeautifulSoup(html, "html.parser")

    signals = []
    if soup.title and soup.title.string:
        signals.append(f"Page title: {soup.title.string.strip()}")
    description = soup.find("meta", attrs={"name": "description"})
    if description and description.get("content"):
        signals.append(f"Meta description: {description['content'].strip()}")
    generator = soup.find("meta", attrs={"name": "generator"})
    if generator and generator.get("content"):
        # Useful signal for "outdated website" style campaigns.
        signals.append(f"Site generator: {generator['content'].strip()}")
    if not soup.find("meta", attrs={"name": "viewport"}):
        signals.append("Note: page has no mobile viewport meta tag")
    years = [int(y) for y in re.findall(r"(?:©|&copy;|copyright)\s*(?:\d{4}\s*[-–]\s*)?((?:19|20)\d{2})",
                                       html, re.I)]
    if years:
        signals.append(f"Copyright year on page: {max(years)}")
    emails = sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", html)))[:5]
    emails = [e for e in emails if not e.lower().endswith((".png", ".jpg", ".webp", ".svg"))]
    if emails:
        signals.append(f"Emails on page: {', '.join(emails)}")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return ("\n".join(signals) + "\n\n" + text).strip()[:char_limit]
