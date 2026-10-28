"""Lead Generation & Pipeline Agent -- orchestration entry point.

Pipeline per query:
    search -> filter blocked/seen -> fetch page -> LLM score -> store -> ding

Usage:
    python main.py check               # validate config and test the LLM + search
    python main.py run                 # one pass over every query in criteria.yaml
    python main.py run --loop 60       # repeat every 60 minutes
    python main.py run --dry-run       # search + score, but don't store or ding
    python main.py list --min-score 7  # show stored leads
    python main.py export leads.csv    # dump to CSV
    python main.py contacted 12        # mark lead #12 as contacted
    python main.py test-ding           # send a sample alert to your channels
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

from analyzer import Analyzer, LLMConfigError
from config import ConfigError, Criteria, Settings
from database import STATUSES, LeadDatabase, is_platform, normalize_url
from notifier import LeadAlert, Notifier
from searcher import Searcher

log = logging.getLogger("leadgen")


@dataclass
class RunStats:
    found: int = 0
    skipped_known: int = 0
    scored: int = 0
    llm_errors: int = 0
    duplicates: int = 0
    rejected: int = 0
    qualified: int = 0
    dinged: int = 0

    def __str__(self) -> str:
        return (
            f"found={self.found} already_seen={self.skipped_known} scored={self.scored} "
            f"llm_errors={self.llm_errors} duplicates={self.duplicates} rejected={self.rejected} "
            f"qualified={self.qualified} dinged={self.dinged}"
        )


class Pipeline:
    def __init__(self, settings: Settings, criteria: Criteria, dry_run: bool = False):
        self.settings = settings
        self.criteria = criteria
        self.dry_run = dry_run
        self.searcher = Searcher(settings, criteria)
        self.analyzer = Analyzer(settings, criteria)
        self.notifier = Notifier(settings)
        self.db = LeadDatabase(settings.database_path)

    def close(self) -> None:
        self.db.close()

    def _seen_key(self, prospect) -> str:
        if self.criteria.dedupe_scope == "domain" and not is_platform(prospect.domain):
            return prospect.domain
        return normalize_url(prospect.url)

    def run_once(self) -> RunStats:
        stats = RunStats()
        self._retry_pending_dings(stats)
        seen_this_run: set[str] = set()
        llm_calls = 0

        for i, query in enumerate(self.criteria.queries):
            if i and self.settings.search_delay:
                time.sleep(self.settings.search_delay)
            for prospect in self.searcher.search(query):
                stats.found += 1
                key = self._seen_key(prospect)
                if key in seen_this_run or self.db.is_known(prospect.url, self.criteria.dedupe_scope):
                    stats.skipped_known += 1
                    continue
                seen_this_run.add(key)

                if llm_calls >= self.criteria.max_llm_calls_per_run:
                    log.warning(
                        "Reached max_llm_calls_per_run=%d; remaining prospects wait for next run.",
                        self.criteria.max_llm_calls_per_run,
                    )
                    log.info("Run complete: %s", stats)
                    return stats

                self.searcher.enrich(prospect)
                llm_calls += 1
                result = self.analyzer.analyze(prospect)  # LLMConfigError propagates: abort run
                if result is None:
                    stats.llm_errors += 1
                    continue  # not stored, so it is retried on the next run
                stats.scored += 1
                self._handle(prospect, result, stats)

        log.info("Run complete: %s", stats)
        return stats

    def _handle(self, prospect, result, stats: RunStats) -> None:
        name = result.lead_name or prospect.title or prospect.domain
        log.info("[%2d/10] %-50s %s", result.score, name[:50], prospect.url)

        qualified = result.is_relevant and result.score >= self.criteria.min_score_to_save
        if qualified and self.db.is_known_name(name):
            # Same business found under a different website (e.g. two domains).
            stats.duplicates += 1
            log.info("        already have a lead named %r; stored as duplicate", name)
            if not self.dry_run:
                self.db.save(prospect, result, status="duplicate")
            return

        if not qualified:
            stats.rejected += 1
            if not self.dry_run:
                self.db.save(prospect, result, status="rejected")
            return

        stats.qualified += 1
        if self.dry_run:
            if result.score >= self.criteria.notify_threshold:
                log.info("        (dry run) would ding: %s", result.reasoning)
            return

        lead_id = self.db.save(prospect, result, status="qualified")
        if lead_id is not None and result.score >= self.criteria.notify_threshold:
            if self.notifier.ding(LeadAlert.from_row(self.db.get(lead_id))):
                self.db.mark_notified(lead_id)
                stats.dinged += 1

    def _retry_pending_dings(self, stats: RunStats) -> None:
        """Re-send alerts that failed last time (e.g. webhook was down)."""
        if self.dry_run or not self.notifier.channels:
            return
        for row in self.db.pending_notifications(self.criteria.notify_threshold):
            if self.notifier.ding(LeadAlert.from_row(row)):
                self.db.mark_notified(row["id"])
                stats.dinged += 1


# ---------------------------------------------------------------------- CLI
def _require_runnable(settings: Settings) -> None:
    problems = settings.problems()
    if problems:
        raise ConfigError("Fix these settings in .env first:\n  - " + "\n  - ".join(problems))


def cmd_run(args, settings: Settings) -> int:
    _require_runnable(settings)
    criteria = Criteria.load(settings.criteria_path)
    pipeline = Pipeline(settings, criteria, dry_run=args.dry_run)
    log.info(
        "Campaign %r | %d queries | search=%s | llm=%s/%s | channels=%s%s",
        criteria.campaign_name,
        len(criteria.queries),
        settings.search_provider,
        settings.llm_provider,
        settings.resolved_model,
        ",".join(pipeline.notifier.channels) or "console only",
        " | DRY RUN" if args.dry_run else "",
    )
    try:
        # Fail fast on a bad key/model/Ollama before spending search quota.
        pipeline.analyzer.ping()
    except LLMConfigError as exc:
        log.error("%s", exc)
        pipeline.close()
        return 2
    except Exception as exc:
        log.warning("LLM preflight check failed (%s); continuing anyway", exc)

    try:
        while True:
            try:
                pipeline.run_once()
            except LLMConfigError as exc:
                log.error("%s", exc)
                return 2
            except Exception:  # keep --loop alive through unexpected errors
                if not args.loop:
                    raise
                log.exception("Run failed; will try again next cycle")
            if not args.loop:
                return 0
            log.info("Next run in %d minutes (Ctrl+C to stop)", args.loop)
            time.sleep(args.loop * 60)
    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0
    finally:
        pipeline.close()


def cmd_check(args, settings: Settings) -> int:
    ok = True

    def report(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))

    print("Configuration")
    for problem in settings.problems():
        report(".env", False, problem)
    try:
        criteria = Criteria.load(settings.criteria_path)
        report("criteria", True, f"{criteria.campaign_name!r}, {len(criteria.queries)} queries, "
               f"rubric={[r.name for r in criteria.rubric]}")
    except ConfigError as exc:
        report("criteria", False, str(exc))
        return 1

    print("LLM")
    try:
        reply = Analyzer(settings, criteria).ping()
        report(f"{settings.llm_provider}/{settings.resolved_model}", True, f"replied {reply.strip()[:60]!r}")
    except Exception as exc:
        report(f"{settings.llm_provider}/{settings.resolved_model}", False, str(exc)[:300])

    print("Search")
    results = Searcher(settings, criteria).search(criteria.queries[0], attempts=1)
    report(settings.search_provider, bool(results), f"{len(results)} results for {criteria.queries[0]!r}")

    print("Notifications")
    channels = Notifier(settings).channels
    report("channels", True, ", ".join(channels) or "none (alerts print to console); run test-ding to verify")

    print("\nAll good." if ok else "\nSome checks failed.")
    return 0 if ok else 1


def cmd_list(args, settings: Settings) -> int:
    with LeadDatabase(settings.database_path) as db:
        rows = db.list_leads(min_score=args.min_score, status=args.status)
        for r in rows[: args.limit]:
            name = (r["name"] or "")[:40]
            print(f"#{r['id']:<5} {r['score']:>2}/10  {r['status']:<10} {name:<40} {r['url']}")
        shown = min(len(rows), args.limit)
        counts = ", ".join(f"{k}={v}" for k, v in sorted(db.counts().items())) or "empty"
        print(f"\nShowing {shown} of {len(rows)} matching lead(s). Totals: {counts}")
    return 0


def cmd_export(args, settings: Settings) -> int:
    with LeadDatabase(settings.database_path) as db:
        count = db.export_csv(args.path, min_score=args.min_score, status=args.status)
    print(f"Exported {count} lead(s) to {args.path}")
    return 0


def cmd_contacted(args, settings: Settings) -> int:
    with LeadDatabase(settings.database_path) as db:
        ok = all([db.mark_contacted(i) for i in args.lead_ids])
    print("Marked as contacted." if ok else "Some lead ids were not found.")
    return 0 if ok else 1


def cmd_test_ding(args, settings: Settings) -> int:
    notifier = Notifier(settings)
    alert = LeadAlert(
        name="Example Plumbing Co.",
        url="https://example.com",
        score=9,
        reasoning="Test alert from the lead pipeline.",
        query="test",
        pain_points=["No mobile-friendly website"],
        budget_signals=["5+ trucks listed on site"],
    )
    delivered = notifier.ding(alert)
    if delivered:
        print(f"Test alert sent via: {', '.join(notifier.channels)}")
    elif notifier.channels:
        print("Every configured channel failed; see the errors above.")
    else:
        print("No channel configured in .env; the alert above was printed to the console only.")
    return 0 if delivered else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lead Generation & Pipeline Agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.set_defaults(func=cmd_run, loop=None, dry_run=False)  # bare `python main.py` == run
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the search -> score -> ding pipeline")
    run.add_argument("--loop", type=int, metavar="MINUTES", help="repeat every N minutes")
    run.add_argument("--dry-run", action="store_true", help="don't store leads or send dings")
    run.set_defaults(func=cmd_run)

    ck = sub.add_parser("check", help="validate config and test LLM + search connectivity")
    ck.set_defaults(func=cmd_check)

    ls = sub.add_parser("list", help="list stored leads")
    ls.add_argument("--min-score", type=int, default=1)
    ls.add_argument("--status", choices=STATUSES)
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(func=cmd_list)

    ex = sub.add_parser("export", help="export leads to CSV")
    ex.add_argument("path", nargs="?", default="leads.csv")
    ex.add_argument("--min-score", type=int, default=1)
    ex.add_argument("--status", choices=STATUSES)
    ex.set_defaults(func=cmd_export)

    ct = sub.add_parser("contacted", help="mark one or more leads as contacted")
    ct.add_argument("lead_ids", type=int, nargs="+")
    ct.set_defaults(func=cmd_contacted)

    td = sub.add_parser("test-ding", help="send a sample alert to configured channels")
    td.set_defaults(func=cmd_test_ding)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "primp", "ddgs", "httpx", "h2", "hpack", "rustls"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return args.func(args, Settings.from_env())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
