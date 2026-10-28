"""Lead Generation & Pipeline Agent -- orchestration entry point.

Pipeline per query:
    search -> filter blocked/seen -> fetch page -> LLM score -> store -> ding

Usage:
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

from analyzer import Analyzer
from config import Criteria, Settings
from database import LeadDatabase
from notifier import LeadAlert, Notifier
from searcher import Searcher

log = logging.getLogger("leadgen")


@dataclass
class RunStats:
    found: int = 0
    skipped_duplicates: int = 0
    scored: int = 0
    rejected: int = 0
    qualified: int = 0
    dinged: int = 0

    def __str__(self) -> str:
        return (
            f"found={self.found} duplicates={self.skipped_duplicates} scored={self.scored} "
            f"rejected={self.rejected} qualified={self.qualified} dinged={self.dinged}"
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

    def run_once(self) -> RunStats:
        stats = RunStats()
        self._retry_pending_dings(stats)

        for query in self.criteria.queries:
            for prospect in self.searcher.search(query):
                stats.found += 1
                if self.db.is_known_url(prospect.url):
                    stats.skipped_duplicates += 1
                    continue

                self.searcher.enrich(prospect)
                result = self.analyzer.analyze(prospect)
                if result is None:
                    continue  # LLM error -- will be retried on the next run
                stats.scored += 1

                name = result.lead_name or prospect.title
                log.info("[%2d/10] %s  %s", result.score, name[:60], prospect.url)

                if self.db.is_known_name(name):
                    # Same business found under a different URL (e.g. Yelp vs own site).
                    stats.skipped_duplicates += 1
                    log.info("        duplicate name, skipping: %s", name)
                    if not self.dry_run:
                        self.db.save(prospect, result, status="rejected")
                    continue

                qualified = result.is_relevant and result.score >= self.criteria.min_score_to_save
                if not qualified:
                    stats.rejected += 1
                    if not self.dry_run:
                        self.db.save(prospect, result, status="rejected")
                    continue

                stats.qualified += 1
                if self.dry_run:
                    continue
                lead_id = self.db.save(prospect, result, status="qualified")

                if result.score >= self.criteria.notify_threshold:
                    row = self.db.conn.execute(
                        "SELECT * FROM leads WHERE id = ?", (lead_id,)
                    ).fetchone()
                    if self.notifier.ding(LeadAlert.from_row(row)):
                        self.db.mark_notified(lead_id)
                        stats.dinged += 1

        log.info("Run complete: %s", stats)
        return stats

    def _retry_pending_dings(self, stats: RunStats) -> None:
        """Re-send alerts that failed last time (e.g. webhook was down)."""
        if self.dry_run or not self.notifier.channels:
            return
        for row in self.db.pending_notifications(self.criteria.notify_threshold):
            if self.notifier.ding(LeadAlert.from_row(row)):
                self.db.mark_notified(row["id"])
                stats.dinged += 1


# ---------------------------------------------------------------------- CLI
def cmd_run(args, settings: Settings) -> int:
    criteria = Criteria.load(settings.criteria_path)
    pipeline = Pipeline(settings, criteria, dry_run=args.dry_run)
    log.info(
        "Campaign %r | search=%s | llm=%s/%s | channels=%s",
        criteria.campaign_name,
        settings.search_provider,
        settings.llm_provider,
        settings.resolved_model,
        ",".join(pipeline.notifier.channels) or "console",
    )
    try:
        while True:
            pipeline.run_once()
            if not args.loop:
                return 0
            log.info("Sleeping %d minutes...", args.loop)
            time.sleep(args.loop * 60)
    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0
    finally:
        pipeline.db.close()


def cmd_list(args, settings: Settings) -> int:
    db = LeadDatabase(settings.database_path)
    rows = db.list_leads(min_score=args.min_score, status=args.status)
    for r in rows:
        print(f"#{r['id']:<4} {r['score']:>2}/10  {r['status']:<10} {(r['name'] or '')[:40]:<40} {r['url']}")
    print(f"\n{len(rows)} lead(s)")
    db.close()
    return 0


def cmd_export(args, settings: Settings) -> int:
    db = LeadDatabase(settings.database_path)
    count = db.export_csv(args.path, min_score=args.min_score)
    db.close()
    print(f"Exported {count} lead(s) to {args.path}")
    return 0


def cmd_contacted(args, settings: Settings) -> int:
    db = LeadDatabase(settings.database_path)
    ok = db.mark_contacted(args.lead_id)
    db.close()
    print(f"Lead #{args.lead_id} marked as contacted." if ok else f"No lead #{args.lead_id}.")
    return 0 if ok else 1


def cmd_test_ding(args, settings: Settings) -> int:
    alert = LeadAlert(
        name="Example Plumbing Co.",
        url="https://example.com",
        score=9,
        reasoning="Test alert from the lead pipeline.",
        query="test",
        pain_points=["No mobile-friendly website"],
        budget_signals=["5+ trucks listed on site"],
    )
    return 0 if Notifier(settings).ding(alert) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lead Generation & Pipeline Agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.set_defaults(func=cmd_run, loop=None, dry_run=False)  # bare `python main.py` == run
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the search -> score -> ding pipeline")
    run.add_argument("--loop", type=int, metavar="MINUTES", help="repeat every N minutes")
    run.add_argument("--dry-run", action="store_true", help="don't store leads or send dings")
    run.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="list stored leads")
    ls.add_argument("--min-score", type=int, default=1)
    ls.add_argument("--status", choices=["rejected", "qualified", "notified", "contacted"])
    ls.set_defaults(func=cmd_list)

    ex = sub.add_parser("export", help="export leads to CSV")
    ex.add_argument("path", nargs="?", default="leads.csv")
    ex.add_argument("--min-score", type=int, default=1)
    ex.set_defaults(func=cmd_export)

    ct = sub.add_parser("contacted", help="mark a lead as contacted")
    ct.add_argument("lead_id", type=int)
    ct.set_defaults(func=cmd_contacted)

    td = sub.add_parser("test-ding", help="send a sample alert to configured channels")
    td.set_defaults(func=cmd_test_ding)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args, Settings.from_env())


if __name__ == "__main__":
    sys.exit(main())
