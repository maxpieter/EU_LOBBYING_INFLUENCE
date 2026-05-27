"""Count org_match_method distribution for lobbying meetings from bronze data.

Reads the raw bronze JSON partitions, splits pipe-separated attendees,
runs each through OrgResolver, and prints the method breakdown.
Does NOT write to the database — output is for the thesis table only.

Usage:
    python scripts/lobbying_org_method_stats.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from supabase import create_client

from pipeline.assets.organizations.resolution import OrgResolver
from pipeline.models.lobbying_models import Organization

BRONZE_DIR = ROOT / "data" / "eu_lobbying_bronze_meetings"
ORG_COLUMNS = "id,name,eu_transparency_register_id,acronym,country"
READ_PAGE_SIZE = 1000


def _build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("lobbying_org_method_stats")


def fetch_all_organizations(client, logger) -> list[Organization]:
    orgs: list[Organization] = []
    offset = 0
    while True:
        page = (
            client.table("organizations")
            .select(ORG_COLUMNS)
            .range(offset, offset + READ_PAGE_SIZE - 1)
            .execute()
        )
        rows = page.data or []
        if not rows:
            break
        for r in rows:
            orgs.append(
                Organization(
                    id=r["id"],
                    name=r["name"] or "",
                    eu_transparency_register_id=r.get("eu_transparency_register_id"),
                    acronym=r.get("acronym"),
                    country=r.get("country"),
                )
            )
        logger.info(f"  Loaded {len(orgs):,} organizations...")
        if len(rows) < READ_PAGE_SIZE:
            break
        offset += READ_PAGE_SIZE
    return orgs


def load_bronze_meetings(logger) -> list[dict]:
    files = sorted(BRONZE_DIR.glob("*.json"))
    logger.info(f"Found {len(files)} bronze partition files in {BRONZE_DIR}")
    meetings = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            meetings.extend(data)
    logger.info(f"Loaded {len(meetings):,} raw bronze meetings.")
    return meetings


def classify_bronze_meetings(
    raw_meetings: list[dict],
    resolver: OrgResolver,
    logger: logging.Logger,
) -> tuple[Counter[str], int, int]:
    """Replicate the silver splitting logic and collect method counts.

    Returns (method_counter, total_attendee_rows, total_resolved_to_tr).
    """
    method_counter: Counter[str] = Counter()
    tr_resolved: Counter[str] = Counter()
    total_rows = 0

    for i, meeting_row in enumerate(raw_meetings, start=1):
        attendees_raw = (meeting_row.get("attendees") or "").strip()
        transparency_id = (meeting_row.get("lobbyist_id") or "").strip() or None

        if not attendees_raw:
            continue

        org_names = [n.strip() for n in attendees_raw.split("|") if n.strip()]
        seen_names: set[str] = set()
        unique_org_names: list[str] = []
        for n in org_names:
            if n.lower() not in seen_names:
                seen_names.add(n.lower())
                unique_org_names.append(n)

        for org_name in unique_org_names:
            tr_id = transparency_id if len(unique_org_names) == 1 else None
            org, method = resolver.resolve(org_name, tr_id)
            method_counter[method] += 1
            total_rows += 1
            if org.eu_transparency_register_id:
                tr_resolved[method] += 1

        if i % 20000 == 0:
            logger.info(f"  Processed {i:,}/{len(raw_meetings):,} raw meetings...")

    return method_counter, total_rows, sum(tr_resolved.values()), tr_resolved


def main() -> int:
    logger = _build_logger()
    t0 = time.time()

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)
    logger.info("Connected to Supabase.")

    logger.info("Loading organizations to build OrgResolver...")
    orgs = fetch_all_organizations(client, logger)
    logger.info(f"Loaded {len(orgs):,} organizations.")

    resolver = OrgResolver(orgs)
    logger.info(f"OrgResolver ready: {resolver.canonical_count:,} canonical orgs.")

    logger.info("Loading bronze meeting partitions...")
    raw_meetings = load_bronze_meetings(logger)

    logger.info("Classifying attendee names through OrgResolver cascade...")
    method_counter, total_rows, total_tr, tr_by_method = classify_bronze_meetings(
        raw_meetings, resolver, logger
    )
    logger.info(f"Classification complete: {total_rows:,} attendee rows.")

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("  Lobbying meetings: org_match_method breakdown (from bronze)")
    print("=" * 70)
    print(f"  {'Method':<35}  {'Rows':>8}  {'TR resolved':>12}  {'%':>6}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*12}  {'-'*6}")

    for method, count in method_counter.most_common():
        tr = tr_by_method.get(method, 0)
        pct = 100.0 * tr / count if count else 0.0
        print(f"  {method:<35}  {count:>8,}  {tr:>12,}  {pct:>5.1f}%")

    print(f"  {'-'*35}  {'-'*8}  {'-'*12}  {'-'*6}")
    pct_total = 100.0 * total_tr / total_rows if total_rows else 0.0
    print(f"  {'TOTAL':<35}  {total_rows:>8,}  {total_tr:>12,}  {pct_total:>5.1f}%")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
