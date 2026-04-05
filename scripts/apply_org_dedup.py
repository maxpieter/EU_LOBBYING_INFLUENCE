#!/usr/bin/env python3
"""Apply reviewed org dedup matches from the CSV to Supabase.

Reads analysis/org_dedup_report.csv and applies all rows where action="apply".
Edit the CSV in Excel/Sheets first — change "review" to "apply" for matches
you've verified, then run this script.

Usage:
    # Dry run (default) — shows what would be applied
    .venv/bin/python scripts/apply_org_dedup.py

    # Live — actually writes to Supabase
    .venv/bin/python scripts/apply_org_dedup.py --live
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_PATH = PROJECT_ROOT / "analysis" / "org_dedup_report.csv"

# Load _apply_tr_enrichment from org_dedup
_spec = importlib.util.spec_from_file_location(
    "org_dedup",
    str(PROJECT_ROOT / "pipeline" / "assets" / "lobbying" / "org_dedup.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_apply_tr_enrichment = _mod._apply_tr_enrichment
_scrape_tr_detail = _mod._scrape_tr_detail


def _get_client():
    from dotenv import dotenv_values
    from supabase import create_client
    env = dotenv_values(PROJECT_ROOT / ".env")
    return create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])


def main():
    parser = argparse.ArgumentParser(description="Apply reviewed org dedup matches to Supabase")
    parser.add_argument("--live", action="store_true", help="Actually write to DB (default: dry run)")
    args = parser.parse_args()

    if not REPORT_PATH.exists():
        print(f"No report found at {REPORT_PATH}")
        return

    # Read all rows with action="apply"
    to_apply = []
    with REPORT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip() == "apply":
                to_apply.append(row)

    print(f"Found {len(to_apply)} rows with action=apply\n")

    if not to_apply:
        print("Nothing to apply. Edit the CSV and set action=apply for matches you want.")
        return

    if not args.live:
        print("DRY RUN — showing what would be applied:\n")
        for row in to_apply:
            print(f"  {row['stub_name'][:40]:40s} -> {row['tr_name'][:40]:40s} [{row['confidence']}]")
        print(f"\nRun with --live to apply these {len(to_apply)} matches.")
        return

    client = _get_client()
    stats = {"relinked": 0, "enriched": 0, "errors": 0}

    for row in to_apply:
        stub_id = row["stub_id"]
        stub_name = row["stub_name"]
        tr_id = row["tr_id"]

        if not tr_id:
            print(f"  SKIP {stub_name} — no TR ID")
            continue

        try:
            # Check if a canonical org with this TR ID already exists
            canonical_resp = (
                client.table("organizations")
                .select("id,name")
                .eq("eu_transparency_register_id", tr_id)
                .execute()
            )
            canonical_orgs = canonical_resp.data or []

            if canonical_orgs and canonical_orgs[0]["id"] != stub_id:
                # Relink meetings from stub to canonical
                canonical = canonical_orgs[0]
                client.table("lobbying_meetings").update(
                    {"organization_id": canonical["id"]}
                ).eq("organization_id", stub_id).execute()
                stats["relinked"] += 1
                print(f"  RELINKED {stub_name[:35]:35s} -> {canonical['name'][:35]} (TR {tr_id})")
            else:
                # Enrich the stub with TR data
                detail = _scrape_tr_detail(tr_id)
                if detail:
                    _apply_tr_enrichment(client, stub_id, tr_id, detail, print)
                    stats["enriched"] += 1
                    print(f"  ENRICHED {stub_name[:35]:35s} with TR {tr_id}")
                else:
                    print(f"  SKIP {stub_name[:35]:35s} — could not fetch TR detail")
                    stats["errors"] += 1

        except Exception as exc:
            print(f"  ERROR {stub_name[:35]:35s} — {exc}")
            stats["errors"] += 1

    print(f"\n=== Done ===")
    print(f"  Relinked: {stats['relinked']}")
    print(f"  Enriched: {stats['enriched']}")
    print(f"  Errors:   {stats['errors']}")


if __name__ == "__main__":
    main()
