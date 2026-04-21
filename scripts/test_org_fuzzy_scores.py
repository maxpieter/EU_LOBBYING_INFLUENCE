"""Test rapidfuzz org matching at different score thresholds.

Runs the 19,325 unprocessed stubs + 1,227 medium stubs through rapidfuzz
against the TR dump and shows score distribution + sample matches at each
tier so you can calibrate auto_accept_threshold and min_score.

Usage:
    python scripts/test_org_fuzzy_scores.py
    python scripts/test_org_fuzzy_scores.py --sample 10  # more examples per tier
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.assets.organizations.fuzzy import (
    build_tr_lookup,
    download_tr_dump,
    fuzzy_match_local,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5, help="Examples per score tier")
    args = parser.parse_args()

    env = dotenv_values(Path(__file__).parent.parent / ".env")
    client = create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])

    # Load stubs that need processing
    print("Loading stubs from Supabase...")
    stubs: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name,dedup_status")
            .is_("eu_transparency_register_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        stubs.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    # Filter to unprocessed + medium (the ones that would be re-evaluated)
    candidates = [
        s for s in stubs
        if s.get("dedup_status") in (None, "medium")
    ]
    print(f"Total stubs: {len(stubs)}, candidates to test: {len(candidates)}")

    # Download TR dump
    print("Downloading TR dump...")
    tr_orgs = download_tr_dump()
    tr_lookup = build_tr_lookup(tr_orgs)
    print(f"TR dump: {len(tr_orgs)} orgs, {len(tr_lookup['all_names'])} unique names")

    # Run fuzzy matching on all candidates
    print(f"\nRunning rapidfuzz on {len(candidates)} stubs...")
    tiers: dict[str, list[tuple[str, str, float]]] = {
        "100_exact": [],
        "96-99_very_high": [],
        "90-95_high": [],
        "80-89_good": [],
        "70-79_medium": [],
        "60-69_low": [],
        "50-59_very_low": [],
        "below_50": [],
    }

    score_counter: Counter[str] = Counter()
    total = len(candidates)
    report_every = max(1, total // 10)

    for i, stub in enumerate(candidates, 1):
        matches = fuzzy_match_local(stub["name"], tr_lookup, top_n=1, min_score=0)
        if not matches:
            score_counter["no_candidate"] += 1
            continue

        best = matches[0]
        score = best["score"]
        tr_name = best["tr_org"]["name"]
        stub_name = stub["name"]

        entry = (stub_name, tr_name, score)

        if score == 100:
            tiers["100_exact"].append(entry)
            score_counter["100_exact"] += 1
        elif score >= 96:
            tiers["96-99_very_high"].append(entry)
            score_counter["96-99_very_high"] += 1
        elif score >= 90:
            tiers["90-95_high"].append(entry)
            score_counter["90-95_high"] += 1
        elif score >= 80:
            tiers["80-89_good"].append(entry)
            score_counter["80-89_good"] += 1
        elif score >= 70:
            tiers["70-79_medium"].append(entry)
            score_counter["70-79_medium"] += 1
        elif score >= 60:
            tiers["60-69_low"].append(entry)
            score_counter["60-69_low"] += 1
        elif score >= 50:
            tiers["50-59_very_low"].append(entry)
            score_counter["50-59_very_low"] += 1
        else:
            tiers["below_50"].append(entry)
            score_counter["below_50"] += 1

        if i % report_every == 0:
            print(f"  {i}/{total}...")

    # Print results
    print(f"\n{'='*80}")
    print(f"SCORE DISTRIBUTION ({len(candidates)} stubs tested)")
    print(f"{'='*80}\n")

    for tier_name in [
        "100_exact", "96-99_very_high", "90-95_high", "80-89_good",
        "70-79_medium", "60-69_low", "50-59_very_low", "below_50",
    ]:
        count = score_counter[tier_name]
        pct = 100.0 * count / len(candidates) if candidates else 0
        print(f"  {tier_name:20s}: {count:6d} ({pct:5.1f}%)")
    print(f"  {'no_candidate':20s}: {score_counter['no_candidate']:6d}")

    print(f"\n{'='*80}")
    print(f"SAMPLES PER TIER (judge true/false positive yourself)")
    print(f"{'='*80}")

    for tier_name in [
        "100_exact", "96-99_very_high", "90-95_high", "80-89_good",
        "70-79_medium", "60-69_low", "50-59_very_low",
    ]:
        entries = tiers[tier_name]
        if not entries:
            continue
        print(f"\n--- {tier_name} ({len(entries)} total) ---")
        # Show a diverse sample (spread across the list)
        step = max(1, len(entries) // args.sample)
        for j in range(0, min(len(entries), args.sample * step), step):
            stub_name, tr_name, score = entries[j]
            print(f"  [{score:5.1f}] \"{stub_name}\"")
            print(f"       → \"{tr_name}\"")

    # Summary recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    safe_auto = score_counter["100_exact"] + score_counter["96-99_very_high"]
    needs_review = score_counter["90-95_high"] + score_counter["80-89_good"]
    ai_candidates = score_counter["70-79_medium"] + score_counter["60-69_low"] + score_counter["50-59_very_low"]
    print(f"  Auto-accept (>=96):  {safe_auto:6d} — review the 96-99 samples above")
    print(f"  Needs review (80-95):{needs_review:6d} — consider raising auto_accept or sending to AI")
    print(f"  AI candidates (50-79):{ai_candidates:5d} — send to Haiku (~${ai_candidates * 0.001:.2f})")
    print(f"  Below 50 / no match: {score_counter['below_50'] + score_counter['no_candidate']:6d} — skip")


if __name__ == "__main__":
    main()
