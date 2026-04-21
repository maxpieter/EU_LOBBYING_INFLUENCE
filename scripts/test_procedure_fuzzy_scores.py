"""Test rapidfuzz meeting→procedure matching at different score thresholds.

Runs a sample of pending meetings (match_status=NULL) through the
ProcedureMatcher's trigram step and shows score distribution + samples
at each tier so you can calibrate the AI threshold (min_ai in matching.py).

Usage:
    python scripts/test_procedure_fuzzy_scores.py
    python scripts/test_procedure_fuzzy_scores.py --sample 8 --max-meetings 50000
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5, help="Examples per score tier")
    parser.add_argument("--max-meetings", type=int, default=0, help="Cap meetings to test (0=all)")
    args = parser.parse_args()

    env = dotenv_values(Path(__file__).parent.parent / ".env")
    client = create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])

    # Load matcher
    from pipeline.assets.procedures.matching import ProcedureMatcher

    print("Loading procedures + aliases...")
    procedures: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("procedures")
            .select("id,title,procedure_type,proposal_date,decision_date,last_activity_date")
            .eq("is_deleted", False)
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        procedures.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    aliases: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("procedure_aliases")
            .select("procedure_id,alias,alias_type")
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        aliases.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    matcher = ProcedureMatcher(procedures, aliases)
    print(f"Loaded {len(procedures)} procedures, {len(aliases)} aliases")

    # Load pending meetings
    print("Loading pending meetings (match_status IS NULL)...")
    meetings: list[dict] = []

    offset = 0
    while True:
        resp = (
            client.table("lobbying_meetings")
            .select("id,title,meeting_date")
            .is_("match_status", "null")
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        for m in batch:
            meetings.append({
                "text": (m.get("title") or "").strip(),
                "date": m.get("meeting_date"),
                "source": "lobbying",
            })
        if len(batch) < 1000:
            break
        offset += 1000

    offset = 0
    while True:
        resp = (
            client.table("commission_meetings")
            .select("id,subject,meeting_date,points_raised")
            .is_("match_status", "null")
            .range(offset, offset + 999)
            .execute()
        )
        batch = resp.data or []
        for m in batch:
            text = (m.get("subject") or "").strip()
            points = m.get("points_raised")
            if points and isinstance(points, str):
                text = f"{text} {points}".strip()
            elif points and isinstance(points, list):
                text = f"{text} {' '.join(str(p) for p in points)}".strip()
            meetings.append({
                "text": text,
                "date": m.get("meeting_date"),
                "source": "commission",
            })
        if len(batch) < 1000:
            break
        offset += 1000

    print(f"Loaded {len(meetings)} pending meetings")

    if args.max_meetings and len(meetings) > args.max_meetings:
        random.seed(42)
        meetings = random.sample(meetings, args.max_meetings)
        print(f"Sampled down to {len(meetings)}")

    # Run rapidfuzz on each meeting
    print(f"\nRunning rapidfuzz on {len(meetings)} meetings...")

    tiers: dict[str, list[tuple[str, str, str, float]]] = {
        "90-100": [],
        "80-89": [],
        "70-79": [],
        "60-69": [],
        "50-59": [],
        "45-49": [],
        "below_45": [],
    }

    fine_tiers: dict[int, list[tuple[str, str, str, float]]] = {s: [] for s in range(45, 101)}

    score_counter: Counter[str] = Counter()
    no_text = 0
    no_candidates = 0
    total = len(meetings)
    report_every = max(1, total // 10)

    for i, m in enumerate(meetings, 1):
        text = m["text"]
        if not text or len(text) < 3:
            no_text += 1
            continue

        candidates = matcher._fuzzy_match(text)
        if not candidates:
            no_candidates += 1
            continue

        best = candidates[0]
        score = best["score"]
        proc_id = best["procedure_id"]
        proc_title = matcher._proc_titles.get(proc_id, "?")

        entry = (text[:80], proc_title[:80], proc_id, score)

        # Fine-grained
        s_int = min(100, max(0, int(score)))
        if s_int in fine_tiers:
            fine_tiers[s_int].append(entry)

        # Coarse tiers
        if score >= 90:
            tiers["90-100"].append(entry)
            score_counter["90-100"] += 1
        elif score >= 80:
            tiers["80-89"].append(entry)
            score_counter["80-89"] += 1
        elif score >= 70:
            tiers["70-79"].append(entry)
            score_counter["70-79"] += 1
        elif score >= 60:
            tiers["60-69"].append(entry)
            score_counter["60-69"] += 1
        elif score >= 50:
            tiers["50-59"].append(entry)
            score_counter["50-59"] += 1
        elif score >= 45:
            tiers["45-49"].append(entry)
            score_counter["45-49"] += 1
        else:
            tiers["below_45"].append(entry)
            score_counter["below_45"] += 1

        if i % report_every == 0:
            print(f"  {i}/{total}...")

    # Print score distribution
    print(f"\n{'='*90}")
    print(f"SCORE DISTRIBUTION ({total} meetings, {no_text} empty, {no_candidates} no candidates)")
    print(f"{'='*90}\n")

    for tier_name in ["90-100", "80-89", "70-79", "60-69", "50-59", "45-49", "below_45"]:
        count = score_counter[tier_name]
        pct = 100.0 * count / total if total else 0
        print(f"  {tier_name:12s}: {count:6d} ({pct:5.1f}%)")

    # Fine-grained around the interesting range (45-70)
    print(f"\n{'='*90}")
    print("FINE-GRAINED DISTRIBUTION (45-80)")
    print(f"{'='*90}\n")
    for s in range(80, 44, -1):
        count = len(fine_tiers[s])
        bar = "#" * min(50, count // 10)
        print(f"  {s:3d}: {count:5d} {bar}")

    # Samples per tier
    print(f"\n{'='*90}")
    print("SAMPLES PER TIER")
    print(f"{'='*90}")

    random.seed(42)
    for tier_name in ["90-100", "80-89", "70-79", "60-69", "50-59", "45-49"]:
        entries = tiers[tier_name]
        if not entries:
            continue
        print(f"\n--- {tier_name} ({len(entries)} total) ---")
        sample = random.sample(entries, min(args.sample, len(entries)))
        for text, proc_title, proc_id, score in sample:
            print(f"  [{score:5.1f}] \"{text}\"")
            print(f"       → \"{proc_title}\" [{proc_id}]")

    # Recommendation
    print(f"\n{'='*90}")
    print("RECOMMENDATION")
    print(f"{'='*90}")
    for threshold in [80, 75, 70, 65, 60, 55, 50, 45]:
        ai_count = sum(
            score_counter[t] for t in score_counter
            if t != "below_45" and _tier_min(t) >= threshold
        )
        print(f"  min_ai={threshold}: {ai_count:6d} meetings sent to AI")


def _tier_min(tier: str) -> int:
    if tier == "below_45":
        return 0
    return int(tier.split("-")[0])


if __name__ == "__main__":
    main()
