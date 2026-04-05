#!/usr/bin/env python3
"""Org dedup Pass 4 — bulk TR download + local fuzzy match + AI confirmation.

Instead of scraping the TR website 18k+ times, this script:
1. Downloads the full TR XML dump (~104MB, 17k orgs) — seconds
2. Fuzzy-matches all DB stubs against the TR dump locally — minutes
3. Sends matches to AI for confirmation — parallel claude CLI calls

Usage:
    # Dry run (default)
    .venv/bin/python scripts/run_org_dedup_pass4_bulk.py --workers 5

    # Live run — applies high-confidence matches
    .venv/bin/python scripts/run_org_dedup_pass4_bulk.py --apply --workers 5

    # Resume after interruption
    .venv/bin/python scripts/run_org_dedup_pass4_bulk.py --resume --workers 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_PATH = PROJECT_ROOT / "analysis" / "org_dedup_report_bulk.csv"
TR_CACHE_PATH = PROJECT_ROOT / "analysis" / "tr_dump.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "org_dedup",
    str(PROJECT_ROOT / "pipeline" / "assets" / "lobbying" / "org_dedup.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_apply_tr_enrichment = _mod._apply_tr_enrichment

FIELDNAMES = [
    "stub_id", "stub_name", "tr_id", "tr_name", "tr_acronym",
    "tr_country", "tr_category", "tr_interests",
    "fuzzy_score", "confidence", "reasoning", "action",
]

_csv_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Pre-filter: skip orgs that will never be in the TR
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = re.compile(
    r"(?:^|\b)("
    r"permanent\s+represent|"
    r"ministry\s+of|minister\s+of|"
    r"embassy\s+of|"
    r"government\s+of|"
    r"ayuntamiento|municipalit|"
    r"^dg\s+for\s+|directorate.general|"
    r"european\s+commission|european\s+parliament|"
    r"council\s+of\s+the\s+eu|"
    r"court\s+of\s+justice|court\s+of\s+auditors|"
    r"committee\s+of\s+the\s+regions|"
    r"european\s+central\s+bank|"
    r"european\s+investment\s+bank|"
    r"^enisa$|^efsa$|^echa$|^ema$|"
    r"united\s+nations|^un\s+|"
    r"world\s+bank|^imf$|^oecd$|^nato$|"
    r"^president\s+of|^prime\s+minister"
    r")",
    re.IGNORECASE,
)


def _should_skip(name: str) -> str | None:
    m = _SKIP_PATTERNS.search(name)
    return f"Pre-filtered: matches '{m.group(0).strip()}'" if m else None


# ---------------------------------------------------------------------------
# Phase 1: Download + parse TR dump
# ---------------------------------------------------------------------------

TR_XML_URL = "https://transparency-register.europa.eu/odplastorganisationxml_en"


def _download_tr_dump() -> list[dict]:
    """Download and parse the full TR XML dump. Caches as JSON."""
    if TR_CACHE_PATH.exists():
        age_hours = (time.time() - TR_CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"  Using cached TR dump ({age_hours:.1f}h old)")
            with TR_CACHE_PATH.open(encoding="utf-8") as f:
                return json.load(f)

    print("  Downloading TR XML dump...")
    resp = requests.get(TR_XML_URL, timeout=120)
    resp.raise_for_status()
    raw = resp.text

    # Clean invalid XML character references
    raw = re.sub(r"&#x[0-1]?[0-9a-fA-F];", "", raw)

    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)

    orgs = []
    for ir in root.findall(".//interestRepresentative"):
        name = ir.findtext("name/originalName", "").strip()
        if not name:
            continue
        acronym = ir.findtext("acronym", "").strip()
        tr_id = ir.findtext("identificationCode", "").strip()
        category = ir.findtext("registrationCategory", "").strip()
        country = ""
        ho = ir.find("headOffice")
        if ho is not None:
            country = ho.findtext("country", "").strip().title()
        interests = ir.findtext("goals", "").strip()[:500]
        website = ir.findtext("webSiteURL", "").strip()

        orgs.append({
            "tr_id": tr_id,
            "name": name,
            "acronym": acronym,
            "category": category,
            "country": country,
            "interests": interests,
            "website": website,
        })

    # Cache as JSON for quick reload
    with TR_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(orgs, f, ensure_ascii=False)

    print(f"  Parsed {len(orgs)} TR organisations")
    return orgs


# ---------------------------------------------------------------------------
# Phase 2: Local fuzzy matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Normalize a name for comparison."""
    s = s.lower().strip()
    # Remove legal suffixes
    s = re.sub(
        r"\b(ltd|limited|gmbh|ag|sa|s\.a\.|s\.r\.l\.|srl|bv|nv|inc|corp|"
        r"plc|llc|e\.v\.|aisbl|asbl|vzw|ry|z\.s\.|a\.s\.|s\.p\.a\.|"
        r"s\.l\.|sl|se)\b\.?",
        "", s,
    )
    # Remove parenthetical content
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_lookup(tr_orgs: list[dict]) -> dict:
    """Build lookup structures for fast matching."""
    from rapidfuzz import process, fuzz  # noqa: F401

    # Index by normalized name and acronym
    by_name: dict[str, list[dict]] = {}
    by_acronym: dict[str, list[dict]] = {}

    for org in tr_orgs:
        norm = _normalize(org["name"])
        by_name.setdefault(norm, []).append(org)
        if org["acronym"]:
            acr = org["acronym"].lower().strip()
            by_acronym.setdefault(acr, []).append(org)

    return {
        "by_name": by_name,
        "by_acronym": by_acronym,
        "all_names": list(by_name.keys()),
        "all_orgs": tr_orgs,
    }


def _fuzzy_match(stub_name: str, lookup: dict, top_n: int = 5) -> list[dict]:
    """Find top fuzzy matches for a stub name against the TR dump.

    Returns list of {tr_org, score} dicts, sorted by score descending.
    """
    from rapidfuzz import process, fuzz

    norm = _normalize(stub_name)

    # Exact match on normalized name
    if norm in lookup["by_name"]:
        return [{"tr_org": org, "score": 100} for org in lookup["by_name"][norm][:top_n]]

    # Acronym match
    acr = norm.upper().replace(" ", "")
    if len(acr) <= 10 and acr.lower() in lookup["by_acronym"]:
        return [{"tr_org": org, "score": 95} for org in lookup["by_acronym"][acr.lower()][:top_n]]

    # Fuzzy match
    results = process.extract(
        norm,
        lookup["all_names"],
        scorer=fuzz.WRatio,
        limit=top_n,
    )

    matches = []
    for match_name, score, _ in results:
        if score < 50:
            continue
        for org in lookup["by_name"][match_name]:
            matches.append({"tr_org": org, "score": score})

    return matches[:top_n]


# ---------------------------------------------------------------------------
# Phase 3: AI confirmation (batched, parallel)
# ---------------------------------------------------------------------------

def _ai_confirm_batch_multi(groups: list[dict]) -> list[dict]:
    """Multi-candidate AI confirmation via claude CLI.

    Each group: {stub_name, candidates: [{name, acronym, country, category}, ...]}.
    Returns [{match, chosen_index, reasoning}, ...].
    """
    if not groups:
        return []

    items = []
    for i, g in enumerate(groups):
        cand_lines = []
        for j, c in enumerate(g["candidates"]):
            cand_lines.append(
                f'    {chr(65+j)}. "{c["name"]}" '
                f'(acronym: "{c.get("acronym", "")}", '
                f'country: "{c.get("country", "")}", '
                f'category: "{c.get("category", "")}")'
            )
        items.append(
            f'{i+1}. DB: "{g["stub_name"]}"\n'
            f'   TR candidates:\n' + "\n".join(cand_lines)
        )

    prompt = (
        "For each organization below, determine which TR candidate (if any) is the same "
        "entity as the DB organization. Consider name variants, acronyms, translations "
        "across EU languages, and abbreviations. The first candidate is not necessarily "
        "the best match — evaluate ALL candidates.\n\n"
        + "\n".join(items)
        + "\n\nRespond ONLY with a JSON array, one entry per DB org:\n"
        '[{"match": "high"|"medium"|"low"|"no_match", "chosen": "A"|"B"|"C"|"D"|"E"|"none", '
        '"reasoning": "one sentence"}, ...]\n'
        "IMPORTANT: Return exactly " + str(len(groups)) + " entries in order."
    )

    fallback = [{"match": "no_match", "chosen_index": -1, "reasoning": "batch_parse_failed"}] * len(groups)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = result.stdout.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return fallback
        parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, list) or len(parsed) != len(groups):
            return fallback
        valid = {"high", "medium", "low", "no_match"}
        out = []
        for entry in parsed:
            if isinstance(entry, dict) and entry.get("match") in valid:
                chosen_letter = str(entry.get("chosen", "none")).upper()
                chosen_index = ord(chosen_letter) - 65 if len(chosen_letter) == 1 and chosen_letter.isalpha() else -1
                out.append({
                    "match": entry["match"],
                    "chosen_index": chosen_index,
                    "reasoning": str(entry.get("reasoning", ""))[:200],
                })
            else:
                out.append({"match": "no_match", "chosen_index": -1, "reasoning": "invalid_entry"})
        return out
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    from dotenv import dotenv_values
    from supabase import create_client
    env = dotenv_values(PROJECT_ROOT / ".env")
    return create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])


def _fetch_stubs(client) -> list[dict]:
    stubs = []
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name")
            .is_("normalized_name", "null")
            .is_("eu_transparency_register_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        stubs.extend(resp.data)
        if len(resp.data) < 1000:
            break
        offset += 1000
    return stubs


def _fetch_active_org_ids(client) -> set[str]:
    active = set()
    for table in ("lobbying_meetings", "commission_meeting_organizations"):
        offset = 0
        while True:
            resp = (
                client.table(table)
                .select("organization_id")
                .range(offset, offset + 999)
                .execute()
            )
            for row in resp.data:
                oid = row.get("organization_id")
                if oid:
                    active.add(oid)
            if len(resp.data) < 1000:
                break
            offset += 1000
    return active


def _load_already_processed() -> set[str]:
    if not REPORT_PATH.exists():
        return set()
    done = set()
    with REPORT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("confidence") not in ("", "pending"):
                done.add(row["stub_id"])
    return done


def _append_row(row: dict) -> None:
    with _csv_lock:
        write_header = not REPORT_PATH.exists() or REPORT_PATH.stat().st_size == 0
        with REPORT_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Org dedup Pass 4 — bulk TR + fuzzy + AI")
    parser.add_argument("--apply", action="store_true", help="Apply high-confidence matches to DB")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed orgs")
    parser.add_argument("--workers", type=int, default=5, help="Parallel AI workers (default: 5)")
    parser.add_argument("--ai-batch", type=int, default=10, help="AI batch size (default: 10)")
    parser.add_argument("--min-score", type=int, default=50, help="Min fuzzy score to send to AI (default: 50)")
    parser.add_argument("--auto-accept", type=int, default=96, help="Auto-accept matches above this score (default: 96)")
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY RUN" if dry_run else "LIVE (will write to DB)"
    print(f"=== Org Dedup Pass 4 (Bulk) — {mode} ===\n")

    # Phase 1: Download TR dump
    print("--- Phase 1: TR dump ---")
    t0 = time.time()
    tr_orgs = _download_tr_dump()
    print(f"  {len(tr_orgs)} TR organisations loaded in {time.time()-t0:.1f}s")

    # Build fuzzy lookup
    print("\n--- Phase 2: Fuzzy matching ---")
    t1 = time.time()
    lookup = _build_lookup(tr_orgs)
    print(f"  Built lookup ({len(lookup['all_names'])} unique names) in {time.time()-t1:.1f}s")

    # Fetch stubs
    client = _get_client()
    print("\n  Fetching stubs...")
    all_stubs = _fetch_stubs(client)
    print(f"  Total stubs: {len(all_stubs)}")

    print("  Fetching active org IDs...")
    active_ids = _fetch_active_org_ids(client)
    stubs = [s for s in all_stubs if s["id"] in active_ids]
    print(f"  Active stubs: {len(stubs)}")

    # Pre-filter + resume
    already_done = _load_already_processed() if args.resume else set()
    if args.resume:
        stubs = [s for s in stubs if s["id"] not in already_done]
        print(f"  After resume filter: {len(stubs)}")
    elif REPORT_PATH.exists():
        REPORT_PATH.unlink()

    filtered = []
    prefilter_count = 0
    for s in stubs:
        skip_reason = _should_skip(s["name"])
        if skip_reason:
            prefilter_count += 1
            _append_row({
                "stub_id": s["id"], "stub_name": s["name"],
                "tr_id": "", "tr_name": "", "tr_acronym": "",
                "tr_country": "", "tr_category": "", "tr_interests": "",
                "fuzzy_score": "", "confidence": "prefiltered",
                "reasoning": skip_reason, "action": "skip",
            })
        else:
            filtered.append(s)

    print(f"  Pre-filtered: {prefilter_count}")
    print(f"  To match: {len(filtered)}")

    # Fuzzy match all stubs
    needs_ai: list[dict] = []
    no_match_count = 0
    auto_accepted = 0

    for i, stub in enumerate(filtered):
        matches = _fuzzy_match(stub["name"], lookup)
        best_score = matches[0]["score"] if matches else 0

        if not matches or best_score < args.min_score:
            no_match_count += 1
            _append_row({
                "stub_id": stub["id"], "stub_name": stub["name"],
                "tr_id": "", "tr_name": "", "tr_acronym": "",
                "tr_country": "", "tr_category": "", "tr_interests": "",
                "fuzzy_score": str(best_score),
                "confidence": "no_match", "reasoning": "Below fuzzy threshold",
                "action": "skip",
            })
        elif best_score >= args.auto_accept:
            auto_accepted += 1
            chosen = matches[0]["tr_org"]
            _append_row({
                "stub_id": stub["id"], "stub_name": stub["name"],
                "tr_id": chosen.get("tr_id", ""),
                "tr_name": chosen.get("name", ""),
                "tr_acronym": chosen.get("acronym", ""),
                "tr_country": chosen.get("country", ""),
                "tr_category": chosen.get("category", ""),
                "tr_interests": chosen.get("interests", "")[:200],
                "fuzzy_score": str(best_score),
                "confidence": "high",
                "reasoning": f"Auto-accepted: fuzzy score {best_score}",
                "action": "apply_dry",
            })
        else:
            needs_ai.append({
                "stub": stub,
                "candidates": matches,
            })

        if (i + 1) % 1000 == 0:
            print(
                f"  [{i+1:6d}/{len(filtered)}] "
                f"{auto_accepted} auto-accepted, {len(needs_ai)} need AI, {no_match_count} no match"
            )

    elapsed2 = time.time() - t1
    print(
        f"  Fuzzy matching done in {elapsed2:.1f}s: "
        f"{auto_accepted} auto-accepted, {len(needs_ai)} need AI, {no_match_count} no match"
    )

    if not needs_ai:
        print("\nNothing to process for Phase 3.")
        return

    # Phase 3: parallel AI confirmation
    AI_PARALLEL = args.workers
    print(f"\n--- Phase 3: AI confirmation ({len(needs_ai)} orgs, batch={args.ai_batch}, {AI_PARALLEL} parallel) ---")
    t2 = time.time()
    stats = {"high": 0, "medium": 0, "low": 0, "no_match": 0, "applied": 0, "errors": 0}
    processed_count = 0

    def _process_ai_batch(batch: list[dict]) -> list[tuple[dict, dict]]:
        groups = [
            {
                "stub_name": item["stub"]["name"],
                "candidates": [
                    {
                        "name": m["tr_org"]["name"],
                        "acronym": m["tr_org"].get("acronym", ""),
                        "country": m["tr_org"].get("country", ""),
                        "category": m["tr_org"].get("category", ""),
                    }
                    for m in item["candidates"][:5]
                ],
            }
            for item in batch
        ]
        ai_results = _ai_confirm_batch_multi(groups)
        return list(zip(batch, ai_results))

    batches = [needs_ai[i:i + args.ai_batch] for i in range(0, len(needs_ai), args.ai_batch)]

    with ThreadPoolExecutor(max_workers=AI_PARALLEL) as ai_pool:
        futures = {ai_pool.submit(_process_ai_batch, b): b for b in batches}
        for fut in as_completed(futures):
            try:
                pairs = fut.result()
            except Exception:
                processed_count += len(futures[fut])
                continue

            for item, ai in pairs:
                # Skip failed AI calls — don't write to CSV so they get retried on --resume
                if ai.get("reasoning") == "batch_parse_failed":
                    continue

                stub = item["stub"]
                idx = ai["chosen_index"]
                candidates = item["candidates"]

                if 0 <= idx < len(candidates):
                    chosen = candidates[idx]["tr_org"]
                else:
                    chosen = candidates[0]["tr_org"]

                confidence = ai["match"]
                stats[confidence] = stats.get(confidence, 0) + 1
                best_score = candidates[0]["score"]

                if confidence == "high":
                    action = "apply" if not dry_run else "apply_dry"
                elif confidence == "medium":
                    action = "review"
                else:
                    action = "skip"

                # Apply to DB if live and high confidence
                if not dry_run and confidence == "high" and chosen.get("tr_id"):
                    tr_id = chosen["tr_id"]
                    try:
                        canonical_resp = (
                            client.table("organizations")
                            .select("id,name")
                            .eq("eu_transparency_register_id", tr_id)
                            .execute()
                        )
                        canonical_orgs = canonical_resp.data or []

                        if canonical_orgs and canonical_orgs[0]["id"] != stub["id"]:
                            client.table("lobbying_meetings").update(
                                {"organization_id": canonical_orgs[0]["id"]}
                            ).eq("organization_id", stub["id"]).execute()
                            action = "applied_relink"
                        else:
                            # Build a detail-like dict for _apply_tr_enrichment
                            detail = {
                                "name": chosen["name"],
                                "acronym": chosen.get("acronym", ""),
                                "category": chosen.get("category", ""),
                                "country": chosen.get("country", ""),
                                "website": chosen.get("website", ""),
                                "interests_represented": chosen.get("interests", ""),
                            }
                            _apply_tr_enrichment(client, stub["id"], tr_id, detail, print)
                            action = "applied_enrich"
                        stats["applied"] += 1
                    except Exception as exc:
                        action = f"apply_failed: {str(exc)[:50]}"
                        stats["errors"] += 1

                _append_row({
                    "stub_id": stub["id"], "stub_name": stub["name"],
                    "tr_id": chosen.get("tr_id", ""),
                    "tr_name": chosen.get("name", ""),
                    "tr_acronym": chosen.get("acronym", ""),
                    "tr_country": chosen.get("country", ""),
                    "tr_category": chosen.get("category", ""),
                    "tr_interests": chosen.get("interests", "")[:200],
                    "fuzzy_score": str(best_score),
                    "confidence": confidence,
                    "reasoning": ai["reasoning"],
                    "action": action,
                })

            processed_count += len(pairs)
            print(
                f"  [{processed_count:5d}/{len(needs_ai)}] "
                f"high={stats['high']} medium={stats['medium']} "
                f"low={stats['low']} no_match={stats['no_match']}"
            )

    elapsed3 = time.time() - t2
    total_elapsed = time.time() - t0
    print(f"\n=== Done in {total_elapsed/60:.1f}m ===")
    print(f"  Phase 1 (TR download):  {time.time()-t0-elapsed2-elapsed3:.1f}s")
    print(f"  Phase 2 (fuzzy match):  {elapsed2:.1f}s")
    print(f"  Phase 3 (AI):           {elapsed3/60:.1f}m")
    print(f"  Pre-filtered:  {prefilter_count}")
    print(f"  No match:      {no_match_count}")
    print(f"  AI confirmed:  {len(needs_ai)}")
    print(f"    High:     {stats['high']}")
    print(f"    Medium:   {stats['medium']}")
    print(f"    Low:      {stats['low']}")
    print(f"    No match: {stats['no_match']}")
    print(f"    Applied:  {stats['applied']}")
    print(f"    Errors:   {stats['errors']}")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
