#!/usr/bin/env python3
"""Org dedup Pass 4 — bulk TR download + local fuzzy match + AI confirmation.

Two-phase design:
  Phase 1+2: Download TR dump, fuzzy-match locally, save candidates to CSV (no AI, minutes)
  Phase 3:   Classify candidates via Anthropic API in large batches (fast, cheap)

Usage:
    # Phase 1+2 only: generate candidates CSV
    .venv/bin/python scripts/run_org_dedup_pass4.py --candidates-only

    # Full run (phases 1-3), dry run
    .venv/bin/python scripts/run_org_dedup_pass4.py

    # Full run, apply high-confidence matches to DB
    .venv/bin/python scripts/run_org_dedup_pass4.py --apply

    # Phase 3 only: classify an existing candidates CSV
    .venv/bin/python scripts/run_org_dedup_pass4.py --classify-only

    # Resume Phase 3 after interruption
    .venv/bin/python scripts/run_org_dedup_pass4.py --classify-only --resume
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
CANDIDATES_PATH = PROJECT_ROOT / "analysis" / "org_dedup_candidates.csv"
REPORT_PATH = PROJECT_ROOT / "analysis" / "org_dedup_report_bulk.csv"
TR_CACHE_PATH = PROJECT_ROOT / "analysis" / "tr_dump.json"
CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "org_dedup",
    str(PROJECT_ROOT / "pipeline" / "assets" / "lobbying" / "org_dedup.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_apply_tr_enrichment = _mod._apply_tr_enrichment

CANDIDATE_FIELDS = [
    "stub_id", "stub_name",
    "c1_tr_id", "c1_name", "c1_acronym", "c1_country", "c1_category", "c1_interests", "c1_score",
    "c2_tr_id", "c2_name", "c2_acronym", "c2_country", "c2_category", "c2_interests", "c2_score",
    "c3_tr_id", "c3_name", "c3_acronym", "c3_country", "c3_category", "c3_interests", "c3_score",
]

REPORT_FIELDS = [
    "stub_id", "stub_name", "tr_id", "tr_name", "tr_acronym",
    "tr_country", "tr_category", "tr_interests",
    "fuzzy_score", "confidence", "reasoning", "action",
]

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

    with TR_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(orgs, f, ensure_ascii=False)

    print(f"  Parsed {len(orgs)} TR organisations")
    return orgs


# ---------------------------------------------------------------------------
# Phase 2: Local fuzzy matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(
        r"\b(ltd|limited|gmbh|ag|sa|s\.a\.|s\.r\.l\.|srl|bv|nv|inc|corp|"
        r"plc|llc|e\.v\.|aisbl|asbl|vzw|ry|z\.s\.|a\.s\.|s\.p\.a\.|"
        r"s\.l\.|sl|se)\b\.?",
        "", s,
    )
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_lookup(tr_orgs: list[dict]) -> dict:
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


def _fuzzy_match(stub_name: str, lookup: dict, top_n: int = 3) -> list[dict]:
    from rapidfuzz import process, fuzz

    norm = _normalize(stub_name)

    if norm in lookup["by_name"]:
        return [{"tr_org": org, "score": 100} for org in lookup["by_name"][norm][:top_n]]

    acr = norm.upper().replace(" ", "")
    if len(acr) <= 10 and acr.lower() in lookup["by_acronym"]:
        return [{"tr_org": org, "score": 95} for org in lookup["by_acronym"][acr.lower()][:top_n]]

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
# Helpers: DB access
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


# ---------------------------------------------------------------------------
# Phase 2 output: write candidates CSV
# ---------------------------------------------------------------------------

def run_candidates(args):
    """Phase 1+2: download TR dump, fuzzy match, write candidates CSV."""
    print("--- Phase 1: TR dump ---")
    t0 = time.time()
    tr_orgs = _download_tr_dump()
    print(f"  {len(tr_orgs)} TR organisations loaded in {time.time()-t0:.1f}s")

    print("\n--- Phase 2: Fuzzy matching ---")
    t1 = time.time()
    lookup = _build_lookup(tr_orgs)
    print(f"  Built lookup ({len(lookup['all_names'])} unique names) in {time.time()-t1:.1f}s")

    client = _get_client()
    print("\n  Fetching stubs...")
    all_stubs = _fetch_stubs(client)
    print(f"  Total stubs: {len(all_stubs)}")

    print("  Fetching active org IDs...")
    active_ids = _fetch_active_org_ids(client)
    stubs = [s for s in all_stubs if s["id"] in active_ids]
    print(f"  Active stubs: {len(stubs)}")

    # Write report for prefiltered + no_match (these don't need AI)
    report_rows = []
    candidates_rows = []
    no_match_count = 0
    prefilter_count = 0
    auto_accepted = 0

    for i, stub in enumerate(stubs):
        skip_reason = _should_skip(stub["name"])
        if skip_reason:
            prefilter_count += 1
            report_rows.append({
                "stub_id": stub["id"], "stub_name": stub["name"],
                "tr_id": "", "tr_name": "", "tr_acronym": "",
                "tr_country": "", "tr_category": "", "tr_interests": "",
                "fuzzy_score": "", "confidence": "prefiltered",
                "reasoning": skip_reason, "action": "skip",
            })
            continue

        matches = _fuzzy_match(stub["name"], lookup)
        best_score = matches[0]["score"] if matches else 0

        if not matches or best_score < args.min_score:
            no_match_count += 1
            report_rows.append({
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
            report_rows.append({
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
            # Save candidates for AI classification
            row = {"stub_id": stub["id"], "stub_name": stub["name"]}
            for ci, m in enumerate(matches[:3], 1):
                org = m["tr_org"]
                row[f"c{ci}_tr_id"] = org.get("tr_id", "")
                row[f"c{ci}_name"] = org.get("name", "")
                row[f"c{ci}_acronym"] = org.get("acronym", "")
                row[f"c{ci}_country"] = org.get("country", "")
                row[f"c{ci}_category"] = org.get("category", "")
                row[f"c{ci}_interests"] = org.get("interests", "")[:200]
                row[f"c{ci}_score"] = str(m["score"])
            # Fill empty candidate slots
            for ci in range(len(matches[:3]) + 1, 4):
                for f in ("tr_id", "name", "acronym", "country", "category", "interests", "score"):
                    row[f"c{ci}_{f}"] = ""
            candidates_rows.append(row)

        if (i + 1) % 1000 == 0:
            print(
                f"  [{i+1:6d}/{len(stubs)}] "
                f"{auto_accepted} auto, {len(candidates_rows)} need AI, "
                f"{no_match_count} no match, {prefilter_count} prefiltered"
            )

    # Write candidates CSV
    with CANDIDATES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidates_rows)

    # Write already-resolved rows to report
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    elapsed = time.time() - t0
    print(f"\n=== Phase 1+2 done in {elapsed:.1f}s ===")
    print(f"  Pre-filtered:  {prefilter_count}")
    print(f"  No match:      {no_match_count}")
    print(f"  Auto-accepted: {auto_accepted}")
    print(f"  Need AI:       {len(candidates_rows)}")
    print(f"  Candidates:    {CANDIDATES_PATH}")
    print(f"  Report:        {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Phase 3: AI classification via Anthropic API
# ---------------------------------------------------------------------------

def _build_ai_prompt(batch: list[dict]) -> str:
    items = []
    for i, row in enumerate(batch):
        cand_lines = []
        for ci in range(1, 4):
            name = row.get(f"c{ci}_name", "")
            if not name:
                continue
            cand_lines.append(
                f'    {chr(64+ci)}. "{name}" '
                f'(acronym: "{row.get(f"c{ci}_acronym", "")}", '
                f'country: "{row.get(f"c{ci}_country", "")}", '
                f'category: "{row.get(f"c{ci}_category", "")}")'
            )
        items.append(
            f'{i+1}. DB: "{row["stub_name"]}"\n'
            f'   TR candidates:\n' + "\n".join(cand_lines)
        )

    return (
        "For each organization below, determine which TR candidate (if any) is the same "
        "entity as the DB organization. Consider name variants, acronyms, translations "
        "across EU languages, and abbreviations. The first candidate is not necessarily "
        "the best match — evaluate ALL candidates.\n\n"
        + "\n".join(items)
        + "\n\nRespond ONLY with a JSON array, one entry per DB org:\n"
        '[{"match": "high"|"medium"|"low"|"no_match", "chosen": "A"|"B"|"C"|"none", '
        '"reasoning": "one sentence"}, ...]\n'
        f"IMPORTANT: Return exactly {len(batch)} entries in order."
    )


def run_classify(args):
    """Phase 3: classify candidates CSV via Anthropic API."""
    import anthropic
    from dotenv import dotenv_values
    env = dotenv_values(PROJECT_ROOT / ".env")
    api_key = env.get("ANTHROPIC_API_KEY")

    if not CANDIDATES_PATH.exists():
        print(f"Error: candidates file not found: {CANDIDATES_PATH}")
        print("Run with --candidates-only first.")
        return

    # Load candidates
    with CANDIDATES_PATH.open(encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_PATH}")

    # Load existing report (prefiltered + no_match + auto-accepted)
    existing_report = []
    already_classified = set()
    if REPORT_PATH.exists():
        with REPORT_PATH.open(encoding="utf-8") as f:
            existing_report = list(csv.DictReader(f))
        if args.resume:
            already_classified = {
                r["stub_id"] for r in existing_report
                if r.get("confidence") not in ("", "pending")
            }
            candidates = [c for c in candidates if c["stub_id"] not in already_classified]
            print(f"  Resuming: {len(candidates)} remaining after skipping {len(already_classified)} already classified")

    client = anthropic.Anthropic(api_key=api_key)
    dry_run = not args.apply

    batch_size = args.ai_batch
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    print(f"\n--- Phase 3: AI classification ({len(candidates)} orgs, {len(batches)} batches of {batch_size}) ---")

    t0 = time.time()
    stats = {"high": 0, "medium": 0, "low": 0, "no_match": 0, "errors": 0}
    new_report_rows = []
    workers = args.workers
    processed = 0

    def _classify_batch(bi_batch):
        bi, batch = bi_batch
        prompt = _build_ai_prompt(batch)
        for attempt in range(5):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    wait = 2 ** attempt * 3  # 3, 6, 12, 24, 48s
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError(f"Rate limited after 5 retries")
        raw = response.content[0].text
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array in response")
        parsed = json.loads(json_match.group(0))
        if len(parsed) != len(batch):
            raise ValueError(f"Expected {len(batch)} entries, got {len(parsed)}")
        return bi, batch, parsed

    print(f"  Using {workers} parallel workers")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_classify_batch, (bi, b)): (bi, b) for bi, b in enumerate(batches)}

        for fut in as_completed(futures):
            bi, batch = futures[fut]
            try:
                _, batch, parsed = fut.result()
            except Exception as e:
                print(f"  Batch {bi+1} failed: {e}")
                stats["errors"] += len(batch)
                processed += len(batch)
                continue

            valid = {"high", "medium", "low", "no_match"}
            for row, entry in zip(batch, parsed):
                if not isinstance(entry, dict) or entry.get("match") not in valid:
                    confidence = "no_match"
                    reasoning = "invalid_entry"
                    chosen_ci = 1
                else:
                    confidence = entry["match"]
                    reasoning = str(entry.get("reasoning", ""))[:200]
                    chosen_letter = str(entry.get("chosen", "none")).upper()
                    chosen_ci = ord(chosen_letter) - 64 if len(chosen_letter) == 1 and chosen_letter.isalpha() and chosen_letter in "ABC" else 1

                stats[confidence] = stats.get(confidence, 0) + 1

                if confidence == "high":
                    action = "apply" if not dry_run else "apply_dry"
                elif confidence == "medium":
                    action = "review"
                else:
                    action = "skip"

                chosen_name = row.get(f"c{chosen_ci}_name", row.get("c1_name", ""))
                best_score = row.get("c1_score", "")

                new_report_rows.append({
                    "stub_id": row["stub_id"],
                    "stub_name": row["stub_name"],
                    "tr_id": row.get(f"c{chosen_ci}_tr_id", ""),
                    "tr_name": chosen_name,
                    "tr_acronym": row.get(f"c{chosen_ci}_acronym", ""),
                    "tr_country": row.get(f"c{chosen_ci}_country", ""),
                    "tr_category": row.get(f"c{chosen_ci}_category", ""),
                    "tr_interests": row.get(f"c{chosen_ci}_interests", ""),
                    "fuzzy_score": best_score,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "action": action,
                })

            processed += len(batch)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(candidates) - processed) / rate if rate > 0 else 0
            print(
                f"  [{processed:5d}/{len(candidates)}] "
                f"high={stats['high']} med={stats['medium']} low={stats['low']} "
                f"no={stats['no_match']} err={stats['errors']} "
                f"({rate:.0f}/s, ETA {eta/60:.1f}m)"
            )

    # Write final report: existing rows + new AI rows
    all_rows = existing_report + new_report_rows
    # Write to temp file first, then rename (safe against interruption)
    tmp_path = REPORT_PATH.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    tmp_path.rename(REPORT_PATH)

    elapsed = time.time() - t0
    print(f"\n=== Phase 3 done in {elapsed/60:.1f}m ===")
    print(f"  High:     {stats['high']}")
    print(f"  Medium:   {stats['medium']}")
    print(f"  Low:      {stats['low']}")
    print(f"  No match: {stats['no_match']}")
    print(f"  Errors:   {stats['errors']}")
    print(f"  Report:   {REPORT_PATH}")

    # Apply to DB if requested
    if not dry_run:
        _apply_results(new_report_rows)


def _apply_results(rows: list[dict]):
    """Apply high-confidence matches to the database."""
    client = _get_client()
    applied = 0
    errors = 0

    high_rows = [r for r in rows if r["confidence"] == "high" and r["tr_id"]]
    print(f"\n--- Applying {len(high_rows)} high-confidence matches ---")

    for r in high_rows:
        tr_id = r["tr_id"]
        stub_id = r["stub_id"]
        try:
            canonical_resp = (
                client.table("organizations")
                .select("id,name")
                .eq("eu_transparency_register_id", tr_id)
                .execute()
            )
            canonical_orgs = canonical_resp.data or []

            if canonical_orgs and canonical_orgs[0]["id"] != stub_id:
                client.table("lobbying_meetings").update(
                    {"organization_id": canonical_orgs[0]["id"]}
                ).eq("organization_id", stub_id).execute()
            else:
                detail = {
                    "name": r["tr_name"],
                    "acronym": r.get("tr_acronym", ""),
                    "category": r.get("tr_category", ""),
                    "country": r.get("tr_country", ""),
                    "website": "",
                    "interests_represented": r.get("tr_interests", ""),
                }
                _apply_tr_enrichment(client, stub_id, tr_id, detail, print)
            applied += 1
        except Exception as exc:
            print(f"  Error applying {stub_id}: {str(exc)[:80]}")
            errors += 1

    print(f"  Applied: {applied}, Errors: {errors}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Org dedup Pass 4 — bulk TR + fuzzy + AI")
    parser.add_argument("--apply", action="store_true", help="Apply high-confidence matches to DB")
    parser.add_argument("--resume", action="store_true", help="Resume Phase 3 from where it left off")
    parser.add_argument("--candidates-only", action="store_true", help="Only run Phase 1+2 (no AI)")
    parser.add_argument("--classify-only", action="store_true", help="Only run Phase 3 (AI classification)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel API workers (default: 5)")
    parser.add_argument("--ai-batch", type=int, default=50, help="AI batch size (default: 50)")
    parser.add_argument("--min-score", type=int, default=50, help="Min fuzzy score (default: 50)")
    parser.add_argument("--auto-accept", type=int, default=96, help="Auto-accept above this score (default: 96)")
    args = parser.parse_args()

    if args.classify_only:
        run_classify(args)
    elif args.candidates_only:
        run_candidates(args)
    else:
        run_candidates(args)
        run_classify(args)


if __name__ == "__main__":
    main()
