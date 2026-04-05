#!/usr/bin/env python3
"""Overnight runner for org dedup Pass 4 (TR web search).

Optimizations:
- Pre-filters government bodies, permanent representations, EU institutions
- Quick string similarity check skips obvious non-matches before AI
- Batched AI calls (10 comparisons per prompt)
- Parallel TR scraping (configurable workers)
- Incremental CSV save (resume-safe)

Usage:
    # Dry run (default) — produces CSV report, no DB writes
    .venv/bin/python scripts/run_org_dedup_pass4.py --resume --workers 5

    # Live run — applies high-confidence matches
    .venv/bin/python scripts/run_org_dedup_pass4.py --apply --resume --workers 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as _requests

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_PATH = PROJECT_ROOT / "analysis" / "org_dedup_report.csv"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "org_dedup",
    str(PROJECT_ROOT / "pipeline" / "assets" / "lobbying" / "org_dedup.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_search_variants = _mod._search_variants
_apply_tr_enrichment = _mod._apply_tr_enrichment

# ---------------------------------------------------------------------------
# Fast HTTP scraping (shared session, no sleeps)
# ---------------------------------------------------------------------------

_TR_SEARCH_URL = "https://ec.europa.eu/transparencyregister/public/search?lang=en&queryText={query}"
_session = _requests.Session()
_session.headers["Accept-Language"] = "en"
# Keep-alive connection pooling — much faster than individual requests
_adapter = _requests.adapters.HTTPAdapter(
    pool_connections=20, pool_maxsize=50, max_retries=2
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _scrape_tr_search(query: str) -> list[dict]:
    url = _TR_SEARCH_URL.format(query=_requests.utils.quote(query))
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []
    entries: list[dict] = []
    for m in re.finditer(
        r'href=["\']search-details_en\?id=([0-9]+-[0-9]+)["\'][^>]*>(.*?)</a>',
        resp.text, re.DOTALL | re.IGNORECASE,
    ):
        tr_id = m.group(1).strip()
        name = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        name = re.sub(r"\s+", " ", name).strip()
        if tr_id and name:
            entries.append({"name": name, "tr_id": tr_id})
        if len(entries) >= 5:
            break
    return entries


def _scrape_tr_detail(tr_id: str) -> dict | None:
    url = f"https://ec.europa.eu/transparencyregister/public/PUBLIC/ORGANISATION/{tr_id}?lang=en"
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return None
    import html as html_mod
    html = html_mod.unescape(resp.text)

    def _cell(label: str) -> str:
        pat = re.compile(
            r'<td[^>]*>[^<]*<strong>\s*' + re.escape(label)
            + r'\s*</strong>[^<]*</td>\s*<td[^>]*>(.*?)</td>',
            re.DOTALL | re.IGNORECASE,
        )
        m = pat.search(html)
        if not m:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(1))).strip()

    name = _cell("Organisation name")
    if not name:
        return None
    interests = _cell("Applicant/registrant's representation") or _cell("Interests represented")
    country = ""
    cm = re.search(
        r'(?:GERMANY|BELGIUM|FRANCE|DENMARK|NETHERLANDS|ITALY|SPAIN|AUSTRIA|'
        r'SWEDEN|FINLAND|PORTUGAL|GREECE|POLAND|CZECH\s*REPUBLIC|HUNGARY|'
        r'ROMANIA|CROATIA|SLOVAKIA|SLOVENIA|BULGARIA|CYPRUS|ESTONIA|LATVIA|'
        r'LITHUANIA|LUXEMBOURG|MALTA|IRELAND|UNITED\s*KINGDOM|SWITZERLAND|'
        r'NORWAY|UNITED\s*STATES)', html,
    )
    if cm:
        country = cm.group(0).strip().title()
    def _c(v: str) -> str:
        return "" if v.strip().lower() in {"n/a", "n/a.", "-", "none"} else v.strip()
    return {
        "name": _c(name), "acronym": _c(_cell("Acronym")),
        "interests_represented": _c(interests), "category": _c(_cell("Category of registration")),
        "country": _c(country), "website": _c(_cell("Website")),
    }

FIELDNAMES = [
    "stub_id", "stub_name", "tr_id", "tr_name", "tr_acronym",
    "tr_country", "tr_category", "tr_interests",
    "confidence", "reasoning", "action",
]

# Thread-safe CSV writing
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
    r"^enisa$|^efsa$|^echa$|^ema$|"  # EU agencies
    r"united\s+nations|^un\s+|"
    r"world\s+bank|^imf$|^oecd$|^nato$|"
    r"^president\s+of|^prime\s+minister"
    r")",
    re.IGNORECASE,
)


def _should_skip(name: str) -> str | None:
    """Return a reason string if this org should be skipped, else None."""
    m = _SKIP_PATTERNS.search(name)
    if m:
        return f"Pre-filtered: matches '{m.group(0).strip()}'"
    return None


# ---------------------------------------------------------------------------
# Quick string similarity (avoid AI for obvious non-matches)
# ---------------------------------------------------------------------------

def _quick_similarity(stub_name: str, tr_name: str, tr_acronym: str) -> bool:
    """Always returns True — we let the AI decide.

    The TR search already does relevance ranking. If the search returned a
    result, it's worth asking the AI about it. The AI is the smart filter.
    """
    return True


# ---------------------------------------------------------------------------
# Batched AI confirmation
# ---------------------------------------------------------------------------

def _ai_confirm_batch(pairs: list[dict]) -> list[dict]:
    """Confirm multiple stub-TR pairs in a single AI call.

    Each pair should have: stub_name, tr_name, tr_acronym, tr_country, tr_category, tr_interests.
    Returns a list of {"match": ..., "reasoning": ...} dicts, same order as input.
    """
    if not pairs:
        return []

    items = []
    for i, p in enumerate(pairs):
        items.append(
            f'{i+1}. DB: "{p["stub_name"]}" -> TR: "{p["tr_name"]}" '
            f'(acronym: "{p.get("tr_acronym", "")}", '
            f'country: "{p.get("tr_country", "")}", '
            f'category: "{p.get("tr_category", "")}")'
        )

    prompt = (
        "For each pair below, determine if the database organization (DB) is the same "
        "entity as the EU Transparency Register result (TR). Consider name variants, "
        "acronyms, translations across EU languages, and abbreviations.\n\n"
        + "\n".join(items)
        + "\n\nRespond ONLY with a JSON array, one entry per pair:\n"
        '[{"match": "high"|"medium"|"low"|"no_match", "reasoning": "one sentence"}, ...]\n'
        "IMPORTANT: Return exactly " + str(len(pairs)) + " entries in order."
    )

    fallback = [{"match": "no_match", "reasoning": "batch_parse_failed"}] * len(pairs)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = result.stdout.strip()
        # Extract JSON array
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return fallback
        parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, list) or len(parsed) != len(pairs):
            return fallback
        # Validate each entry
        valid = {"high", "medium", "low", "no_match"}
        results = []
        for entry in parsed:
            if isinstance(entry, dict) and entry.get("match") in valid:
                results.append({
                    "match": entry["match"],
                    "reasoning": str(entry.get("reasoning", ""))[:200],
                })
            else:
                results.append({"match": "no_match", "reasoning": "invalid_entry"})
        return results
    except Exception:
        return fallback


def _ai_confirm_batch_multi(groups: list[dict]) -> list[dict]:
    """Multi-candidate AI confirmation. Each group: {stub_name, candidates: [...]}.

    The AI picks the best candidate (A/B/C/D/E) or "none".
    Returns [{match, chosen_index, reasoning}, ...] in same order.
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
# Phase 1: TR search + detail (parallelized, no AI)
# ---------------------------------------------------------------------------

def _search_and_detail(stub: dict) -> dict:
    """Search TR and fetch detail for one stub. Returns enriched stub dict."""
    stub_id = stub["id"]
    stub_name = stub["name"].strip()

    variants = _search_variants(stub_name)
    results = []
    for query in variants:
        results = _scrape_tr_search(query)
        if results:
            break

    if not results:
        return {
            **stub,
            "_status": "no_results",
            "_reasoning": f"No TR results for: {variants}",
        }

    top = results[0]
    tr_id = top["tr_id"]

    # Quick similarity check — skip detail fetch for obvious non-matches
    if not _quick_similarity(stub_name, top["name"], ""):
        return {
            **stub,
            "_status": "no_similarity",
            "_tr_id": tr_id,
            "_tr_name": top["name"],
            "_reasoning": f"Search result '{top['name']}' too dissimilar to '{stub_name}'",
        }

    detail = _scrape_tr_detail(tr_id)
    if detail is None:
        return {
            **stub,
            "_status": "detail_failed",
            "_tr_id": tr_id,
            "_tr_name": top["name"],
            "_reasoning": "Could not fetch TR detail page",
        }

    return {
        **stub,
        "_status": "needs_ai",
        "_tr_id": tr_id,
        "_tr_name": detail.get("name", top["name"]),
        "_tr_acronym": detail.get("acronym", ""),
        "_tr_country": detail.get("country", ""),
        "_tr_category": detail.get("category", ""),
        "_tr_interests": detail.get("interests_represented", ""),
        "_detail": detail,
    }


def _search_and_detail_multi(stub: dict) -> dict:
    """Search TR and fetch details for ALL results (up to 5).

    Used by --retry-failed to evaluate all candidates, not just the top hit.
    """
    stub_name = stub["name"].strip()
    variants = _search_variants(stub_name)
    results = []
    for query in variants:
        results = _scrape_tr_search(query)
        if results:
            break

    if not results:
        return {**stub, "_status": "no_results", "_candidates": []}

    # Fetch all detail pages in parallel (up to 5 concurrent)
    candidates = []
    with ThreadPoolExecutor(max_workers=5) as detail_pool:
        future_to_r = {detail_pool.submit(_scrape_tr_detail, r["tr_id"]): r for r in results}
        for fut in as_completed(future_to_r):
            r = future_to_r[fut]
            try:
                detail = fut.result()
            except Exception:
                continue
            if detail:
                candidates.append({
                    "tr_id": r["tr_id"],
                    "name": detail.get("name", r["name"]),
                    "acronym": detail.get("acronym", ""),
                    "country": detail.get("country", ""),
                    "category": detail.get("category", ""),
                    "interests": detail.get("interests_represented", ""),
                    "detail": detail,
                })

    if not candidates:
        return {**stub, "_status": "detail_failed", "_candidates": []}

    return {**stub, "_status": "needs_ai", "_candidates": candidates}


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
    """Load stub IDs that are fully done (skip pending_ai — those still need Phase 2)."""
    if not REPORT_PATH.exists():
        return set()
    done = set()
    with REPORT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["confidence"] != "pending_ai":
                done.add(row["stub_id"])
    return done


def _load_pending_ai() -> list[dict]:
    """Load rows with confidence=pending_ai from the CSV for Phase 2 resume."""
    if not REPORT_PATH.exists():
        return []
    pending = []
    with REPORT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["confidence"] == "pending_ai":
                pending.append({
                    "id": row["stub_id"],
                    "name": row["stub_name"],
                    "_tr_id": row["tr_id"],
                    "_tr_name": row["tr_name"],
                    "_tr_acronym": row.get("tr_acronym", ""),
                    "_tr_country": row.get("tr_country", ""),
                    "_tr_category": row.get("tr_category", ""),
                    "_tr_interests": row.get("tr_interests", ""),
                    "_status": "needs_ai",
                })
    return pending


def _deduplicate_csv() -> None:
    """Remove stale pending_ai rows that have a newer resolved entry."""
    if not REPORT_PATH.exists():
        return
    rows = []
    with REPORT_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Keep the last (most recent) entry per stub_id — resolved rows are
    # appended after pending_ai, so they come later in the file.
    seen: dict[str, dict] = {}
    for row in rows:
        stub_id = row["stub_id"]
        prev = seen.get(stub_id)
        # Prefer resolved over pending_ai; otherwise keep the later row
        if prev is None or prev["confidence"] == "pending_ai":
            seen[stub_id] = row
    deduped = list(seen.values())
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(deduped)
    removed = len(rows) - len(deduped)
    if removed:
        print(f"  Deduplicated CSV: removed {removed} stale rows ({len(deduped)} remaining)")


def _append_row(row: dict) -> None:
    with _csv_lock:
        write_header = not REPORT_PATH.exists() or REPORT_PATH.stat().st_size == 0
        with REPORT_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _load_failed_stubs() -> list[dict]:
    """Load stubs with low or no_match confidence from the report CSV.

    Skips stubs that were already retried (confidence starts with 'retried_').
    """
    if not REPORT_PATH.exists():
        return []
    # Build set of already-retried stub IDs
    retried_ids: set[str] = set()
    candidates: list[dict] = []
    with REPORT_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["confidence"].startswith("retried_"):
                retried_ids.add(row["stub_id"])
            elif row["confidence"] in ("low", "no_match"):
                candidates.append({"id": row["stub_id"], "name": row["stub_name"]})
    return [c for c in candidates if c["id"] not in retried_ids]


def _update_csv_rows(updates: dict[str, dict]) -> None:
    """Update specific rows in the CSV by stub_id."""
    if not REPORT_PATH.exists() or not updates:
        return
    rows = []
    with REPORT_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["stub_id"] in updates:
            row.update(updates[row["stub_id"]])

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_retry_failed(args, dry_run: bool) -> None:
    """Retry low/no_match stubs with multi-candidate matching."""
    client = _get_client()

    failed_stubs = _load_failed_stubs()
    if not failed_stubs:
        print("No failed stubs (low/no_match) to retry.")
        return

    print(f"Retrying {len(failed_stubs)} failed stubs with multi-candidate matching...\n")

    # Phase 1: parallel TR search + detail for ALL candidates
    print(f"--- Phase 1: TR search + detail for all candidates ({args.workers} workers) ---")
    t0 = time.time()
    needs_ai: list[dict] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_search_and_detail_multi, s): s for s in failed_stubs}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result()
            except Exception:
                skipped += 1
                continue
            if result["_status"] == "needs_ai" and result["_candidates"]:
                needs_ai.append(result)
            else:
                skipped += 1
            if done_count % 50 == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed
                remaining = (len(failed_stubs) - done_count) / rate
                print(
                    f"  [{done_count:5d}/{len(failed_stubs)}] "
                    f"{len(needs_ai)} need AI, {skipped} skipped "
                    f"({rate:.1f}/s, ~{remaining/60:.0f}m left)"
                )

    elapsed1 = time.time() - t0
    print(
        f"  Phase 1 done in {elapsed1/60:.1f}m: "
        f"{len(needs_ai)} need AI, {skipped} skipped"
    )

    if not needs_ai:
        print("\nNo candidates to evaluate.")
        return

    # Phase 2: multi-candidate AI confirmation (parallel AI calls, incremental save)
    AI_PARALLEL = min(args.workers, 5)  # concurrent claude CLI processes
    print(f"\n--- Phase 2: Multi-candidate AI ({len(needs_ai)} orgs, batch={args.ai_batch}, {AI_PARALLEL} parallel) ---")
    t1 = time.time()
    stats = {"high": 0, "medium": 0, "low": 0, "no_match": 0, "applied": 0, "errors": 0}
    processed_count = 0

    def _process_ai_batch(batch: list[dict]) -> list[tuple[dict, dict]]:
        """Run AI on one batch and return [(result, ai_result), ...]."""
        groups = [
            {"stub_name": r["name"], "candidates": r["_candidates"]}
            for r in batch
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

            batch_updates: dict[str, dict] = {}
            for result, ai in pairs:
                idx = ai["chosen_index"]
                candidates = result["_candidates"]

                if 0 <= idx < len(candidates):
                    chosen = candidates[idx]
                else:
                    chosen = candidates[0]

                confidence = ai["match"]
                stats[confidence] = stats.get(confidence, 0) + 1

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
                        detail = chosen.get("detail")

                        if canonical_orgs and canonical_orgs[0]["id"] != result["id"]:
                            client.table("lobbying_meetings").update(
                                {"organization_id": canonical_orgs[0]["id"]}
                            ).eq("organization_id", result["id"]).execute()
                            action = "applied_relink"
                        elif detail:
                            _apply_tr_enrichment(client, result["id"], tr_id, detail, print)
                            action = "applied_enrich"
                        stats["applied"] += 1
                    except Exception as exc:
                        action = f"apply_failed: {str(exc)[:50]}"
                        stats["errors"] += 1

                batch_updates[result["id"]] = {
                    "tr_id": chosen.get("tr_id", ""),
                    "tr_name": chosen.get("name", ""),
                    "tr_acronym": chosen.get("acronym", ""),
                    "tr_country": chosen.get("country", ""),
                    "tr_category": chosen.get("category", ""),
                    "tr_interests": chosen.get("interests", ""),
                    "confidence": f"retried_{confidence}",
                    "reasoning": ai["reasoning"],
                    "action": action,
                }

            # Save after each batch so progress survives interruption
            _update_csv_rows(batch_updates)

            processed_count += len(pairs)
            print(
                f"  [{processed_count:5d}/{len(needs_ai)}] "
                f"high={stats['high']} medium={stats['medium']} "
                f"low={stats['low']} no_match={stats['no_match']}"
            )

    elapsed2 = time.time() - t1
    total_elapsed = time.time() - t0
    print(f"\n=== Retry done in {total_elapsed/60:.1f}m ===")
    print(f"  Phase 1 (search+detail): {elapsed1/60:.1f}m")
    print(f"  Phase 2 (AI):            {elapsed2/60:.1f}m")
    print(f"  Retried:   {len(failed_stubs)}")
    print(f"  With candidates: {len(needs_ai)}")
    print(f"    High:     {stats['high']}")
    print(f"    Medium:   {stats['medium']}")
    print(f"    Low:      {stats['low']}")
    print(f"    No match: {stats['no_match']}")
    print(f"    Applied:  {stats['applied']}")
    print(f"    Errors:   {stats['errors']}")
    print(f"  Report: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run org dedup Pass 4 (TR web search)")
    parser.add_argument("--apply", action="store_true", help="Apply high-confidence matches to DB")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed orgs")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry low/no_match stubs with multi-candidate matching")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--ai-batch", type=int, default=10, help="AI batch size (default: 10)")
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY RUN" if dry_run else "LIVE (will write to DB)"
    print(f"=== Org Dedup Pass 4 — {mode} ===\n")

    if args.retry_failed:
        _run_retry_failed(args, dry_run)
        return

    client = _get_client()

    print("Fetching stubs...")
    all_stubs = _fetch_stubs(client)
    print(f"  Total stubs: {len(all_stubs)}")

    print("Fetching active org IDs...")
    active_ids = _fetch_active_org_ids(client)
    stubs = [s for s in all_stubs if s["id"] in active_ids]
    print(f"  Active stubs: {len(stubs)}")

    # Pre-filter government bodies etc
    filtered = []
    prefilter_count = 0
    already_done = _load_already_processed() if args.resume else set()

    if args.resume:
        stubs = [s for s in stubs if s["id"] not in already_done]
        print(f"  After resume filter: {len(stubs)}")
    elif REPORT_PATH.exists():
        REPORT_PATH.unlink()

    for s in stubs:
        skip_reason = _should_skip(s["name"])
        if skip_reason:
            prefilter_count += 1
            _append_row({
                "stub_id": s["id"], "stub_name": s["name"],
                "tr_id": "", "tr_name": "", "tr_acronym": "",
                "confidence": "prefiltered", "reasoning": skip_reason,
                "action": "skip",
            })
        else:
            filtered.append(s)

    print(f"  Pre-filtered (govt/institutions): {prefilter_count}")
    print(f"  To search: {len(filtered)}")

    # Phase 1: parallel TR search + detail (no AI yet)
    t0 = time.time()
    needs_ai = []
    skipped_phase1 = 0
    elapsed1 = 0.0

    if filtered:
        print(f"\n--- Phase 1: TR search + detail ({args.workers} workers) ---")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_search_and_detail, s): s for s in filtered}
            done_count = 0

            for future in as_completed(futures):
                done_count += 1
                try:
                    result = future.result()
                except Exception as exc:
                    stub = futures[future]
                    _append_row({
                        "stub_id": stub["id"], "stub_name": stub["name"],
                        "tr_id": "", "tr_name": "", "tr_acronym": "",
                        "confidence": "error", "reasoning": str(exc)[:100],
                        "action": "skip",
                    })
                    skipped_phase1 += 1
                    continue

                status = result.get("_status")
                if status == "needs_ai":
                    needs_ai.append(result)
                    # Persist to CSV so Phase 2 can resume if interrupted
                    _append_row({
                        "stub_id": result["id"], "stub_name": result["name"],
                        "tr_id": result.get("_tr_id", ""),
                        "tr_name": result.get("_tr_name", ""),
                        "tr_acronym": result.get("_tr_acronym", ""),
                        "tr_country": result.get("_tr_country", ""),
                        "tr_category": result.get("_tr_category", ""),
                        "tr_interests": result.get("_tr_interests", ""),
                        "confidence": "pending_ai",
                        "reasoning": "",
                        "action": "pending",
                    })
                else:
                    _append_row({
                        "stub_id": result["id"], "stub_name": result["name"],
                        "tr_id": result.get("_tr_id", ""),
                        "tr_name": result.get("_tr_name", ""),
                        "tr_acronym": result.get("_tr_acronym", ""),
                        "confidence": status,
                        "reasoning": result.get("_reasoning", ""),
                        "action": "skip",
                    })
                    skipped_phase1 += 1

                if done_count % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed
                    remaining = (len(filtered) - done_count) / rate
                    print(
                        f"  [{done_count:5d}/{len(filtered)}] "
                        f"{len(needs_ai)} need AI, {skipped_phase1} skipped "
                        f"({rate:.1f}/s, ~{remaining/60:.0f}m left)"
                    )

        elapsed1 = time.time() - t0
        print(
            f"  Phase 1 done in {elapsed1/60:.1f}m: "
            f"{len(needs_ai)} need AI, {skipped_phase1} skipped"
        )
    else:
        print("\n  Phase 1: no new stubs to search.")

    # Phase 2: batched AI confirmation
    # On resume, reload pending_ai rows from CSV if Phase 1 produced nothing new
    if args.resume and not needs_ai:
        needs_ai = _load_pending_ai()
        if needs_ai:
            print(f"\n  Resumed {len(needs_ai)} pending_ai rows from CSV")

    if not needs_ai:
        print("\nNothing to process for Phase 2.")
        return

    print(f"\n--- Phase 2: AI confirmation ({len(needs_ai)} orgs, batch={args.ai_batch}) ---")
    t1 = time.time()
    stats = {"high": 0, "medium": 0, "low": 0, "no_match": 0, "applied": 0, "errors": 0}

    batches = [needs_ai[i:i + args.ai_batch] for i in range(0, len(needs_ai), args.ai_batch)]
    for batch_idx, batch in enumerate(batches):
        pairs = [
            {
                "stub_name": r["name"],
                "tr_name": r["_tr_name"],
                "tr_acronym": r.get("_tr_acronym", ""),
                "tr_country": r.get("_tr_country", ""),
                "tr_category": r.get("_tr_category", ""),
            }
            for r in batch
        ]

        ai_results = _ai_confirm_batch(pairs)

        for result, ai in zip(batch, ai_results):
            confidence = ai["match"]
            stats[confidence] = stats.get(confidence, 0) + 1

            if confidence == "high":
                action = "apply"
            elif confidence == "medium":
                action = "review"
            else:
                action = "skip"

            # Apply to DB if live mode and high confidence
            if not dry_run and confidence == "high" and result.get("_tr_id"):
                tr_id = result["_tr_id"]
                try:
                    canonical_resp = (
                        client.table("organizations")
                        .select("id,name")
                        .eq("eu_transparency_register_id", tr_id)
                        .execute()
                    )
                    canonical_orgs = canonical_resp.data or []
                    detail = result.get("_detail")

                    if canonical_orgs and canonical_orgs[0]["id"] != result["id"]:
                        client.table("lobbying_meetings").update(
                            {"organization_id": canonical_orgs[0]["id"]}
                        ).eq("organization_id", result["id"]).execute()
                        action = "applied_relink"
                    elif detail:
                        _apply_tr_enrichment(client, result["id"], tr_id, detail, print)
                        action = "applied_enrich"

                    stats["applied"] += 1
                except Exception as exc:
                    action = f"apply_failed: {str(exc)[:50]}"
                    stats["errors"] += 1

            _append_row({
                "stub_id": result["id"], "stub_name": result["name"],
                "tr_id": result.get("_tr_id", ""),
                "tr_name": result.get("_tr_name", ""),
                "tr_acronym": result.get("_tr_acronym", ""),
                "confidence": confidence,
                "reasoning": ai["reasoning"],
                "action": action,
            })

        processed = (batch_idx + 1) * args.ai_batch
        print(
            f"  [{min(processed, len(needs_ai)):5d}/{len(needs_ai)}] "
            f"high={stats['high']} medium={stats['medium']} "
            f"low={stats['low']} no_match={stats['no_match']}"
        )

    elapsed2 = time.time() - t1
    total_elapsed = time.time() - t0

    # Deduplicate CSV: remove stale pending_ai rows that now have resolved entries
    _deduplicate_csv()

    print(f"\n=== Done in {total_elapsed/60:.1f}m ===")
    print(f"  Phase 1 (search+detail): {elapsed1/60:.1f}m")
    print(f"  Phase 2 (AI):            {elapsed2/60:.1f}m")
    print(f"  Pre-filtered:  {prefilter_count}")
    print(f"  Skipped (ph1): {skipped_phase1}")
    print(f"  AI confirmed:  {len(needs_ai)}")
    print(f"    High:    {stats['high']}")
    print(f"    Medium:  {stats['medium']}")
    print(f"    Low:     {stats['low']}")
    print(f"    No match:{stats['no_match']}")
    print(f"    Applied: {stats['applied']}")
    print(f"    Errors:  {stats['errors']}")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
