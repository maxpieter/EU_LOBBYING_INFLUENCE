"""Fuzzy organisation matching: rapidfuzz + Anthropic API.

Consolidates:
- lobbying/fuzzy_match.py (pg_trgm + Claude CLI subprocess)
- lobbying/org_dedup.py pass 4 (TR web search + AI)
- scripts/run_org_dedup_pass4.py (rapidfuzz + Anthropic API)

Uses the organizations.match_method column in Supabase as the persistent
ledger. Stubs with any match_method value are never re-sent to AI.
Only stubs with match_method IS NULL hit the API.

match_method vocabulary mirrors meeting_procedure_links.match_method:
    prefiltered, fuzzy_auto_accept, ai_high, no_match.
For fuzzy_auto_accept and ai_high, matched_tr_id is also written so the
matcher's selection is queryable independently of the relink/enrich
choice in Phase 2 of definitions.py.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline.models.lobbying_models import Organization

from .resolution import OrgResolver, normalize_for_key, search_variants

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TR_CACHE_PATH = _PROJECT_ROOT / "analysis" / "tr_dump.json"

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
    """Return skip reason if this org name should be excluded from fuzzy matching."""
    m = _SKIP_PATTERNS.search(name)
    return f"Pre-filtered: matches '{m.group(0).strip()}'" if m else None


# ---------------------------------------------------------------------------
# TR dump download + local fuzzy matching
# ---------------------------------------------------------------------------

_TR_XML_URL = "https://transparency-register.europa.eu/odplastorganisationxml_en"


def download_tr_dump() -> list[dict]:
    """Download and parse the full TR XML dump. Caches as JSON for 24h."""
    import requests

    if TR_CACHE_PATH.exists():
        age_hours = (time.time() - TR_CACHE_PATH.stat().st_mtime) / 3600
        if age_hours < 24:
            with TR_CACHE_PATH.open(encoding="utf-8") as f:
                return json.load(f)

    resp = requests.get(_TR_XML_URL, timeout=120)
    resp.raise_for_status()
    raw = resp.text
    raw = re.sub(r"&#x[0-1]?[0-9a-fA-F];", "", raw)

    import xml.etree.ElementTree as ET
    root = ET.fromstring(raw)

    orgs = []
    for ir in root.findall(".//interestRepresentative"):
        name = ir.findtext("name/originalName", "").strip()
        if not name:
            continue
        orgs.append({
            "tr_id": ir.findtext("identificationCode", "").strip(),
            "name": name,
            "acronym": ir.findtext("acronym", "").strip(),
            "category": ir.findtext("registrationCategory", "").strip(),
            "country": (ir.find("headOffice").findtext("country", "").strip().title()
                        if ir.find("headOffice") is not None else ""),
            "interests": ir.findtext("goals", "").strip()[:500],
            "website": ir.findtext("webSiteURL", "").strip(),
        })

    TR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TR_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(orgs, f, ensure_ascii=False)

    return orgs


def _normalize_for_fuzzy(s: str) -> str:
    """Normalize for fuzzy comparison (keeps spaces for token matching)."""
    from .resolution import _LEGAL_SUFFIXES
    s = s.lower().strip()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_tr_lookup(tr_orgs: list[dict]) -> dict:
    """Build lookup structures from TR dump for fuzzy matching."""
    by_name: dict[str, list[dict]] = {}
    by_acronym: dict[str, list[dict]] = {}

    for org in tr_orgs:
        norm = _normalize_for_fuzzy(org["name"])
        by_name.setdefault(norm, []).append(org)
        if org["acronym"]:
            acr = org["acronym"].lower().strip()
            by_acronym.setdefault(acr, []).append(org)

    return {
        "by_name": by_name,
        "by_acronym": by_acronym,
        "all_names": list(by_name.keys()),
    }


def fuzzy_match_local(
    stub_name: str,
    lookup: dict,
    top_n: int = 3,
    min_score: int = 50,
) -> list[dict]:
    """Local rapidfuzz matching against TR dump.

    Returns list of {tr_org, score} dicts, highest score first.
    """
    from rapidfuzz import fuzz, process

    norm = _normalize_for_fuzzy(stub_name)

    # Priority 1: exact normalized match
    if norm in lookup["by_name"]:
        return [{"tr_org": org, "score": 100} for org in lookup["by_name"][norm][:top_n]]

    # Priority 2: acronym match
    acr = norm.upper().replace(" ", "")
    if len(acr) <= 10 and acr.lower() in lookup["by_acronym"]:
        return [{"tr_org": org, "score": 95} for org in lookup["by_acronym"][acr.lower()][:top_n]]

    # Priority 3: fuzzy match
    results = process.extract(norm, lookup["all_names"], scorer=fuzz.WRatio, limit=top_n)
    matches = []
    for match_name, score, _ in results:
        if score < min_score:
            continue
        for org in lookup["by_name"][match_name]:
            matches.append({"tr_org": org, "score": score})
    return matches[:top_n]


# ---------------------------------------------------------------------------
# Supabase match_method helpers
# ---------------------------------------------------------------------------

def _set_match_method(
    supabase_client: Any,
    stub_ids: list[str],
    method: str,
) -> None:
    """Batch-update match_method for a list of org IDs.

    Uses 50-id chunks — proven safe size against Supabase's writer
    statement_timeout (matches pipeline/assets/procedures/matching.py:765).
    """
    if not stub_ids:
        return
    for i in range(0, len(stub_ids), 50):
        batch = stub_ids[i:i + 50]
        try:
            supabase_client.table("organizations").update(
                {"match_method": method}
            ).in_("id", batch).execute()
        except Exception:
            for sid in batch:
                try:
                    supabase_client.table("organizations").update(
                        {"match_method": method}
                    ).eq("id", sid).execute()
                except Exception:
                    pass


def _set_matched_tr_ids(
    supabase_client: Any,
    pairs: list[tuple[str, str]],
) -> None:
    """Per-row PATCH writing matched_tr_id. One UPDATE per pair (small)."""
    for sid, tr_id in pairs:
        try:
            supabase_client.table("organizations").update(
                {"matched_tr_id": tr_id}
            ).eq("id", sid).execute()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI confirmation via Anthropic SDK
# ---------------------------------------------------------------------------

def _build_ai_prompt(batch: list[dict]) -> str:
    """Build classification prompt for a batch of stub->candidate groups."""
    items = []
    for i, row in enumerate(batch):
        cand_lines = []
        for j, c in enumerate(row["candidates"]):
            cand_lines.append(
                f'    {chr(65+j)}. "{c["name"]}" '
                f'(acronym: "{c.get("acronym", "")}", '
                f'country: "{c.get("country", "")}", '
                f'category: "{c.get("category", "")}")'
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
        '[{"match": "high"|"no_match", "chosen": "A"|"B"|"C"|"none", '
        '"reasoning": "one sentence"}, ...]\n'
        "Only use 'high'. Do not use 'medium' or 'low' — if unsure, return no_match.\n"
        f"IMPORTANT: Return exactly {len(batch)} entries in order."
    )


def ai_confirm_batch(
    groups: list[dict],
    anthropic_client: Any,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Classify a batch of stub->candidate groups via Anthropic API.

    Each group: {stub_name, stub_id, candidates: [{name, acronym, country, category, ...}]}
    Returns [{match, chosen_index, reasoning}, ...] in same order.
    """
    if not groups:
        return []

    prompt = _build_ai_prompt(groups)
    fallback = [{"match": "no_match", "chosen_index": -1, "reasoning": "batch_failed"}] * len(groups)

    for attempt in range(5):
        try:
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                time.sleep(2 ** attempt * 3)
                continue
            return fallback
    else:
        return fallback

    raw = response.content[0].text
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        return fallback
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return fallback
    if not isinstance(parsed, list) or len(parsed) != len(groups):
        return fallback

    # Mirror procedure matcher (matching.py:458): collapse non-"high" to no_match.
    # The prompt only authorises 'high' or 'no_match', but old/strict-mode
    # responses may still include 'medium' or 'low' — treat those as no_match.
    out = []
    for entry in parsed:
        if not isinstance(entry, dict):
            out.append({"match": "no_match", "chosen_index": -1, "reasoning": "invalid_entry"})
            continue
        match_val = "high" if entry.get("match") == "high" else "no_match"
        chosen_letter = str(entry.get("chosen", "none")).upper()
        chosen_index = (
            ord(chosen_letter) - 65
            if len(chosen_letter) == 1 and chosen_letter.isalpha() and chosen_letter in "ABC"
            else -1
        )
        out.append({
            "match": match_val,
            "chosen_index": chosen_index,
            "reasoning": str(entry.get("reasoning", ""))[:200],
        })
    return out


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def resolve_stubs(
    stubs: list[Organization],
    tr_lookup: dict,
    supabase_client: Any,
    logger: Any = None,
    anthropic_client: Any = None,
    auto_accept_threshold: int = 96,
    min_score: int = 50,
    ai_batch_size: int = 50,
    workers: int = 5,
    dry_run: bool = True,
) -> dict[str, str]:
    """Resolve stub orgs via fuzzy matching + AI.

    Returns {stub_id: canonical_tr_id} for high-confidence matches.
    Writes match_method (and matched_tr_id where applicable) to Supabase.
    Only processes stubs where match_method IS NULL.

    match_method values (parallels meeting_procedure_links.match_method):
        prefiltered          — _should_skip pattern hit
        fuzzy_auto_accept    — fuzzy score >= auto_accept_threshold
        ai_high              — AI returned "high"
        no_match             — no fuzzy candidates OR AI returned non-high
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    # Filter to stubs that haven't been classified yet
    new_stubs = [s for s in stubs if not s.match_method]
    already_count = len(stubs) - len(new_stubs)
    _log(f"Stubs: {len(new_stubs)} new, {already_count} already classified (skipped)")

    need_ai: list[dict] = []
    mappings: dict[str, str] = {}  # stub_id -> canonical_tr_id

    # Harvest existing mappings from already-classified stubs
    for s in stubs:
        if s.match_method in ("ai_high", "fuzzy_auto_accept") and s.matched_tr_id:
            mappings[s.id] = s.matched_tr_id

    # Track status updates to batch-write at end
    status_updates: dict[str, list[str]] = {}  # match_method -> [stub_ids]
    tr_id_writes: list[tuple[str, str]] = []   # (stub_id, tr_id) for matched_tr_id PATCHes

    for stub in new_stubs:
        skip_reason = _should_skip(stub.name)
        if skip_reason:
            status_updates.setdefault("prefiltered", []).append(stub.id)
            continue

        matches = fuzzy_match_local(stub.name, tr_lookup, min_score=min_score)
        best_score = matches[0]["score"] if matches else 0

        if not matches or best_score < min_score:
            status_updates.setdefault("no_match", []).append(stub.id)
        elif best_score >= auto_accept_threshold:
            chosen = matches[0]["tr_org"]
            tr_id = chosen.get("tr_id", "")
            status_updates.setdefault("fuzzy_auto_accept", []).append(stub.id)
            if tr_id:
                mappings[stub.id] = tr_id
                tr_id_writes.append((stub.id, tr_id))
        else:
            need_ai.append({
                "stub_id": stub.id,
                "stub_name": stub.name,
                "candidates": [
                    {
                        "name": m["tr_org"].get("name", ""),
                        "tr_id": m["tr_org"].get("tr_id", ""),
                        "acronym": m["tr_org"].get("acronym", ""),
                        "country": m["tr_org"].get("country", ""),
                        "category": m["tr_org"].get("category", ""),
                        "interests": m["tr_org"].get("interests", "")[:200],
                        "score": m["score"],
                    }
                    for m in matches[:3]
                ],
            })

    fuzzy_high = len(status_updates.get("fuzzy_auto_accept", []))
    _log(
        f"Fuzzy phase: {fuzzy_high} auto-accepted, "
        f"{len(status_updates.get('no_match', []))} no match, "
        f"{len(status_updates.get('prefiltered', []))} prefiltered, "
        f"{len(need_ai)} need AI"
    )

    # AI classification
    if need_ai and anthropic_client is not None:
        batches = [need_ai[i:i + ai_batch_size] for i in range(0, len(need_ai), ai_batch_size)]
        _log(f"AI classification: {len(need_ai)} stubs in {len(batches)} batches")

        def _classify(bi_batch: tuple[int, list[dict]]) -> tuple[int, list[dict], list[dict]]:
            bi, batch = bi_batch
            results = ai_confirm_batch(batch, anthropic_client)
            return bi, batch, results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_classify, (bi, b)): bi for bi, b in enumerate(batches)}
            for fut in as_completed(futures):
                try:
                    _, batch, ai_results = fut.result()
                except Exception as e:
                    _warn(f"AI batch failed: {e}")
                    continue

                for group, ai in zip(batch, ai_results):
                    confidence = ai["match"]   # "high" or "no_match" only
                    idx = ai["chosen_index"]
                    chosen = group["candidates"][idx] if 0 <= idx < len(group["candidates"]) else {}

                    method = "ai_high" if confidence == "high" else "no_match"
                    status_updates.setdefault(method, []).append(group["stub_id"])

                    if confidence == "high" and chosen.get("tr_id"):
                        mappings[group["stub_id"]] = chosen["tr_id"]
                        tr_id_writes.append((group["stub_id"], chosen["tr_id"]))

    elif need_ai:
        _warn(f"Skipping AI classification for {len(need_ai)} stubs (no anthropic_client)")

    # Write match_method and matched_tr_id to Supabase
    if not dry_run:
        for method, ids in status_updates.items():
            _set_match_method(supabase_client, ids, method)
            _log(f"  Set match_method='{method}' on {len(ids)} orgs")
        if tr_id_writes:
            _set_matched_tr_ids(supabase_client, tr_id_writes)
            _log(f"  Wrote matched_tr_id on {len(tr_id_writes)} orgs")
    else:
        for method, ids in status_updates.items():
            _log(f"  [dry-run] Would set match_method='{method}' on {len(ids)} orgs")
        _log(f"  [dry-run] Would write matched_tr_id on {len(tr_id_writes)} orgs")

    total_new = sum(len(ids) for ids in status_updates.values())
    auto = len(status_updates.get("fuzzy_auto_accept", []))
    aih = len(status_updates.get("ai_high", []))
    _log(
        f"Fuzzy resolution complete: {total_new} classified, "
        f"{auto} fuzzy_auto_accept, {aih} ai_high, "
        f"{len(mappings)} total mappings"
    )

    return mappings


def backfill_unmatched_junction_rows(
    supabase_client: Any,
    resolver: OrgResolver,
    tr_lookup: dict,
    logger: Any = None,
    anthropic_client: Any = None,
) -> int:
    """Backfill commission_meeting_organizations WHERE organization_id IS NULL.

    1. Run each organization_name through OrgResolver (deterministic, free)
    2. Check match_method on matching stub orgs
    3. Only truly new names go through fuzzy + AI
    4. UPDATE organization_id in Supabase

    Returns count of rows backfilled.
    """
    _log = logger.info if logger else print

    # Fetch unmatched junction rows
    rows: list[dict] = []
    offset = 0
    while True:
        resp = (
            supabase_client.table("commission_meeting_organizations")
            .select("id,organization_name,meeting_id")
            .is_("organization_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        rows.extend(resp.data or [])
        if len(resp.data or []) < 1000:
            break
        offset += 1000

    if not rows:
        _log("Backfill: no unmatched junction rows found")
        return 0

    _log(f"Backfill: {len(rows)} unmatched junction rows to process")

    backfilled = 0
    still_unmatched: list[dict] = []

    for row in rows:
        name = (row.get("organization_name") or "").strip()
        if not name:
            continue

        # Step 1: deterministic resolution
        org, method = resolver.resolve(name)
        if method != "stub":
            try:
                supabase_client.table("commission_meeting_organizations").update(
                    {"organization_id": org.id}
                ).eq("id", row["id"]).execute()
                backfilled += 1
            except Exception:
                pass
            continue

        # Step 2: stub already matched (fuzzy_auto_accept or ai_high) with a TR ID
        if org.match_method in ("fuzzy_auto_accept", "ai_high") and org.matched_tr_id:
            try:
                supabase_client.table("commission_meeting_organizations").update(
                    {"organization_id": org.id}
                ).eq("id", row["id"]).execute()
                backfilled += 1
            except Exception:
                pass
            continue

        # Skip stubs already classified as no_match/prefiltered
        if org.match_method in ("no_match", "prefiltered"):
            continue

        # Step 3: truly new — collect for fuzzy
        still_unmatched.append(row)

    # Run fuzzy on truly new names
    if still_unmatched:
        _log(f"Backfill: {len(still_unmatched)} truly new names, running fuzzy matching")
        fake_stubs = [
            Organization(name=r["organization_name"], normalized_name=r["organization_name"])
            for r in still_unmatched
        ]
        mappings = resolve_stubs(
            fake_stubs, tr_lookup,
            supabase_client=supabase_client,
            logger=logger,
            anthropic_client=anthropic_client,
            dry_run=False,
        )
        for row in still_unmatched:
            name = row["organization_name"]
            stub_id = _generate_stub_id(name)
            canonical_tr_id = mappings.get(stub_id)
            if canonical_tr_id:
                try:
                    supabase_client.table("commission_meeting_organizations").update(
                        {"organization_id": canonical_tr_id}
                    ).eq("id", row["id"]).execute()
                    backfilled += 1
                except Exception:
                    pass

    _log(f"Backfill complete: {backfilled} junction rows updated")
    return backfilled


def _generate_stub_id(name: str) -> str:
    """Import from resolution module."""
    from .resolution import generate_stub_id
    return generate_stub_id(name)
