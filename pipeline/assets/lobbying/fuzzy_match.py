"""Fuzzy org matching at ingestion time.

Before uploading new stub organizations to Supabase, check them against
existing canonical orgs using pg_trgm similarity.  High-similarity candidates
are confirmed by Claude Haiku (via ``claude -p`` subprocess / OAuth).
Confirmed matches get their meetings remapped to the canonical org so the
stub is never inserted.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from pipeline.models.lobbying_models import LobbyingMeeting, Organization

from .org_dedup import _search_variants


def _find_fuzzy_candidates(
    supabase_resource: Any,
    stub_names: list[str],
    similarity_threshold: float = 0.25,
    max_results: int = 3,
) -> dict[str, list[dict]]:
    """Query Supabase for canonical orgs similar to each stub name.

    Tries multiple name variants (original, geo-stripped, no-parens) and
    keeps the best candidates across all variants.

    Returns {stub_name: [candidate_rows]} where each candidate has keys:
    id, name, eu_transparency_register_id, acronym, country,
    organization_type, interests_represented, similarity_score.
    """
    results: dict[str, list[dict]] = {}
    for name in stub_names:
        best: dict[str, dict] = {}  # candidate_id -> best row
        for variant in _search_variants(name):
            try:
                resp = supabase_resource.rpc(
                    "match_org_by_similarity",
                    {
                        "query_name": variant,
                        "similarity_threshold": similarity_threshold,
                        "max_results": max_results,
                    },
                )
                for row in (resp.data or []):
                    cid = row["id"]
                    if cid not in best or row["similarity_score"] > best[cid]["similarity_score"]:
                        best[cid] = row
            except Exception:
                pass
        if best:
            # Sort by similarity descending
            results[name] = sorted(best.values(), key=lambda r: r["similarity_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Batched AI confirmation (reused pattern from run_org_dedup_pass4.py)
# ---------------------------------------------------------------------------

def _ai_confirm_batch(groups: list[dict]) -> list[dict]:
    """Confirm multiple stub→candidates groups in a single Haiku call.

    Each group: {stub_name, candidates: [{name, acronym, country, org_type}, ...]}.
    Returns [{match, chosen_index, reasoning}, ...] in same order.
    chosen_index is 0-based index into candidates, or -1 for no match.
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
                f'type: "{c.get("org_type", "")}")'
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
        '[{"match": "high"|"medium"|"low"|"no_match", "chosen": "A"|"B"|"C"|"none", '
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
                # Convert letter choice to index
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
# Main entry point — called from diamond.py before upload
# ---------------------------------------------------------------------------

def resolve_stubs(
    organizations: list[Organization],
    meetings: list[LobbyingMeeting],
    supabase_resource: Any,
    logger: Any = None,
    similarity_threshold: float = 0.25,
    ai_batch_size: int = 10,
) -> tuple[list[Organization], list[LobbyingMeeting]]:
    """Resolve stub orgs to existing canonical orgs via fuzzy matching + AI.

    Mutates meeting.organization_id in-place for confirmed matches and removes
    matched stubs from the organizations list.

    Returns (filtered_organizations, meetings).
    """
    _log = logger.info if logger else print
    _warn = logger.warning if logger else print

    # Partition into canonical vs stubs
    stubs = [o for o in organizations if not o.eu_transparency_register_id]
    if not stubs:
        return organizations, meetings

    _log(f"Fuzzy match: checking {len(stubs)} stubs against existing canonical orgs")

    # Step 1: pg_trgm similarity search
    stub_names = [s.name for s in stubs]
    candidates = _find_fuzzy_candidates(
        supabase_resource, stub_names, similarity_threshold
    )

    if not candidates:
        _log("Fuzzy match: no candidates found above similarity threshold")
        return organizations, meetings

    _log(f"Fuzzy match: {len(candidates)} stubs have similarity candidates")

    # Build groups for AI confirmation — all candidates per stub
    stub_by_name: dict[str, Organization] = {s.name: s for s in stubs}
    groups_to_confirm: list[dict] = []

    for stub_name, cands in candidates.items():
        groups_to_confirm.append({
            "stub_name": stub_name,
            "stub_id": stub_by_name[stub_name].id,
            "candidates": [
                {
                    "name": c["name"],
                    "id": c["id"],
                    "acronym": c.get("acronym") or "",
                    "country": c.get("country") or "",
                    "org_type": c.get("organization_type") or "",
                    "similarity": c.get("similarity_score", 0),
                }
                for c in cands
            ],
        })

    # Step 2: batched AI confirmation
    confirmed: dict[str, str] = {}  # stub_id -> canonical_id
    batches = [
        groups_to_confirm[i:i + ai_batch_size]
        for i in range(0, len(groups_to_confirm), ai_batch_size)
    ]

    for batch in batches:
        ai_results = _ai_confirm_batch(batch)
        for group, ai in zip(batch, ai_results):
            idx = ai["chosen_index"]
            if ai["match"] == "high" and 0 <= idx < len(group["candidates"]):
                chosen = group["candidates"][idx]
                confirmed[group["stub_id"]] = chosen["id"]
                _log(
                    f"  Fuzzy matched: '{group['stub_name']}' -> "
                    f"'{chosen['name']}' "
                    f"(sim={chosen['similarity']:.2f}, {ai['reasoning']})"
                )

    if not confirmed:
        _log("Fuzzy match: no high-confidence matches confirmed by AI")
        return organizations, meetings

    # Step 3: remap meetings and filter out matched stubs
    remapped = 0
    for m in meetings:
        if m.organization_id in confirmed:
            m.organization_id = confirmed[m.organization_id]
            remapped += 1

    filtered_orgs = [o for o in organizations if o.id not in confirmed]

    _log(
        f"Fuzzy match complete: {len(confirmed)} stubs resolved, "
        f"{remapped} meetings remapped, "
        f"{len(stubs) - len(confirmed)} stubs remain"
    )

    return filtered_orgs, meetings
