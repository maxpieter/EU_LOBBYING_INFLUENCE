"""Organisation deduplication: resolve stub orgs to canonical TR entries.

Three-pass strategy:
1. TR ID extraction — stub names containing a Transparency Register ID
   (e.g. "Bundesverband deutscher Banken e.V. (Bankenverband) 0764199368-97")
   are matched deterministically to the canonical org with that TR ID.
2. Case-insensitive name matching — exact match after lowercasing and
   stripping legal suffixes (Ltd, GmbH, SA, etc.).
3. Acronym matching — stub name matches a canonical org's acronym field.

Each pass relinks lobbying_meetings.organization_id from the stub to the
canonical org.  Deterministic only — no AI, no false positives.
"""

from __future__ import annotations

import re
from typing import Any


# Legal suffixes to strip for fuzzy matching
_LEGAL_SUFFIXES = re.compile(
    r"\s+(ltd|gmbh|sa|ag|bv|nv|plc|inc|e\.v\.|aisbl|asbl|eeig|se|s\.a\.|s\.p\.a\.|s\.r\.l\.)\.?\s*$",
    re.IGNORECASE,
)

# TR ID pattern: 10+ digits followed by dash and 2 digits
_TR_ID_RE = re.compile(r"(\d{10,}-\d{2})")


def _clean_name(name: str) -> str:
    """Lowercase, strip legal suffixes and parenthetical acronyms."""
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", name).strip().lower()
    cleaned = _LEGAL_SUFFIXES.sub("", cleaned).strip()
    return cleaned


def run_org_dedup(client: Any, logger: Any = None) -> dict[str, int]:
    """Run all three dedup passes against Supabase.

    Parameters
    ----------
    client:
        Raw Supabase client.
    logger:
        Optional logger with .info() / .warning() methods.

    Returns
    -------
    Dict with counts: tr_id_relinked, name_relinked, acronym_relinked, total.
    """
    _log = logger.info if logger else print
    _err = logger.warning if logger else print

    # Fetch canonical orgs (those with a TR ID)
    real_orgs: list[dict] = []
    offset = 0
    while True:
        resp = (
            client.table("organizations")
            .select("id,name,acronym,eu_transparency_register_id")
            .not_.is_("eu_transparency_register_id", "null")
            .range(offset, offset + 999)
            .execute()
        )
        real_orgs.extend(resp.data)
        if len(resp.data) < 1000:
            break
        offset += 1000
    _log(f"Canonical orgs (with TR ID): {len(real_orgs)}")

    # Build lookups
    by_tr_id: dict[str, dict] = {}
    by_lower: dict[str, dict] = {}
    by_cleaned: dict[str, dict] = {}
    by_acronym: dict[str, dict] = {}
    _acronym_seen: dict[str, int] = {}  # track ambiguous acronyms

    for o in real_orgs:
        tr_id = o.get("eu_transparency_register_id")
        if tr_id:
            by_tr_id[tr_id] = o
        name = o["name"].strip()
        by_lower[name.lower()] = o
        by_cleaned[_clean_name(name)] = o
        if o.get("acronym"):
            acr = o["acronym"].strip().lower()
            _acronym_seen[acr] = _acronym_seen.get(acr, 0) + 1
            by_acronym[acr] = o

    # Remove ambiguous acronyms (shared by multiple orgs)
    for acr, count in _acronym_seen.items():
        if count > 1:
            by_acronym.pop(acr, None)

    # Fetch stubs (no TR ID, no normalized_name)
    stubs: list[dict] = []
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
    _log(f"Stub orgs to check: {len(stubs)}")

    stats = {"tr_id_relinked": 0, "name_relinked": 0, "acronym_relinked": 0}

    for s in stubs:
        raw = s["name"].strip()
        match = None
        method = None

        # Pass 1: TR ID embedded in name
        tr_match = _TR_ID_RE.search(raw)
        if tr_match:
            candidate = by_tr_id.get(tr_match.group(1))
            if candidate and candidate["id"] != s["id"]:
                match = candidate
                method = "tr_id_relinked"

        # Pass 2: case-insensitive name / cleaned name
        if not match:
            candidate = by_lower.get(raw.lower()) or by_cleaned.get(_clean_name(raw))
            if candidate and candidate["id"] != s["id"]:
                match = candidate
                method = "name_relinked"

        # Pass 3: acronym (only for unambiguous acronyms with 5+ chars)
        if not match:
            acronym_lower = raw.strip().lower()
            if len(acronym_lower) >= 5 and acronym_lower in by_acronym:
                # Check it's unambiguous (only one real org has this acronym)
                candidate = by_acronym[acronym_lower]
                if candidate and candidate["id"] != s["id"]:
                    match = candidate
                    method = "acronym_relinked"

        if match:
            try:
                client.table("lobbying_meetings").update(
                    {"organization_id": match["id"]}
                ).eq("organization_id", s["id"]).execute()
                stats[method] += 1
            except Exception as exc:
                _err(f"Failed to relink {s['name']}: {exc}")

    stats["total"] = sum(stats.values())
    _log(
        f"Org dedup complete: {stats['tr_id_relinked']} TR ID, "
        f"{stats['name_relinked']} name, {stats['acronym_relinked']} acronym, "
        f"{stats['total']} total relinks"
    )
    return stats
