"""Step 8: Amendment-level evidence assembly (enrichment).

Carry commission-meeting positions forward to the amendment stage via thematic
matching.  For each amendment, find positions that share themes, rank by match
count, and assemble evidence dossiers.  Purely deterministic -- no AI calls.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def step8_amendment_evidence(
    amendments: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    quant: dict[str, Any],
    logger: Any = None,
    max_dossiers: int = 50,
) -> list[dict[str, Any]]:
    """Assemble amendment-level evidence dossiers via thematic enrichment.

    For each amendment, finds positions from commission meetings that share at
    least one theme.  Returns the top *max_dossiers* amendments ranked by
    matching-position count (descending).
    """
    _log = logger.info if logger else print

    if not amendments or not positions:
        _log("STEP 8: No amendments or positions — skipping.")
        return []

    _log(f"STEP 8: Enriching {len(amendments)} amendments with {len(positions)} positions ...")

    # Index positions by theme
    positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            positions_by_theme[t].append(pos)

    # Key MEPs lookup for author details
    key_meps = quant.get("key_meps", {})

    dossiers: list[dict[str, Any]] = []
    for am in amendments:
        am_themes = set(am.get("themes", []))
        if not am_themes:
            continue

        # Find all positions sharing at least one theme, deduplicate by meeting_id
        seen_ids: set[str] = set()
        matching: list[dict[str, Any]] = []
        for t in am_themes:
            for pos in positions_by_theme.get(t, []):
                mid = pos.get("meeting_id", "")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                shared = sorted(am_themes & set(pos.get("themes", [])))
                matching.append(
                    {
                        "org": (pos.get("orgs") or ["Unknown"])[0],
                        "summary": pos.get("summary", ""),
                        "date": pos.get("date", ""),
                        "commissioner": pos.get("commissioner", ""),
                        "shared_themes": shared,
                    }
                )

        if not matching:
            continue

        # Resolve author details from key_meps
        authors = am.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        author_details = {}
        for author in authors:
            for mep_name, meta in key_meps.items():
                if mep_name.lower() in author.lower() or (
                    mep_name.split()[-1].lower() in author.lower() if mep_name.split() else False
                ):
                    author_details[mep_name] = {
                        "role": meta.get("role", ""),
                        "party": meta.get("party", ""),
                    }

        dossiers.append(
            {
                "amendment_number": am.get("number"),
                "source": am.get("source", ""),
                "location": am.get("location", ""),
                "amendment_text": (am.get("body") or "")[:800],
                "themes": sorted(am_themes),
                "authors": authors,
                "author_details": author_details,
                "matching_position_count": len(matching),
                "matching_positions": matching,
            }
        )

    # Sort by match count descending, keep top N
    dossiers.sort(key=lambda d: d["matching_position_count"], reverse=True)
    dossiers = dossiers[:max_dossiers]

    _log(
        f"STEP 8 complete: {len(dossiers)} amendment evidence dossiers assembled "
        f"(top by matching position count)."
    )
    return dossiers
