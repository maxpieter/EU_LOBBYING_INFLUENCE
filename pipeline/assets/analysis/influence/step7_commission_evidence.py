"""Step 7: Commission-level evidence assembly.

For top themes by lobbying density, extract the relevant legislation section,
AI-summarise what it proposes, and collect all position summaries from
organisations that lobbied on that theme.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._ai import ai_complete_parallel
from ._helpers import _extract_theme_sections


def step7_commission_evidence(
    taxonomy: dict[str, Any],
    positions: list[dict[str, Any]],
    compiled_patterns: dict[str, list],
    documents: dict[str, Any],
    theme_lobbying_density: list[dict[str, Any]],
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Assemble commission-level evidence dossiers per theme.

    Returns a list of dossiers sorted by commission meeting count (descending).
    Each dossier contains the legislation excerpt, an AI summary of what that
    section proposes, and all position summaries from organisations that lobbied
    on the theme.
    """
    _log = logger.info if logger else print

    proposal_text = (documents or {}).get("commission_proposal", "") or ""
    if not proposal_text:
        _log("STEP 7: No commission proposal text available — skipping.")
        return []

    if not taxonomy or not theme_lobbying_density:
        _log("STEP 7: No taxonomy or density data — skipping.")
        return []

    # Only themes that had at least one commission meeting
    active_themes = [
        t for t in theme_lobbying_density
        if t["commission_meeting_count"] > 0
    ]
    if not active_themes:
        _log("STEP 7: No themes with commission meetings — skipping.")
        return []

    _log(f"STEP 7: Assembling commission evidence for {len(active_themes)} themes ...")

    # Group positions by theme
    positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            positions_by_theme[t].append(pos)

    # Extract legislation excerpts and build AI prompts for summaries
    theme_keys: list[str] = []
    excerpts: dict[str, str] = {}
    prompts: list[str] = []

    for entry in active_themes:
        theme_key = entry["theme"]
        excerpt = _extract_theme_sections(proposal_text, theme_key, compiled_patterns)
        if not excerpt.strip():
            continue
        theme_keys.append(theme_key)
        excerpts[theme_key] = excerpt
        prompts.append(
            f"""Summarise what the following section of EU legislation proposes to do, in one paragraph (3-5 sentences). Focus on the concrete rules and requirements, not background.

THEME: {theme_key} -- {taxonomy.get(theme_key, {}).get("description", "")}

LEGISLATION TEXT:
{excerpt}

Respond with a single paragraph, no JSON."""
        )

    # Fire AI summaries in parallel
    summaries: dict[str, str] = {}
    if prompts:
        raw_responses = ai_complete_parallel(
            prompts, json_mode=False, label="step7_summaries", logger=logger
        )
        for theme_key, raw in zip(theme_keys, raw_responses):
            summaries[theme_key] = (raw or "").strip() or "Summary not available."

    # Assemble dossiers
    dossiers: list[dict[str, Any]] = []
    for entry in active_themes:
        theme_key = entry["theme"]
        if theme_key not in excerpts:
            continue

        theme_positions = positions_by_theme.get(theme_key, [])
        position_records = [
            {
                "org": (p.get("orgs") or ["Unknown"])[0],
                "summary": p.get("summary", ""),
                "date": p.get("date", ""),
                "commissioner": p.get("commissioner", ""),
            }
            for p in theme_positions
        ]

        dossiers.append(
            {
                "theme": theme_key,
                "theme_description": taxonomy.get(theme_key, {}).get("description", ""),
                "commission_meeting_count": entry["commission_meeting_count"],
                "legislation_excerpt": excerpts[theme_key],
                "legislation_summary": summaries.get(theme_key, "Summary not available."),
                "positions": position_records,
            }
        )

    _log(f"STEP 7 complete: {len(dossiers)} commission evidence dossiers assembled.")
    return dossiers
