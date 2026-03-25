"""Step 8: Proposal alignment — lobby positions vs commission proposal text."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from ._helpers import _extract_theme_sections


def step8_proposal_alignment(
    positions: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    documents: dict[str, Any],
    logger: Any = None,
) -> dict[str, Any]:
    """Compare lobby positions against the commission proposal text.

    Per theme, ask the AI whether the proposal already reflects each position.
    Returns a dict with ``theme_results`` and ``aggregate`` counts.
    """
    _log = logger.info if logger else print

    proposal_text = (documents or {}).get("commission_proposal", "") or ""
    if not proposal_text:
        _log("STEP 8: No commission proposal text available — skipping.")
        return {"skipped": True, "reason": "no_commission_proposal_text"}

    if not taxonomy:
        _log("STEP 8: No taxonomy — skipping proposal alignment.")
        return {"skipped": True, "reason": "no_taxonomy"}

    _log("STEP 8: Scoring lobby positions against commission proposal ...")

    # Group positions by theme
    positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            positions_by_theme[t].append(pos)

    themes_to_score = [t for t in taxonomy if positions_by_theme.get(t)]
    if not themes_to_score:
        _log("STEP 8: No positions with known themes — skipping.")
        return {"skipped": True, "reason": "no_positions_with_themes"}

    prompts: list[str] = []
    for theme_key in themes_to_score:
        proposal_excerpt = _extract_theme_sections(proposal_text, theme_key, compiled_patterns)
        theme_desc = taxonomy[theme_key].get("description", theme_key)
        pos_summaries = "\n".join(
            f"- [{p.get('orgs', ['?'])[0]}] {p.get('summary', '')[:200]}"
            for p in positions_by_theme[theme_key][:10]
        )
        prompts.append(
            f"""You are analysing whether the Commission's legislative proposal already reflects the positions of lobbyists on a specific policy theme.

THEME: {theme_key}
DESCRIPTION: {theme_desc}

COMMISSION PROPOSAL EXCERPT (relevant to this theme):
{proposal_excerpt}

LOBBY POSITIONS ON THIS THEME:
{pos_summaries}

For each lobby position listed, assess whether the Commission proposal text already reflects it.
Return a JSON object:
{{
  "theme": "{theme_key}",
  "results": [
    {{
      "org": "organisation name",
      "position_summary": "one sentence",
      "reflection_score": "reflected" | "partially_reflected" | "not_reflected",
      "reasoning": "one sentence"
    }}
  ]
}}
Respond ONLY with valid JSON."""
        )

    raw_responses = ai_complete_parallel(
        prompts, json_mode=True, label="step8", logger=logger
    )

    theme_results: dict[str, Any] = {}
    agg = {"total": 0, "reflected": 0, "partially_reflected": 0, "not_reflected": 0}

    for theme_key, raw in zip(themes_to_score, raw_responses):
        parsed = parse_json_response(raw) if raw else None
        if not parsed or not isinstance(parsed, dict):
            theme_results[theme_key] = {"error": "parse_failed", "raw": (raw or "")[:200]}
            continue
        results = parsed.get("results", [])
        theme_results[theme_key] = results
        for entry in results:
            score = entry.get("reflection_score", "")
            agg["total"] += 1
            if score in agg:
                agg[score] += 1

    _log(
        f"STEP 9 complete: {agg['total']} positions scored — "
        f"reflected={agg['reflected']} partially={agg['partially_reflected']} "
        f"not={agg['not_reflected']}"
    )
    return {"theme_results": theme_results, "aggregate": agg}
