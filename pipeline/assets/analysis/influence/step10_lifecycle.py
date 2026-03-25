"""Step 10: Lifecycle Influence Index (LII) per theme."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from ._helpers import _extract_theme_sections


def step10_lifecycle_score(
    positions: list[dict[str, Any]],
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    documents: dict[str, Any],
    proposal_alignment: dict[str, Any] | None,
    alignment: dict[str, Any],
    logger: Any = None,
) -> dict[str, Any]:
    """Compute per-theme Lifecycle Influence Index (LII).

    Part A (AI): Score lobby positions against the final adopted text.
    Part B (deterministic): Combine component rates into LII.

    LII = 0.25 * commission_reflection_rate
        + 0.35 * amendment_toward_rate
        + 0.25 * final_reflection_rate
        + 0.15 * persistence_rate

    Missing components cause weight redistribution.
    """
    _log = logger.info if logger else print

    docs = documents or {}
    adopted_text = docs.get("text_adopted", "") or ""

    # --- Part A: AI scoring of positions vs. text_adopted ---
    final_alignment: dict[str, Any] = {}
    if not adopted_text:
        _log("STEP 10: No text_adopted — skipping Part A (final text scoring).")
    elif not taxonomy:
        _log("STEP 10: No taxonomy — skipping Part A.")
    else:
        positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pos in positions:
            for t in pos.get("themes", []):
                positions_by_theme[t].append(pos)

        themes_to_score = [t for t in taxonomy if positions_by_theme.get(t)]
        if themes_to_score:
            _log(
                f"STEP 11 Part A: Scoring {len(themes_to_score)} themes against adopted text ..."
            )
            prompts: list[str] = []
            for theme_key in themes_to_score:
                adopted_excerpt = _extract_theme_sections(adopted_text, theme_key, compiled_patterns)
                theme_desc = taxonomy[theme_key].get("description", theme_key)
                pos_summaries = "\n".join(
                    f"- [{p.get('orgs', ['?'])[0]}] {p.get('summary', '')[:200]}"
                    for p in positions_by_theme[theme_key][:10]
                )
                prompts.append(
                    f"""Assess whether the final adopted text reflects the lobby positions on a policy theme.

THEME: {theme_key}
DESCRIPTION: {theme_desc}

ADOPTED TEXT EXCERPT:
{adopted_excerpt}

LOBBY POSITIONS:
{pos_summaries}

Return a JSON object:
{{
  "theme": "{theme_key}",
  "results": [
    {{
      "org": "organisation name",
      "reflection_score": "reflected" | "partially_reflected" | "not_reflected",
      "reasoning": "one sentence"
    }}
  ]
}}
Respond ONLY with valid JSON."""
                )

            raw_responses = ai_complete_parallel(
                prompts, json_mode=True, label="step10a", logger=logger
            )
            for theme_key, raw in zip(themes_to_score, raw_responses):
                parsed = parse_json_response(raw) if raw else None
                if parsed and isinstance(parsed, dict):
                    final_alignment[theme_key] = parsed.get("results", [])

    # --- Part B: Compute LII per theme ---
    lii_scores: dict[str, Any] = {}
    proposal_theme_results = (proposal_alignment or {}).get("theme_results", {})

    for theme_key in taxonomy:
        components: dict[str, float | None] = {
            "commission_reflection_rate": None,
            "amendment_toward_rate": None,
            "final_reflection_rate": None,
            "persistence_rate": None,
        }

        # commission_reflection_rate — from step 9
        prop_results = proposal_theme_results.get(theme_key)
        if isinstance(prop_results, list) and prop_results:
            scores = [r.get("reflection_score", "") for r in prop_results]
            reflected = sum(1 for s in scores if s == "reflected")
            partial = sum(1 for s in scores if s == "partially_reflected")
            total = len(scores)
            if total > 0:
                components["commission_reflection_rate"] = (reflected + 0.5 * partial) / total

        # amendment_toward_rate — from step 7 alignment
        theme_toward_total = 0
        theme_pairs_total = 0
        for mep_data in alignment.values():
            ts = mep_data.get("theme_scores", {})
            if theme_key in ts:
                entry = ts[theme_key]
                theme_toward_total += entry.get("toward", 0)
                theme_pairs_total += entry.get("pairs_evaluated", 0)
        if theme_pairs_total > 0:
            components["amendment_toward_rate"] = theme_toward_total / theme_pairs_total

        # final_reflection_rate — from Part A above
        final_results = final_alignment.get(theme_key)
        if isinstance(final_results, list) and final_results:
            scores = [r.get("reflection_score", "") for r in final_results]
            reflected = sum(1 for s in scores if s == "reflected")
            partial = sum(1 for s in scores if s == "partially_reflected")
            total = len(scores)
            if total > 0:
                components["final_reflection_rate"] = (reflected + 0.5 * partial) / total

        # persistence_rate
        theme_ams = [a for a in amendments if theme_key in (a.get("themes") or [])]
        if theme_ams:
            survived = sum(
                1 for a in theme_ams
                if (a.get("amended_text") or "").strip()
                and (a.get("amended_text") or "").strip().lower() not in ("deleted", "deletion")
            )
            components["persistence_rate"] = survived / len(theme_ams)

        # Compute LII with weight redistribution for missing components
        base_weights = {
            "commission_reflection_rate": 0.25,
            "amendment_toward_rate": 0.35,
            "final_reflection_rate": 0.25,
            "persistence_rate": 0.15,
        }
        available = {k: v for k, v in components.items() if v is not None}
        if not available:
            lii = None
        else:
            missing_weight = sum(base_weights[k] for k in components if components[k] is None)
            available_total = sum(base_weights[k] for k in available)
            if available_total == 0:
                lii = None
            else:
                lii = sum(
                    v * (base_weights[k] + base_weights[k] / available_total * missing_weight)
                    for k, v in available.items()
                )
                lii = round(min(1.0, max(0.0, lii)), 4)

        lii_scores[theme_key] = {
            "lii": lii,
            "components": {k: (round(v, 4) if v is not None else None) for k, v in components.items()},
        }

    _log(f"STEP 11 complete: LII computed for {len(lii_scores)} themes.")
    return {
        "final_text_alignment": final_alignment,
        "lii_scores": lii_scores,
    }
