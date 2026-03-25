"""Step 9: Text evolution — track how provisions changed across document stages."""

from __future__ import annotations

import re
from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from ._helpers import _extract_theme_sections


def step9_text_evolution(
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list[re.Pattern]],
    documents: dict[str, Any],
    logger: Any = None,
) -> dict[str, Any]:
    """Compare document stages per theme: proposal -> committee report -> text adopted.

    Returns ``theme_evolution`` dict and ``aggregate`` counts.
    """
    _log = logger.info if logger else print

    docs = documents or {}
    proposal_text = docs.get("commission_proposal", "") or ""
    committee_text = docs.get("committee_report", "") or ""
    adopted_text = docs.get("text_adopted", "") or ""

    available_stages: list[tuple[str, str]] = []
    if proposal_text:
        available_stages.append(("Commission Proposal", proposal_text))
    if committee_text:
        available_stages.append(("Committee Report", committee_text))
    if adopted_text:
        available_stages.append(("Text Adopted", adopted_text))

    if len(available_stages) < 2:
        _log("STEP 9: Fewer than 2 document stages available — skipping text evolution.")
        return {"skipped": True, "reason": "insufficient_document_stages"}

    if not taxonomy:
        _log("STEP 9: No taxonomy — skipping text evolution.")
        return {"skipped": True, "reason": "no_taxonomy"}

    _log(
        f"STEP 9: Analysing text evolution across {len(available_stages)} stages "
        f"for {len(taxonomy)} themes ..."
    )

    prompts: list[str] = []
    theme_keys: list[str] = list(taxonomy.keys())
    for theme_key in theme_keys:
        theme_desc = taxonomy[theme_key].get("description", theme_key)
        stage_texts = "\n\n".join(
            f"=== {label} ===\n{_extract_theme_sections(txt, theme_key, compiled_patterns)}"
            for label, txt in available_stages
        )
        prompts.append(
            f"""Analyse how provisions related to a specific policy theme changed across legislative document stages.

THEME: {theme_key}
DESCRIPTION: {theme_desc}

DOCUMENT STAGES:
{stage_texts}

Assess how the provisions relevant to this theme changed from the earliest to the latest stage.
Return a JSON object:
{{
  "theme": "{theme_key}",
  "evolution_score": "strengthened" | "weakened" | "modified" | "unchanged",
  "summary": "two sentences describing what changed and how",
  "stage_notes": [{{"stage": "...", "note": "..."}}]
}}
Respond ONLY with valid JSON."""
        )

    raw_responses = ai_complete_parallel(
        prompts, json_mode=True, label="step9", logger=logger
    )

    theme_evolution: dict[str, Any] = {}
    agg = {"strengthened": 0, "weakened": 0, "modified": 0, "unchanged": 0, "error": 0}

    for theme_key, raw in zip(theme_keys, raw_responses):
        parsed = parse_json_response(raw) if raw else None
        if not parsed or not isinstance(parsed, dict):
            theme_evolution[theme_key] = {"error": "parse_failed", "raw": (raw or "")[:200]}
            agg["error"] += 1
            continue
        theme_evolution[theme_key] = parsed
        score = parsed.get("evolution_score", "")
        if score in agg:
            agg[score] += 1
        else:
            agg["error"] += 1

    _log(
        f"STEP 10 complete: strengthened={agg['strengthened']} weakened={agg['weakened']} "
        f"modified={agg['modified']} unchanged={agg['unchanged']} errors={agg['error']}"
    )
    return {"theme_evolution": theme_evolution, "aggregate": agg}
