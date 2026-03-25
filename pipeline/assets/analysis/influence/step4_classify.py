"""Step 4: Theme classification of amendments — AI-assisted with regex fallback."""

from __future__ import annotations

import re
from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from ._helpers import compile_taxonomy_patterns, _classify_by_regex, _taxonomy_summary_for_prompt
from . import _config


def step4_classify_amendments(
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Classify each amendment against the theme taxonomy."""
    _log = logger.info if logger else print

    if not amendments:
        _log("No amendments to classify.")
        return amendments

    if not taxonomy:
        _log("No taxonomy available — cannot classify amendments.")
        return amendments

    compiled = compile_taxonomy_patterns(taxonomy)
    taxonomy_summary = _taxonomy_summary_for_prompt(taxonomy)

    # --- Compute diff summaries and filter no-change amendments ---
    def _compute_diff_summary(am: dict[str, Any]) -> str:
        """Return a concise diff string showing what actually changed."""
        orig = (am.get("original_text") or "").strip()
        amend = (am.get("amended_text") or "").strip()

        # If both sides are empty, fall back to body excerpt
        if not orig and not amend:
            return am.get("body", "")[:600]

        # Identical or near-identical after stripping
        if orig and amend and orig == amend:
            return "__no_change__"

        if not orig:
            return f"NEW: {amend[:500]}"

        amend_lower = amend.lower()
        if not amend or amend_lower == "deleted" or amend_lower == "deletion":
            return f"DELETED: {orig[:500]}"

        return f"BEFORE: {orig[:300]}\nAFTER: {amend[:300]}"

    # Mark no-change amendments and give them empty themes
    no_change_count = 0
    classifiable: list[dict[str, Any]] = []
    for am in amendments:
        diff = _compute_diff_summary(am)
        am["_diff_summary"] = diff
        if diff == "__no_change__":
            am["themes"] = []
            no_change_count += 1
        else:
            classifiable.append(am)

    if no_change_count:
        _log(f"Skipping {no_change_count} no-change amendments (identical original/amended text)")

    batch_size = 15
    batches = [classifiable[i : i + batch_size] for i in range(0, len(classifiable), batch_size)]
    _log(
        f"Classifying {len(classifiable)} amendments in {len(batches)} batch(es) "
        f"of up to {batch_size} (parallel, {_config.AI_MAX_WORKERS} workers) ..."
    )

    batch_prompts: list[str] = []
    for batch in batches:
        items_text = "\n\n".join(
            f"[AM-{am['number']}] {am.get('location', '')}\n"
            f"CHANGE: {am['_diff_summary']}\n"
            f"JUSTIFICATION: {(am.get('justification') or '')[:300]}"
            for am in batch
        )
        batch_prompts.append(
            f"""Classify each amendment by policy theme based on WHAT CHANGED in the text, not merely what topic it concerns.

TAXONOMY:
{taxonomy_summary}

AMENDMENTS TO CLASSIFY:
{items_text}

For each amendment, return a JSON array where each entry has:
  "number": the amendment number (integer)
  "themes": list of theme keys from the taxonomy (may be empty list)

Example: [{{"number": 42, "themes": ["holding_limits", "privacy_data_protection"]}}, ...]

Only use theme keys from the taxonomy. An amendment may match 0, 1, or multiple themes.
Respond ONLY with the JSON array."""
        )

    raw_responses = ai_complete_parallel(
        batch_prompts, json_mode=True, label="classify", logger=logger
    )

    for batch, raw in zip(batches, raw_responses):
        parsed = parse_json_response(raw) if raw else None
        if parsed and isinstance(parsed, list):
            result_map = {entry.get("number"): entry.get("themes", []) for entry in parsed}
            for am in batch:
                ai_themes = result_map.get(am["number"])
                if ai_themes is not None:
                    am["themes"] = [t for t in ai_themes if t in taxonomy]
                else:
                    text = am.get("body", "") + " " + am.get("justification", "") + " " + am.get("location", "")
                    am["themes"] = _classify_by_regex(text, compiled)
        else:
            for am in batch:
                text = am.get("body", "") + " " + am.get("justification", "") + " " + am.get("location", "")
                am["themes"] = _classify_by_regex(text, compiled)

    # Clean up temporary diff key
    for am in amendments:
        am.pop("_diff_summary", None)

    classified = sum(1 for a in amendments if a["themes"])
    _log(f"Classified: {classified}/{len(amendments)} matched at least one theme")
    return amendments
