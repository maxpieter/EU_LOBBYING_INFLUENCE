"""Step 5: Position extraction — AI-assisted structured positions from commission meetings."""

from __future__ import annotations

from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from ._helpers import _classify_by_regex, _meeting_text, _taxonomy_summary_for_prompt


def step5_extract_positions(
    commission_meetings: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    compiled_patterns: dict[str, list],
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Extract structured positions from commission meeting texts."""
    _log = logger.info if logger else print

    substantive = [
        m for m in commission_meetings
        if m.get("points_raised")
    ]
    _log(f"Commission meetings with points_raised text: {len(substantive)}")

    if not substantive:
        _log("Nothing to extract.")
        return []

    positions: list[dict[str, Any]] = []
    for m in substantive:
        text = _meeting_text(m)
        themes = _classify_by_regex(text, compiled_patterns)
        resolved = m.get("resolved_orgs") or []
        org_names = [o["name"] for o in resolved if o.get("name")]
        if not org_names:
            raw = (m.get("organizations_raw") or "").strip()
            org_names = [raw.split("|")[0].strip()] if raw else ["Unknown"]
        positions.append(
            {
                "meeting_id": m.get("id"),
                "date": str(m.get("meeting_date", ""))[:10],
                "commissioner": m.get("commissioner_name", ""),
                "orgs": org_names,
                "themes": themes,
                "summary": (m.get("points_raised") or "")[:200],
                "ai_enhanced": False,
            }
        )

    if not taxonomy:
        _log(f"Extracted {len(positions)} positions (no taxonomy — skipping AI enhancement).")
        return positions

    taxonomy_summary = _taxonomy_summary_for_prompt(taxonomy)
    batch_size = 4
    batches = [substantive[i : i + batch_size] for i in range(0, len(substantive), batch_size)]
    _log(
        f"Enhancing {len(substantive)} meetings in {len(batches)} batch(es) via AI ..."
    )

    meeting_id_to_position = {p["meeting_id"]: p for p in positions}

    batch_prompts: list[str] = []
    for batch in batches:
        items = []
        for m in batch:
            resolved = m.get("resolved_orgs") or []
            org_names_str = ", ".join(o["name"] for o in resolved if o.get("name")) or (
                m.get("organizations_raw") or "Unknown"
            )
            items.append(
                f'Meeting ID: {m["id"]}\n'
                f'Date: {m.get("meeting_date", "")}\n'
                f'Organisations: {org_names_str}\n'
                f'Points raised: {(m.get("points_raised") or "")[:800]}'
            )

        batch_prompts.append(
            f"""Extract structured lobbying positions from these Commission meeting records.

IMPORTANT: Some meetings may discuss a DIFFERENT regulation or topic that is unrelated to the taxonomy below. If a meeting's points_raised text is NOT about the themes in this taxonomy, set "relevant" to false. Only extract positions from meetings that are genuinely about this legislation.

TAXONOMY:
{taxonomy_summary}

MEETINGS:
{"---".join(items)}

For each meeting, return a JSON array where each entry has:
  "meeting_id": the meeting ID string
  "relevant": true if the meeting is actually about this legislation, false if it discusses unrelated topics
  "themes": list of relevant theme keys from the taxonomy (empty list if not relevant)
  "summary": one sentence capturing the core position taken (or "Not relevant to this procedure" if not relevant)

Respond ONLY with the JSON array."""
        )

    raw_responses = ai_complete_parallel(
        batch_prompts, json_mode=True, label="positions", logger=logger
    )

    for raw in raw_responses:
        parsed = parse_json_response(raw) if raw else None
        if parsed and isinstance(parsed, list):
            for entry in parsed:
                mid = entry.get("meeting_id")
                if mid and mid in meeting_id_to_position:
                    pos = meeting_id_to_position[mid]
                    # Check relevance flag
                    relevant = entry.get("relevant", True)
                    if relevant is False or (isinstance(relevant, str) and relevant.lower() == "false"):
                        pos["relevant"] = False
                        pos["ai_enhanced"] = True
                        continue
                    pos["relevant"] = True
                    ai_themes = [t for t in (entry.get("themes") or []) if t in taxonomy]
                    if ai_themes:
                        pos["themes"] = ai_themes
                    if entry.get("summary"):
                        pos["summary"] = entry["summary"][:500]
                    pos["ai_enhanced"] = True

    enhanced = sum(1 for p in positions if p.get("ai_enhanced"))

    # Filter out meetings the AI flagged as irrelevant to this procedure
    before = len(positions)
    positions = [p for p in positions if p.get("relevant", True) is not False]
    filtered = before - len(positions)
    if filtered:
        _log(f"Filtered {filtered} meetings flagged as irrelevant to this procedure")

    _log(f"Positions extracted: {len(positions)} ({enhanced} AI-enhanced)")
    return positions
