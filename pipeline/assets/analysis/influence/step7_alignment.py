"""Step 7: Directional alignment — AI-scored amendment-to-lobby alignment."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ._ai import ai_complete_parallel, parse_json_response
from . import _config


def _amendment_mentions_mep(amendment: dict[str, Any], mep_name: str) -> bool:
    authors = amendment.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    name_low = mep_name.lower()
    parts = name_low.split()
    surname = parts[-1] if parts else ""
    for author in authors:
        author_low = author.lower()
        if surname and surname in author_low:
            return True
        if name_low in author_low:
            return True
    return not authors


def step7_directional_alignment(
    quant: dict[str, Any],
    positions: list[dict[str, Any]],
    amendments: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    logger: Any = None,
) -> dict[str, Any]:
    """Score directional alignment between org positions and MEP amendments."""
    _log = logger.info if logger else print

    mep_crossref = quant.get("mep_crossref", {})
    ranked = sorted(
        mep_crossref.items(), key=lambda x: x[1].get("total_meetings", 0), reverse=True
    )
    top_meps = ranked[:10]

    if not top_meps:
        _log("No MEP data available.")
        return {}

    positions_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos in positions:
        for t in pos.get("themes", []):
            positions_by_theme[t].append(pos)

    mep_alignment: dict[str, Any] = {}
    all_tasks: list[tuple[str, str, list[dict], str]] = []

    for mep_name, crossref in top_meps:
        overlapping_themes = crossref.get("overlapping_themes", [])
        if not overlapping_themes:
            mep_alignment[mep_name] = {
                "total_pairs": 0, "toward": 0, "away": 0, "neutral": 0,
                "alignment_fraction": None, "theme_scores": {},
            }
            continue

        for theme in overlapping_themes:
            theme_positions = positions_by_theme.get(theme, [])
            theme_amendments = [
                a for a in amendments
                if theme in (a.get("themes") or []) and _amendment_mentions_mep(a, mep_name)
            ]
            if not theme_positions or not theme_amendments:
                continue

            pairs: list[dict[str, str]] = []
            for pos in theme_positions[:5]:
                for am in theme_amendments[:5]:
                    pairs.append(
                        {
                            "position_org": (pos.get("orgs") or ["Unknown"])[0],
                            "position_summary": pos.get("summary", "")[:300],
                            "amendment_number": str(am.get("number", "")),
                            "amendment_excerpt": (am.get("body") or "")[:400],
                        }
                    )
            if not pairs:
                continue

            prompt = f"""Assess whether each amendment moves regulation TOWARD or AWAY FROM the organisation's stated position.

THEME: {theme}
TAXONOMY CONTEXT: {taxonomy.get(theme, {}).get("description", "")}

PAIRS TO ASSESS:
{json.dumps(pairs, indent=2)}

For each pair, return a JSON array where each entry has:
  "position_org": organisation name (unchanged from input)
  "amendment_number": amendment number (unchanged from input)
  "score": 1 (amendment moves TOWARD the org's position), 0 (neutral/unclear), or -1 (AWAY from the org's position)
  "reasoning": one short sentence explaining the score

Respond ONLY with the JSON array."""
            all_tasks.append((mep_name, theme, pairs, prompt))

    if all_tasks:
        _log(f"Firing {len(all_tasks)} alignment prompts in parallel ({_config.AI_MAX_WORKERS} workers) ...")
        prompts_only = [t[3] for t in all_tasks]
        raw_responses = ai_complete_parallel(
            prompts_only, json_mode=True, label="alignment", logger=logger
        )

        mep_scores: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"theme_scores": {}, "total_toward": 0, "total_away": 0, "total_neutral": 0}
        )

        for (mep_name, theme, pairs, _prompt), raw in zip(all_tasks, raw_responses):
            scored_pairs = parse_json_response(raw) if raw else None
            t_toward = t_away = t_neutral = 0
            if scored_pairs and isinstance(scored_pairs, list):
                for entry in scored_pairs:
                    score = entry.get("score", 0)
                    if score == 1:
                        t_toward += 1
                    elif score == -1:
                        t_away += 1
                    else:
                        t_neutral += 1
            else:
                t_neutral = len(pairs)

            mep_scores[mep_name]["total_toward"] += t_toward
            mep_scores[mep_name]["total_away"] += t_away
            mep_scores[mep_name]["total_neutral"] += t_neutral
            mep_scores[mep_name]["theme_scores"][theme] = {
                "pairs_evaluated": len(pairs),
                "toward": t_toward,
                "away": t_away,
                "neutral": t_neutral,
                "pair_details": scored_pairs or [],
            }

        for mep_name, scores_data in mep_scores.items():
            total_toward = scores_data["total_toward"]
            total_away = scores_data["total_away"]
            total_neutral = scores_data["total_neutral"]
            total_pairs = total_toward + total_away + total_neutral
            alignment_fraction = total_toward / total_pairs if total_pairs > 0 else None

            mep_alignment[mep_name] = {
                "total_pairs": total_pairs,
                "toward": total_toward,
                "away": total_away,
                "neutral": total_neutral,
                "alignment_fraction": (
                    round(alignment_fraction, 3) if alignment_fraction is not None else None
                ),
                "theme_scores": scores_data["theme_scores"],
            }
            _log(
                f"  {mep_name}: {total_pairs} pairs | "
                f"toward={total_toward} away={total_away} neutral={total_neutral}"
            )

    for mep_name, _ in top_meps:
        if mep_name not in mep_alignment:
            mep_alignment[mep_name] = {
                "total_pairs": 0, "toward": 0, "away": 0, "neutral": 0,
                "alignment_fraction": None, "theme_scores": {},
            }

    return mep_alignment
