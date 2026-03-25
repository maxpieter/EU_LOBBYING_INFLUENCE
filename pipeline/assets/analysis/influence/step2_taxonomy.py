"""Step 2: Theme taxonomy generation — AI-assisted, cached to disk."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import _config
from ._ai import ai_complete, parse_json_response
from ._helpers import safe_id


def step2_generate_taxonomy(
    procedure_id: str,
    data: dict[str, Any],
    regen: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate or load a policy-theme taxonomy for the procedure.

    Parameters
    ----------
    procedure_id:
        EU procedure reference.
    data:
        Output of ``step1_collect_data``.
    regen:
        When True, delete the cached taxonomy and regenerate via AI.
    logger:
        Optional logger.
    """
    _log = logger.info if logger else print

    _config.TAXONOMY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _config.TAXONOMY_CACHE_DIR / f"{safe_id(procedure_id)}.json"

    if regen and cache_path.exists():
        cache_path.unlink()
        _log(f"Cleared taxonomy cache: {cache_path}")

    if cache_path.exists():
        _log(f"Loading cached taxonomy from {cache_path}")
        with cache_path.open(encoding="utf-8") as fh:
            taxonomy = json.load(fh)
        _log(f"Loaded {len(taxonomy)} themes: {', '.join(taxonomy.keys())}")
        return taxonomy

    if _config.AI_PROVIDER is None:
        raise RuntimeError(
            "AI provider is not configured. Run configure_ai_provider() before calling "
            "step2_generate_taxonomy, and ensure the 'claude' CLI is installed and on PATH."
        )

    procedure = data["procedure"]
    title = procedure.get("title", "Unknown procedure")
    description = (procedure.get("description") or "")[:2000]
    proposal_text = data.get("proposal_text", "")

    system_prompt = (
        "You are an expert EU policy analyst specialising in legislative analysis "
        "and lobbying research. You identify contested political dimensions in "
        "EU legislative proposals by studying the text and understanding the "
        "stakeholder landscape."
    )

    user_prompt = f"""Analyse the following EU legislative procedure and identify 5-12 distinct, contested policy themes (political dimensions) that different stakeholders are likely to lobby on.

PROCEDURE: {procedure_id}
TITLE: {title}
DESCRIPTION: {description}

{"COMMISSION PROPOSAL EXCERPT:" + chr(10) + proposal_text if proposal_text else ""}

For each theme, respond with a JSON object structured as follows:

{{
  "themes": [
    {{
      "key": "snake_case_theme_key",
      "description": "One-sentence human-readable description of the policy dimension",
      "articles": ["Article 3", "Recital 5"],
      "keywords": [
        "regex_pattern_1",
        "regex_pattern_2",
        "regex_pattern_3",
        "regex_pattern_4",
        "regex_pattern_5"
      ],
      "salience": "Why this dimension is politically contested and which stakeholders care"
    }}
  ]
}}

Requirements:
- 5-12 themes, each genuinely distinct from the others
- keywords must be valid Python regex patterns (use \\b for word boundaries, \\s+ for spaces)
- articles should reference specific Articles or Recitals from the proposal
- themes should reflect real stakeholder conflicts (industry vs. civil society, member state vs. Commission, etc.)

Respond ONLY with the JSON object above."""

    _log("Calling AI to generate theme taxonomy ...")
    raw = ai_complete(user_prompt, system=system_prompt, json_mode=True)
    time.sleep(_config.AI_RATE_SLEEP)

    parsed = parse_json_response(raw, retry_prompt=user_prompt)
    if not parsed or not isinstance(parsed, dict):
        _log("[WARN] Could not parse AI taxonomy response — using empty taxonomy.")
        return {}

    themes_list = parsed.get("themes", [])
    if not themes_list:
        _log("[WARN] AI returned no themes — using empty taxonomy.")
        return {}

    taxonomy: dict[str, Any] = {}
    for theme in themes_list:
        key = theme.get("key", "").strip()
        if not key:
            continue
        taxonomy[key] = {
            "description": theme.get("description", ""),
            "articles": theme.get("articles", []),
            "keywords": theme.get("keywords", []),
            "salience": theme.get("salience", ""),
        }

    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(taxonomy, fh, indent=2, ensure_ascii=False)
    _log(f"Generated {len(taxonomy)} themes; cached to {cache_path}")
    _log(f"Themes: {', '.join(taxonomy.keys())}")
    return taxonomy
