"""Procedure alias generation via Claude Opus.

Seeds the procedure_aliases table with acronyms, short names, informal names,
and policy package names for the top 100 detailed procedures. Uses Claude Opus
4.6 for high-quality alias generation.

The other ~4,800 catalog-only procedures rely on title matching + trigram
similarity (no AI-generated aliases needed).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


def generate_aliases_for_procedures(
    procedures: list[dict],
    anthropic_client: Any,
    logger: Optional[Any] = None,
    batch_size: int = 10,
    workers: int = 3,
) -> list[dict]:
    """Generate aliases for a list of procedures via Claude Opus.

    Args:
        procedures: list of {id, title, procedure_type, subjects, ...}
        anthropic_client: Anthropic SDK client
        batch_size: procedures per API call
        workers: parallel workers

    Returns:
        list of {procedure_id, alias, alias_type} dicts for upsert.
    """
    _log = logger.info if logger else print

    if not procedures:
        return []

    batches = [procedures[i:i + batch_size] for i in range(0, len(procedures), batch_size)]
    _log(f"Generating aliases for {len(procedures)} procedures in {len(batches)} batches")

    all_aliases: list[dict] = []

    def _process_batch(batch_idx: int, batch: list[dict]) -> list[dict]:
        prompt = _build_alias_prompt(batch)
        for attempt in range(3):
            try:
                response = anthropic_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    time.sleep(2 ** attempt * 5)
                    continue
                raise
        else:
            return []

        raw = response.content[0].text
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return []

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []

        results = []
        for entry in parsed:
            proc_id = entry.get("id", "")
            for alias_obj in entry.get("aliases", []):
                alias_text = alias_obj.get("alias", "").strip()
                alias_type = alias_obj.get("type", "informal")
                if alias_text and proc_id:
                    results.append({
                        "procedure_id": proc_id,
                        "alias": alias_text,
                        "alias_type": alias_type,
                    })
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                batch_aliases = fut.result()
                all_aliases.extend(batch_aliases)
                _log(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch_aliases)} aliases")
            except Exception as e:
                _log(f"  Batch {batch_idx + 1}/{len(batches)} failed: {e}")

    _log(f"Generated {len(all_aliases)} total aliases")
    return all_aliases


def _build_alias_prompt(procedures: list[dict]) -> str:
    """Build the alias generation prompt for a batch of procedures."""
    items = []
    for p in procedures:
        subjects = p.get("subjects", [])
        subjects_str = ", ".join(subjects[:5]) if subjects else "N/A"
        items.append(
            f'- ID: "{p["id"]}"\n'
            f'  Title: "{p["title"]}"\n'
            f'  Type: {p.get("procedure_type", "N/A")}\n'
            f'  Subjects: {subjects_str}'
        )

    return (
        "For each EU legislative procedure below, generate common aliases that "
        "people use to refer to it. Include:\n"
        "- Acronyms (e.g., 'DSA', 'DMA', 'CBAM', 'CSRD')\n"
        "- Short names (e.g., 'AI Act', 'Chips Act', 'Nature Restoration Law')\n"
        "- Informal names used in media/policy discussions\n"
        "- Policy package names if applicable (e.g., 'Fit for 55', 'Green Deal')\n\n"
        "Only include aliases that are COMMONLY used — do not invent new ones.\n"
        "If a procedure has no well-known aliases beyond its title, return an empty list.\n\n"
        "Procedures:\n" + "\n".join(items) + "\n\n"
        "Respond with a JSON array:\n"
        '[{"id": "2020/0361(COD)", "aliases": [{"alias": "DSA", "type": "acronym"}, '
        '{"alias": "Digital Services Act", "type": "short_name"}]}, ...]\n'
        f"Return exactly {len(procedures)} entries in order."
    )
