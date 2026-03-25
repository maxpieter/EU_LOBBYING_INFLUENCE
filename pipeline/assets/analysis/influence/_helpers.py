"""Shared utility functions: regex classification, text helpers."""

from __future__ import annotations

import re
from typing import Any


def safe_id(procedure_id: str) -> str:
    """Convert '2023/0212(COD)' to a safe filename stem."""
    return re.sub(r"[/()]+", "_", procedure_id).strip("_")


def compile_taxonomy_patterns(taxonomy: dict[str, Any]) -> dict[str, list[re.Pattern]]:
    compiled: dict[str, list[re.Pattern]] = {}
    for key, cfg in taxonomy.items():
        patterns: list[re.Pattern] = []
        for raw_pattern in cfg.get("keywords", []):
            try:
                patterns.append(re.compile(raw_pattern, re.IGNORECASE))
            except re.error:
                pass
        compiled[key] = patterns
    return compiled


def _classify_by_regex(text: str, patterns: dict[str, list[re.Pattern]]) -> list[str]:
    if not text:
        return []
    matched: list[str] = []
    for theme, pats in patterns.items():
        for pat in pats:
            if pat.search(text):
                matched.append(theme)
                break
    return matched


def _meeting_text(meeting: dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                meeting.get("subject"),
                meeting.get("points_raised"),
                meeting.get("conclusions"),
                meeting.get("title"),
            ],
        )
    )


def _taxonomy_summary_for_prompt(taxonomy: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, cfg in taxonomy.items():
        desc = cfg.get("description", "")
        kw_sample = ", ".join(cfg.get("keywords", [])[:4])
        lines.append(f'  "{key}": {desc}  [keywords: {kw_sample}]')
    return "\n".join(lines)


def _extract_theme_sections(
    text: str,
    theme_key: str,
    compiled_patterns: dict[str, list[re.Pattern]],
    max_chars: int = 2000,
) -> str:
    """Return theme-relevant paragraphs from *text*, truncated to *max_chars*."""
    if not text:
        return ""
    pats = compiled_patterns.get(theme_key, [])
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    if pats:
        matched = [p for p in paragraphs if any(pat.search(p) for pat in pats)]
    else:
        matched = []
    if matched:
        joined = "\n\n".join(matched)
    else:
        joined = text
    return joined[:max_chars]
