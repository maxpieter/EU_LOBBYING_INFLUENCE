"""Module-level configuration: paths, constants, AI provider state."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "analysis"
TAXONOMY_CACHE_DIR = DATA_DIR / "theme_taxonomies"
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"

# ---------------------------------------------------------------------------
# AI provider configuration
# ---------------------------------------------------------------------------

AI_PROVIDER: str | None = None
AI_RATE_SLEEP: float = 0.5
AI_MAX_WORKERS: int = 3  # Conservative for CLI subprocess calls


def set_ai_provider(value: str | None) -> None:
    """Set the AI_PROVIDER global (called by configure_ai_provider)."""
    global AI_PROVIDER
    AI_PROVIDER = value
