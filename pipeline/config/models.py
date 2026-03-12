"""Centralized AI model configuration for pipeline.

All LLM model names are defined here for easy updates and consistency.
"""

from pathlib import Path


class DocumentCacheConfig:
    """Document cache configuration for eliminating redundant downloads.

    Shared cache between Silver and Gold stages prevents duplicate EUR-Lex requests.
    """

    # Cache directory (relative to project root)
    CACHE_DIR = Path("data/.cache/documents")

    # Time-to-live in days (legislative documents rarely change)
    TTL_DAYS = 7

    # Enable/disable caching (useful for testing)
    ENABLED = True


class ModelConfig:
    """AI model configuration for EU Parliament pipeline.

    This centralizes all model names used across the pipeline stages.
    Update here to change models globally.
    """

    # Translation models
    TRANSLATION = "gpt-5-nano"
    TRANSLATION_REASONING_EFFORT = "minimal"

    # Summarization and analysis models
    SUMMARIZATION = "gpt-5-mini"
    ANALYSIS = "gpt-5-mini"
    ANALYSIS_REASONING_EFFORT = "minimal"

    # Embedding models
    EMBEDDING = "text-embedding-3-small"

    # Evaluation models
    EVALUATION = "gpt-5-mini"

    # Batch processing configuration
    EMBEDDING_BATCH_SIZE = 50
    EMBEDDING_POLL_INTERVAL = 10

    # Text processing limits
    MAX_TEXT_LENGTH = 32000

    @classmethod
    def get_model_for_task(cls, task: str) -> str:
        """Get the appropriate model for a given task."""
        task_map = {
            "translation": cls.TRANSLATION,
            "summarization": cls.SUMMARIZATION,
            "analysis": cls.ANALYSIS,
            "embedding": cls.EMBEDDING,
            "evaluation": cls.EVALUATION,
        }
        return task_map.get(task.lower(), cls.ANALYSIS)


class LobbyingConfig:
    """Configuration for lobbying data extraction."""

    MEETINGS_API_URL = "https://www.europarl.europa.eu/meps/en/search-meetings"
    TRANSPARENCY_REGISTER_XLS_PATH = None
    API_DATE_FORMAT = "%d/%m/%Y"
    DEFAULT_DATE_RANGE_DAYS = 30
