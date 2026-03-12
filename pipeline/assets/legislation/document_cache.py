"""Document cache for eliminating redundant downloads across pipeline stages.

Prevents duplicate EUR-Lex requests by caching downloaded HTML/text content.
Silver and Gold stages can reuse cached documents, avoiding bot detection.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


def get_cache_path(url: str, cache_dir: Path) -> Path:
    """Generate cache file path from URL.

    Uses SHA256 hash with 2-char subdirectory to avoid too many files per directory.

    Args:
        url: Document URL
        cache_dir: Base cache directory

    Returns:
        Path to cache file
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    # Use first 2 chars for subdirectory (limits files per dir to ~256)
    subdir = url_hash[:2]
    return cache_dir / subdir / f"{url_hash}.json"


def is_cache_fresh(cache_path: Path, ttl_days: int = 7) -> bool:
    """Check if cached document is still fresh (within TTL).

    Args:
        cache_path: Path to cache file
        ttl_days: Time-to-live in days (default 7 days)

    Returns:
        True if cache exists and is fresh
    """
    if not cache_path.exists():
        return False

    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=ttl_days)
    except (OSError, ValueError):
        return False


def get_cache_age_hours(cache_path: Path) -> float:
    """Get cache age in hours for logging.

    Args:
        cache_path: Path to cache file

    Returns:
        Age in hours, or 0 if error
    """
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age.total_seconds() / 3600
    except (OSError, ValueError):
        return 0


def read_from_cache(cache_path: Path, logger: Optional[Any] = None) -> Optional[tuple[str, str]]:
    """Read (html, text) from cache.

    Args:
        cache_path: Path to cache file
        logger: Optional logger

    Returns:
        Tuple of (html_content, text_content) or None if read failed
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        html = data.get("html", "")
        text = data.get("text", "")

        if not html or not text:
            _log("Cache file missing html or text content", "warning")
            return None

        age_hours = get_cache_age_hours(cache_path)
        _log(f"Cache hit (age: {age_hours:.1f}h)", "debug")

        return (html, text)
    except (json.JSONDecodeError, OSError, KeyError) as e:
        _log(f"Failed to read cache: {e}", "warning")
        return None


def write_to_cache(
    cache_path: Path, html: str, text: str, url: str, logger: Optional[Any] = None
) -> bool:
    """Write (html, text) to cache.

    Args:
        cache_path: Path to cache file
        html: HTML content
        text: Plain text content
        url: Source URL (for metadata)
        logger: Optional logger

    Returns:
        True if write succeeded
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "url": url,
            "html": html,
            "text": text,
            "cached_at": datetime.now().isoformat(),
            "html_size": len(html),
            "text_size": len(text),
        }

        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _log(
            f"Cached document ({len(html):,} chars HTML, {len(text):,} chars text)",
            "debug",
        )
        return True
    except (OSError, TypeError) as e:
        _log(f"Failed to write cache: {e}", "warning")
        return False


def clear_stale_cache(cache_dir: Path, ttl_days: int = 7, logger: Optional[Any] = None):
    """Remove cache files older than TTL.

    Args:
        cache_dir: Base cache directory
        ttl_days: Time-to-live in days
        logger: Optional logger
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    if not cache_dir.exists():
        return

    removed_count = 0
    total_size = 0

    try:
        for cache_file in cache_dir.rglob("*.json"):
            if not is_cache_fresh(cache_file, ttl_days):
                try:
                    size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    total_size += size
                except OSError:
                    pass

        if removed_count > 0:
            _log(
                f"Cleared {removed_count} stale cache files ({total_size / 1024 / 1024:.1f} MB)",
                "info",
            )
    except Exception as e:
        _log(f"Error clearing cache: {e}", "warning")


def get_cache_stats(cache_dir: Path) -> dict:
    """Get cache statistics for monitoring.

    Args:
        cache_dir: Base cache directory

    Returns:
        Dict with cache statistics
    """
    if not cache_dir.exists():
        return {"total_files": 0, "total_size_mb": 0, "oldest_age_hours": 0}

    try:
        cache_files = list(cache_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        oldest_age = 0
        if cache_files:
            oldest_mtime = min(f.stat().st_mtime for f in cache_files)
            oldest_age = (
                datetime.now() - datetime.fromtimestamp(oldest_mtime)
            ).total_seconds() / 3600

        return {
            "total_files": len(cache_files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "oldest_age_hours": round(oldest_age, 1),
        }
    except (OSError, ValueError):
        return {"total_files": 0, "total_size_mb": 0, "oldest_age_hours": 0}
