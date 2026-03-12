"""Utility functions for document downloading, conversion (DOCX→HTML), and URL construction.

NOTE: This module uses HttpClientResource when available (passed as parameter),
falling back to standalone session only for backward compatibility.
"""

import io
import re
from typing import Any, Dict, Optional

import requests


def _download_with_selenium(
    url: str,
    selenium_resource: Any,
    logger: Optional[Any] = None,
    cache_dir: Optional[Any] = None,
    cache_ttl_days: int = 7,
) -> Optional[tuple[str, str]]:
    """Download a document using a headless browser to bypass AWS WAF challenges.

    Used automatically when plain HTTP requests trigger bot-detection (HTTP 202
    with WAF challenge page).  The Selenium driver executes JavaScript, acquires
    the WAF session cookie, and returns the real page HTML.

    Args:
        url: Document URL
        selenium_resource: SeleniumResource instance
        logger: Optional logger
        cache_dir: Optional cache directory — result is written to cache on success
        cache_ttl_days: Cache TTL in days

    Returns:
        Tuple of (html_content, text_content) or None if failed
    """
    import time

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    driver = None
    try:
        _log(f"🌐 Selenium fallback: navigating to {url}", "info")
        driver = selenium_resource.get_driver()
        driver.get(url)

        # Wait for AWS WAF JS challenge to execute and redirect to real page
        time.sleep(5)

        html_content = driver.page_source

        # If WAF page is still showing, wait longer
        if "awsWafCookieDomainList" in html_content or len(html_content) < 1000:
            _log("⏳ WAF challenge still active, waiting 10s more...", "info")
            time.sleep(10)
            html_content = driver.page_source

        # Validate we got real content and not a persistent block
        if "awsWafCookieDomainList" in html_content:
            _log("❌ Selenium: AWS WAF challenge did not resolve", "warning")
            return None

        if len(html_content) < 500:
            _log(f"❌ Selenium: page too short ({len(html_content)} chars), skipping", "warning")
            return None

        from .document_parser import extract_text_from_html

        text_content = extract_text_from_html(html_content)

        if not text_content or len(text_content) < 500:
            _log("❌ Selenium: extracted text too short, skipping", "warning")
            return None

        _log(
            f"✅ Selenium: {len(html_content):,} chars HTML, {len(text_content):,} chars text",
            "info",
        )

        # Write to cache so subsequent pipeline stages skip the download entirely
        if cache_dir:
            from pathlib import Path

            from .document_cache import get_cache_path, write_to_cache

            cache_path = get_cache_path(url, Path(cache_dir))
            write_to_cache(cache_path, html_content, text_content, url, logger)

        return (html_content, text_content)

    except Exception as e:
        _log(f"❌ Selenium fallback error for {url}: {e}", "error")
        return None
    finally:
        if driver is not None:
            try:
                selenium_resource.return_driver(driver)
            except Exception:
                pass


def download_document(
    url: str,
    logger: Optional[Any] = None,
    max_retries: int = 3,
    retry_delay: int = 10,
    http_client: Optional[Any] = None,
    cache_dir: Optional[Any] = None,
    cache_ttl_days: int = 7,
    selenium_resource: Optional[Any] = None,
) -> Optional[tuple[str, str]]:
    """Download document (HTML or DOCX) and return (html, text).

    For DOCX files, converts to HTML using mammoth library.

    Prefers using HttpClientResource (if provided) which handles:
    - EUR-Lex 10s rate limiting per robots.txt
    - Connection pooling and cookie persistence
    - Domain-based rate limiting

    Falls back to standalone session if http_client not provided (legacy).

    If the response is an AWS WAF bot-detection challenge (HTTP 202 + WAF JS
    page), and selenium_resource is provided, automatically retries using a
    headless browser that executes JavaScript to pass the challenge.  The result
    is written to cache so subsequent requests use the cached copy.

    Supports caching to eliminate redundant downloads across pipeline stages:
    - Silver downloads → cache → Gold reads from cache
    - Prevents EUR-Lex bot detection from duplicate requests

    Args:
        url: Document URL (prefer CELEX format for EUR-Lex)
        logger: Optional logger
        max_retries: Maximum number of retries for HTTP 202 responses
        retry_delay: Initial delay between retries in seconds
        http_client: Optional HttpClientResource for rate-limited requests
        cache_dir: Optional cache directory path (enables caching if provided)
        cache_ttl_days: Cache time-to-live in days (default 7)
        selenium_resource: Optional SeleniumResource for WAF bypass fallback

    Returns:
        Tuple of (html_content, text_content) or None if failed
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    # Check cache first (if caching enabled)
    if cache_dir:
        from pathlib import Path

        from .document_cache import (
            get_cache_path,
            is_cache_fresh,
            read_from_cache,
            write_to_cache,
        )

        cache_path = get_cache_path(url, Path(cache_dir))
        if is_cache_fresh(cache_path, cache_ttl_days):
            cached_result = read_from_cache(cache_path, logger)
            if cached_result:
                return cached_result
            # Cache read failed, fall through to download

    retry_count = 0

    while retry_count <= max_retries:
        try:
            _log(f"Downloading: {url}", "debug")

            # Add Referer for EUR-Lex
            request_headers = {}
            if "eur-lex.europa.eu" in url:
                request_headers["Referer"] = "https://eur-lex.europa.eu/"

            # Use http_client.get() for rate-limited requests, or fallback session
            if http_client:
                response = http_client.get(url, headers=request_headers, timeout=60)
            else:
                # Fallback: create basic session (for backward compatibility)
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    }
                )
                response = session.get(url, headers=request_headers, timeout=60)

            if response.status_code == 202:
                # Log diagnostic info to identify if this is bot-detection vs document generation
                body_preview = response.text[:500] if response.text else ""
                _log(f"HTTP 202 body preview (first 500 chars): {body_preview}", "info")

                # Check if this is bot-detection (JS verification gate)
                is_bot_gate = any(
                    phrase in response.text.lower()
                    for phrase in [
                        "verify that you're not a robot",
                        "javascript is disabled",
                        "javascript must be enabled",
                        "this requires javascript",
                        "bot",
                        "captcha",
                        "awswafcookie",  # AWS WAF challenge
                        "gokuprops",  # AWS WAF challenge
                        "window.awswaf",  # AWS WAF challenge
                    ]
                )

                if is_bot_gate:
                    _log(
                        "⚠️ EUR-Lex bot-detection gate triggered. "
                        "This is not rate limiting - EUR-Lex is requiring JS/browser verification.",
                        "warning",
                    )
                    # Try Selenium fallback if available
                    if selenium_resource is not None:
                        _log("🌐 Attempting Selenium WAF bypass...", "info")
                        return _download_with_selenium(
                            url,
                            selenium_resource,
                            logger=logger,
                            cache_dir=cache_dir,
                            cache_ttl_days=cache_ttl_days,
                        )
                    # No Selenium available — give up on this URL
                    break

                if retry_count < max_retries:
                    # EUR-Lex document generation in progress
                    # Rely on http_client's rate limiting (10s for EUR-Lex) between retries
                    # No manual sleep needed - the rate limiter ensures proper delays
                    _log(
                        f"Document generation in progress (HTTP 202). Retry {retry_count + 1}/{max_retries} "
                        f"(rate limiter will enforce 10s delay on next request)",
                        "info",
                    )
                    retry_count += 1
                    # Don't sleep here - let http_client.get() enforce the proper rate limit
                    continue
                else:
                    _log(
                        f"EUR-Lex HTTP 202 after {max_retries} attempts: {url}. Visit URL manually in browser to warm cache, then re-run.",
                        "warning",
                    )
                    return None
            elif response.status_code != 200:
                _log(f"Failed to download: HTTP {response.status_code}", "warning")
                return None

            # Success - process the document
            # Check if it's a DOCX file
            is_docx = url.endswith(
                ".docx"
            ) or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in response.headers.get(
                "content-type", ""
            )

            if is_docx:
                # Convert DOCX to HTML using mammoth
                try:
                    import mammoth

                    # Convert binary content to HTML
                    docx_file = io.BytesIO(response.content)
                    result = mammoth.convert_to_html(docx_file)
                    html_content = result.value

                    # Check for conversion warnings
                    if result.messages:
                        _log(f"DOCX conversion warnings: {len(result.messages)}", "debug")

                    _log(f"Converted DOCX to HTML: {len(html_content):,} chars", "debug")
                except ImportError:
                    _log("mammoth library not installed, cannot convert DOCX", "error")
                    return None
                except Exception as e:
                    _log(f"Error converting DOCX to HTML: {e}", "error")
                    return None
            else:
                # Assume HTML
                html_content = response.text

            # Extract text
            from .document_parser import extract_text_from_html

            text_content = extract_text_from_html(html_content)

            # Quality checks for downloaded content
            if not text_content or len(text_content) < 500:
                _log("Downloaded document too short or empty", "warning")
                return None

            # Check if we accidentally got a PDF instead of HTML
            # (happens if URL normalization failed and we requested :PDF instead of :HTML)
            if html_content.startswith("%PDF") or b"%PDF" in response.content[:100]:
                _log(
                    f"ERROR: Downloaded binary PDF instead of HTML. URL: {url}. "
                    "This indicates URL normalization failed. Fix the URL to use :HTML format.",
                    "error",
                )
                return None

            # Check for common error pages
            if any(
                marker in html_content.lower()[:2000]
                for marker in ["404 not found", "page not found", "error occurred", "access denied"]
            ):
                _log(f"Downloaded error page instead of document: {url}", "warning")
                return None

            _log(
                f"Downloaded: {len(html_content):,} chars HTML, {len(text_content):,} chars text",
                "debug",
            )

            # Write to cache (if caching enabled)
            if cache_dir:
                from pathlib import Path

                from .document_cache import write_to_cache

                cache_path = get_cache_path(url, Path(cache_dir))
                write_to_cache(cache_path, html_content, text_content, url, logger)

            return (html_content, text_content)

        except Exception as e:
            _log(f"Error downloading document: {e}", "error")
            return None


def construct_eurlex_html_url(com_ref: str) -> Optional[str]:
    """Construct EUR-Lex HTML URL from COM reference.

    Args:
        com_ref: Reference like "COM(2025)142" or "SWD(2025)565"

    Returns:
        EUR-Lex HTML URL or None
    """
    # Parse reference: COM(2025)142
    match = re.match(r"(COM|SWD|SEC)\((\d{4})\)(\d+)", com_ref, re.IGNORECASE)
    if not match:
        return None

    doc_type = match.group(1).upper()
    year = match.group(2)
    number = match.group(3).zfill(4)  # Pad to 4 digits

    return f"https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri={doc_type}:{year}:{number}:FIN:EN:HTML"


def get_document_urls_from_procedure(procedure: Dict[str, Any]) -> Dict[str, str]:
    """Extract all document URLs from a procedure.

    Args:
        procedure: Procedure dict with events

    Returns:
        Dict mapping document_reference -> URL
    """
    urls = {}

    for event in procedure.get("events", []):
        for doc in event.get("documents", []):
            doc_id = doc.get("id", "")  # Documents use 'id' not 'reference'
            doc_url = doc.get("url", "")

            if doc_id and doc_url:
                urls[doc_id] = doc_url
            elif doc_id:
                # Try to construct URL
                constructed_url = construct_eurlex_html_url(doc_id)
                if constructed_url:
                    urls[doc_id] = constructed_url

    return urls


def extract_directive_reference_from_text(text: str) -> Optional[str]:
    """Extract directive reference from text (e.g., 'Directive 2024/1234/EU').

    Args:
        text: Document text

    Returns:
        Directive reference or None
    """
    # Match patterns like "Directive 2024/1234/EU" or "Regulation (EU) 2024/1234"
    patterns = [
        r"Directive\s+(\d{4}/\d+(?:/EU)?)",
        r"Regulation\s+\(EU\)\s+(\d{4}/\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_title_from_url(url: str) -> str:
    """Extract document title from URL for logging/display."""
    # Extract COM/SWD reference from URL
    match = re.search(r"uri=(COM|SWD|SEC):(\d{4}):(\d+)", url, re.IGNORECASE)
    if match:
        doc_type = match.group(1).upper()
        year = match.group(2)
        number = match.group(3)
        return f"{doc_type}({year}){number}"

    # Fallback to last part of URL
    return url.split("/")[-1]
