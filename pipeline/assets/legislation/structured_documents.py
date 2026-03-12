"""Silver layer: Extract structured content from legislative documents.

Parses HTML/text from EUR-Lex to extract:
- Recitals (with numbering and order)
- Articles (with titles, full text, section/chapter grouping)
- Hierarchy (sections → chapters → articles)

This is a SILVER enrichment step (no AI) that Gold layer can reuse.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .amendment_parser import parse_amendment_document
from .document_parser import parse_legislative_structure, sanitize_text
from .document_utils import download_document
from .legal_text_parser import parse_legal_document


def extract_article_number(target: str) -> str:
    """Extract article number from target string.

    Examples:
        "Article 5 - paragraph 2" -> "Article 5"
        "Recital 12" -> "Recital 12"
        "Article 28a (new)" -> "Article 28a"

    Args:
        target: Target article/recital string

    Returns:
        Normalized article/recital identifier
    """
    if not target:
        return "Unknown"

    # Match "Article X" or "Recital X" (with optional letter)
    match = re.match(r"(Article|Recital)\s+(\d+[a-z]?)", target, re.IGNORECASE)
    if match:
        return f"{match.group(1).capitalize()} {match.group(2)}"

    return target.split("-")[0].strip() if "-" in target else target.strip()


def enrich_event_with_interpretation(event: Dict[str, Any], logger: Optional[Any] = None) -> None:
    """Add _document_interpretation to events with key documents.

    Aggregates structured facts from already-parsed data:
    - Amendments summary (from _amendments)
    - Committee opinions (from _amendments.opinions)
    - Vote results (future: when available)

    Modifies event in-place. Applies to ANY event that has _amendments data.

    Args:
        event: Event dictionary with potential _amendments data
        logger: Optional logger
    """
    # Enrich ANY event that has parsed amendments
    if event.get("_amendments"):
        amendments = event["_amendments"].get("amendments", [])
        opinions = event["_amendments"].get("opinions", [])

        if not amendments:
            return

        # Aggregate by target article
        by_target = {}
        for amend in amendments:
            target = amend.get("target_article", "Unknown")
            # Normalize: "Article 5 - paragraph 2" -> "Article 5"
            article = extract_article_number(target)
            by_target[article] = by_target.get(article, 0) + 1

        # Aggregate by committee
        by_committee = {}
        for amend in amendments:
            committee = amend.get("committee", "Unknown")
            if committee and committee != "Unknown":
                by_committee[committee] = by_committee.get(committee, 0) + 1

        # Get hotspots (top 5 most amended articles)
        hotspots = sorted(by_target.items(), key=lambda x: x[1], reverse=True)[:5]

        # Build opinions consolidated list
        opinions_consolidated = []
        for op in opinions:
            committee_code = op.get("committee_code") or op.get("committee")
            if committee_code:
                # Count amendments from this opinion committee
                amend_count = sum(1 for a in amendments if a.get("committee") == committee_code)
                opinions_consolidated.append(
                    {
                        "committee": op.get("committee"),
                        "committee_code": committee_code,
                        "rapporteur": op.get("rapporteur"),
                        "amendments_from_opinion": amend_count,
                    }
                )

        event["_document_interpretation"] = {
            "amendments_summary": {
                "total": len(amendments),
                "by_target": by_target,
                "by_committee": by_committee,
                "hotspots": [article for article, count in hotspots],
            },
            "opinions_consolidated": opinions_consolidated,
        }

        if logger:
            logger.debug(
                f"Enriched {event.get('event_id')} with interpretation: "
                f"{len(amendments)} amendments, {len(opinions)} opinions, "
                f"{len(hotspots)} hotspot articles"
            )


def load_previous_silver_data(
    procedure_id: str, current_partition: str, logger: Optional[Any] = None
) -> Dict[str, Any] | None:
    """Load most recent Silver data for a procedure.

    Args:
        procedure_id: Procedure ID (e.g., "2025/0424(COD)")
        current_partition: Current partition key (e.g., "2025-01-19")
        logger: Optional logger

    Returns:
        Previous Silver data dict or None if not found
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    silver_dir = Path("data/eu/legislation/silver")
    if not silver_dir.exists():
        return None

    current_date = datetime.strptime(current_partition, "%Y-%m-%d")

    # Search Silver files in reverse chronological order
    for silver_file in sorted(silver_dir.glob("legislation_*.json"), reverse=True):
        file_date_str = silver_file.stem.replace("legislation_", "")
        try:
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Skip current and future partitions
        if file_date >= current_date:
            continue

        # Load and search for procedure
        try:
            with open(silver_file, encoding="utf-8") as f:
                data = json.load(f)
                for proc in data:
                    if proc.get("id") == procedure_id:
                        _log(f"Found {procedure_id} in Silver partition {file_date_str}", "debug")
                        return proc
        except Exception as e:
            _log(f"Error reading {silver_file}: {e}", "warning")
            continue

    return None


def extract_structure_from_url(
    url: str,
    logger: Optional[Any] = None,
    fallback_urls: Optional[list[str]] = None,
    http_client: Optional[Any] = None,
    cache_dir: Optional[Any] = None,
    cache_ttl_days: int = 7,
    selenium_resource: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract structured recitals and articles from a legislative document URL.

    Uses both HTML structure parsing and text-based parsing as fallback.
    If primary URL fails (e.g., HTTP 202), tries fallback URLs.

    Args:
        url: Primary document URL
        logger: Optional logger
        fallback_urls: Alternative URLs to try if primary fails
        http_client: Optional HttpClientResource for rate-limited downloads
        cache_dir: Optional cache directory path (enables caching if provided)
        cache_ttl_days: Cache time-to-live in days (default 7)
        selenium_resource: Optional SeleniumResource for WAF bypass fallback

    Returns:
        Dict with 'recitals', 'articles', and 'hierarchy'
    """
    if not url:
        return {"recitals": [], "articles": [], "hierarchy": {}}

    urls_to_try = [url]
    if fallback_urls:
        urls_to_try.extend(fallback_urls)

    for attempt_url in urls_to_try:
        try:
            if logger:
                if attempt_url == url:
                    logger.info(f"📄 Extracting structure from document: {attempt_url}")
                else:
                    logger.info(f"📄 Trying fallback URL: {attempt_url}")

            # 3 retries allows up to 40s for EUR-Lex document generation
            result = download_document(
                attempt_url,
                logger,
                max_retries=3,
                http_client=http_client,
                cache_dir=cache_dir,
                cache_ttl_days=cache_ttl_days,
                selenium_resource=selenium_resource,
            )
            if not result:
                continue  # Try next URL

            html_content, text_content = result

            # Try HTML structure parsing first
            structure = parse_legislative_structure(html_content)
            html_recitals = structure.get("recitals", [])
            html_articles = structure.get("articles", [])
            # Hierarchy is now metadata-only (no nested articles)
            hierarchy = structure.get("hierarchy", {})
            # Add totals to hierarchy if not present
            hierarchy.update(
                {
                    "total_articles": structure.get("total_articles", 0),
                    "total_sections": structure.get("total_sections", 0),
                    "total_chapters": structure.get("total_chapters", 0),
                    "total_annexes": structure.get("total_annexes", 0),
                    "has_explanatory_memo": structure.get("has_explanatory_memo", False),
                }
            )

            # If HTML parsing didn't find anything, try text-based parsing
            if not html_recitals and not html_articles:
                if logger:
                    logger.info("HTML parsing yielded no results, using text-based fallback")

                parsed_doc = parse_legal_document(text_content)

                text_recitals = [
                    {
                        "recital_number": int(rec.number) if rec.number else 0,
                        "text": sanitize_text(rec.content),
                        "order_index": int(rec.number) if rec.number else 0,
                    }
                    for rec in parsed_doc.recitals
                ]

                text_articles = [
                    {
                        "article_number": art.number,
                        "title": sanitize_text(art.title) if art.title else "",
                        "full_text": sanitize_text(art.content),
                        "section": None,
                        "chapter": None,
                    }
                    for art in parsed_doc.articles
                ]

                if logger:
                    logger.info(
                        f"✅ Text fallback extracted {len(text_recitals)} recitals, "
                        f"{len(text_articles)} articles"
                    )

                return {
                    "recitals": text_recitals,
                    "articles": text_articles,
                    "hierarchy": {
                        "total_recitals": len(text_recitals),
                        "total_articles": len(text_articles),
                    },
                }

            if logger:
                logger.info(
                    f"✅ HTML parsing extracted {len(html_recitals)} recitals, "
                    f"{len(html_articles)} articles"
                )

            return {
                "recitals": html_recitals,
                "articles": html_articles,
                "hierarchy": hierarchy,
            }

        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract structure from {attempt_url}: {e}")
            # Continue to next URL
            continue

    # All URLs failed - return None to signal failure
    # Caller should preserve existing data rather than overwrite with empty structure
    if logger:
        logger.warning("All URLs failed for proposal extraction")
    return None


def enrich_events_with_document_structures(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
    http_client: Optional[Any] = None,
    partition_key: Optional[str] = None,
    cache_dir: Optional[Any] = None,
    cache_ttl_days: int = 7,
    selenium_resource: Optional[Any] = None,
) -> None:
    """Enrich events with parsed document structures (in-place, with caching).

    Anchors document structures to their triggering events:
    - Proposal → PROPOSAL_PUBLICATION event (COM documents)
    - Amendments → ANY event with AMENDMENT_LIST documents (committee, opinion, plenary)
    - Final text → PUBLICATION_OFFICIAL_JOURNAL or SIGNATURE event

    Amendment extraction uses v2 API work_type to ensure completeness:
    - Committee amendments (INTA-AM-*, AGRI-AM-*, etc.)
    - Opinion amendments (embedded in A10 reports)
    - Plenary amendments (A-10-*-AM-*)

    Args:
        procedure: Legislative procedure dict with events (modified in-place)
        logger: Optional Dagster logger
        http_client: Optional HttpClientResource for rate-limited downloads
        partition_key: Optional partition key for loading previous data (caching)
        cache_dir: Optional cache directory path (enables caching if provided)
        cache_ttl_days: Cache time-to-live in days (default 7)
    """
    # Load previous Silver data for caching
    previous_silver = None
    if partition_key:
        previous_silver = load_previous_silver_data(procedure.get("id"), partition_key, logger)

    # Build lookup of previous event data by event_id
    previous_events = {}
    if previous_silver:
        for event in previous_silver.get("events", []):
            event_id = event.get("event_id")
            if event_id:
                previous_events[event_id] = event

    # Get URLs for proposal and final text (from procedure-level fields)
    proposal_url = procedure.get("eurlex_proposal_url") or procedure.get("commission_document")
    final_text_url = procedure.get("eurlex_final_act_url")

    for event in procedure.get("events", []):
        activity_type = event.get("activity_type", "")
        event_id = event.get("event_id")
        prev_event = previous_events.get(event_id, {})

        # 1. Anchor PROPOSAL to PROPOSAL_PUBLICATION event (with caching)
        if activity_type == "PROPOSAL_PUBLICATION" and proposal_url:
            # Check if already extracted in previous partition
            if prev_event.get("_proposal"):
                if logger:
                    logger.debug("✓ Proposal already extracted, copying from previous partition")
                event["_proposal"] = prev_event["_proposal"]
                continue

            # Not cached - download and process
            if logger:
                logger.info(
                    f"📝 Extracting proposal structure (first time) for event {event.get('event_date')}"
                )

            # Collect fallback URLs from event documents (in case EUR-Lex HTML isn't ready)
            fallback_urls = []
            for doc in event.get("documents", []):
                doc_url = doc.get("url", "")
                if doc_url and doc_url != proposal_url:
                    # Convert PDF to HTML for EUR-Lex URLs
                    if "eur-lex.europa.eu" in doc_url and "/PDF/" in doc_url:
                        doc_url = doc_url.replace("/PDF/", "/HTML/")
                    fallback_urls.append(doc_url)

            structure = extract_structure_from_url(
                proposal_url,
                logger,
                fallback_urls,
                http_client,
                cache_dir=cache_dir,
                cache_ttl_days=cache_ttl_days,
                selenium_resource=selenium_resource,
            )

            # Preserve existing silver layer data if download fails (structure is None)
            if structure is None:
                # Download failed - preserve previous data if available
                if prev_event.get("_proposal"):
                    if logger:
                        logger.warning(
                            f"⚠️ Download failed for {proposal_url}, preserving previous partition data"
                        )
                    event["_proposal"] = prev_event["_proposal"]
                else:
                    # No previous data and download failed (likely EUR-Lex bot detection).
                    # Log a warning but continue — Gold handles missing _proposal gracefully
                    # by skipping proposal-based analysis steps.
                    if logger:
                        logger.warning(
                            f"⚠️ Could not download proposal document: {proposal_url}. "
                            "No previous partition data available. "
                            "Procedure will continue without proposal structure — "
                            "this may be due to EUR-Lex bot detection (AWS WAF). "
                            "Re-run or seed cache manually (seed_document_cache.py) to recover."
                        )
            else:
                # Successfully extracted - update with new data
                event["_proposal"] = {
                    **structure,
                    "url": proposal_url,
                }

        # 2. Extract ALL AMENDMENT documents across all events (with caching)
        # Get previously processed document IDs
        prev_doc_ids = set()
        if prev_event.get("_amendments"):
            prev_doc_ids = {
                a.get("document_id") for a in prev_event["_amendments"].get("amendments", [])
            }

        # Find new amendment documents to process
        new_amendment_docs = []
        for doc in event.get("documents", []):
            doc_id = doc.get("id", "")
            doc_url = doc.get("url", "")
            work_type = doc.get("_v2_work_type", "")

            if not doc_url:
                continue

            # Check for amendment documents by work_type OR doc ID pattern
            is_amendment_list = "AMENDMENT_LIST" in work_type
            is_amendment_doc = any(pattern in doc_id for pattern in ["-AM-", "A10-", "A9-", "PE-"])

            if is_amendment_list or (is_amendment_doc and activity_type == "COMMITTEE_REPORT"):
                # Skip if already processed
                if doc_id in prev_doc_ids:
                    if logger:
                        logger.debug(f"✓ Amendment {doc_id} already extracted, skipping")
                    continue
                new_amendment_docs.append((doc_id, doc_url, work_type))

        # Carry forward previous amendments + add new ones
        all_amendments = []
        if prev_event.get("_amendments"):
            all_amendments.extend(prev_event["_amendments"].get("amendments", []))

        # Parse new amendment documents only
        if new_amendment_docs:
            for doc_id, doc_url, work_type in new_amendment_docs:
                if logger:
                    logger.info(
                        f"📄 Extracting amendments from {doc_id} (first time) ({work_type or 'COMMITTEE_REPORT'})"
                    )

                # For A10/A9 reports from europarl.europa.eu, always use DOCX
                actual_url = doc_url
                if "europarl.europa.eu/doceo/document" in doc_url and "_EN.html" in doc_url:
                    actual_url = doc_url.replace("_EN.html", "_EN.docx")
                    if logger:
                        logger.debug(f"Using DOCX version: {actual_url}")

                # Download the document
                download_result = download_document(actual_url, logger, http_client=http_client)
                if download_result:
                    html_content, _ = download_result

                    # Parse amendments
                    parsed_result = parse_amendment_document(html_content)

                    if logger:
                        parsed_total = parsed_result.get("total_amendments", 0)
                        logger.info(f"✅ Extracted {parsed_total} amendments from {doc_id}")

                    # Add document metadata to each amendment
                    amendments_list = parsed_result.get("all_amendments", [])
                    for amendment in amendments_list:
                        amendment["document_id"] = doc_id
                        amendment["document_url"] = doc_url
                        amendment["work_type"] = work_type or "COMMITTEE_REPORT"

                    all_amendments.extend(amendments_list)
                else:
                    if logger:
                        logger.warning(f"Failed to download amendment document: {doc_id}")

        # Store all amendments (previous + new)
        if all_amendments:
            event["_amendments"] = {
                "amendments": all_amendments,
                "total_amendments": len(all_amendments),
            }

            # NEW: Enrich event with structured interpretation
            enrich_event_with_interpretation(event, logger)

        # 3. Anchor FINAL TEXT to PUBLICATION_OFFICIAL_JOURNAL or SIGNATURE event (with caching)
        if activity_type in ["PUBLICATION_OFFICIAL_JOURNAL", "SIGNATURE"] and final_text_url:
            # Only attach to first matching event (prefer PUBLICATION_OFFICIAL_JOURNAL)
            if any(e.get("_final_text") for e in procedure.get("events", [])):
                continue

            # Check if already extracted in previous partition
            if prev_event.get("_final_text"):
                if logger:
                    logger.debug("✓ Final text already extracted, copying from previous partition")
                event["_final_text"] = prev_event["_final_text"]
                continue

            # Not cached - download and process
            if logger:
                logger.info(
                    f"📜 Extracting final text structure (first time) for event {event.get('event_date')}"
                )

            structure = extract_structure_from_url(
                final_text_url, logger, http_client=http_client, selenium_resource=selenium_resource
            )
            if structure:
                event["_final_text"] = {
                    **structure,
                    "url": final_text_url,
                    "celex_number": procedure.get("celex_number"),
                }
            elif prev_event.get("_final_text"):
                # Download failed, preserve previous
                event["_final_text"] = prev_event["_final_text"]


def extract_document_structures(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """DEPRECATED: Use enrich_events_with_document_structures() instead.

    This function is kept for backwards compatibility but returns empty structures.
    Document structures are now anchored directly to events.

    Args:
        procedure: Bronze/Silver procedure data
        logger: Optional logger

    Returns:
        Empty dict (structures are in events)
    """
    # Document structures are now in events, not in a separate dict
    return {}
