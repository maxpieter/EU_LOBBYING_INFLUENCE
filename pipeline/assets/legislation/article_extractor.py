"""Extract and structure articles/recitals from legislative documents.

UPDATED: Now uses pre-parsed Silver layer data instead of re-downloading documents.

Silver layer provides:
- events[]._proposal: recitals[], articles[]
- events[]._final_text: recitals[], articles[], celex_number

This module extracts that data for the procedure_articles table.
"""

from typing import Any, Dict, List, Optional


def extract_articles_from_silver(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract all articles/recitals from Silver's pre-parsed event data.

    Processes:
    - events[]._proposal -> version='proposal'
    - events[]._final_text -> version='adopted'

    Args:
        procedure: Procedure dict with events containing _proposal, _final_text
        logger: Optional logger

    Returns:
        List of article dicts ready for database insertion
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    procedure_id = procedure.get("id", "unknown")
    articles: List[Dict[str, Any]] = []

    for event in procedure.get("events", []):
        # Extract from proposal
        if "_proposal" in event:
            proposal = event["_proposal"]
            doc_id = None
            for doc in event.get("documents", []):
                if doc.get("id", "").startswith("COM("):
                    doc_id = doc["id"]
                    break
            if not doc_id:
                doc_id = "proposal"

            articles.extend(
                _extract_from_structure(
                    structure=proposal,
                    procedure_id=procedure_id,
                    document_source=doc_id,
                    document_version="proposal",
                    logger=logger,
                )
            )

        # Extract from final text
        if "_final_text" in event:
            final_text = event["_final_text"]
            doc_id = final_text.get("celex_number") or "final"

            articles.extend(
                _extract_from_structure(
                    structure=final_text,
                    procedure_id=procedure_id,
                    document_source=doc_id,
                    document_version="adopted",
                    logger=logger,
                )
            )

    _log(f"Extracted {len(articles)} total elements from {procedure_id}")
    return articles


def _extract_from_structure(
    structure: Dict[str, Any],
    procedure_id: str,
    document_source: str,
    document_version: str,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract articles/recitals from a pre-parsed structure.

    Args:
        structure: Dict with 'recitals' and 'articles' arrays
        procedure_id: Procedure ID
        document_source: Document reference
        document_version: "proposal" or "adopted"
        logger: Optional logger

    Returns:
        List of article dicts for database
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    articles: List[Dict[str, Any]] = []
    sort_order = 0

    # Extract recitals
    for recital in structure.get("recitals", []):
        text = recital.get("text", "")
        if text and len(text.strip()) > 10:
            articles.append(
                {
                    "procedure_id": procedure_id,
                    "element_type": "recital",
                    "element_number": str(recital.get("recital_number", "")),
                    "title": None,
                    "content": text.strip()[:5000],
                    "document_source": document_source,
                    "document_version": document_version,
                    "sort_order": recital.get("order_index", sort_order),
                }
            )
            sort_order += 1

    # Extract articles
    for article in structure.get("articles", []):
        text = article.get("full_text", "")
        if text and len(text.strip()) > 10:
            articles.append(
                {
                    "procedure_id": procedure_id,
                    "element_type": "article",
                    "element_number": str(article.get("article_number", "")),
                    "title": article.get("title"),
                    "content": text.strip()[:10000],
                    "document_source": document_source,
                    "document_version": document_version,
                    "sort_order": sort_order,
                }
            )
            sort_order += 1

    _log(
        f"Extracted from {document_source}: "
        f"{sum(1 for a in articles if a['element_type'] == 'recital')} recitals, "
        f"{sum(1 for a in articles if a['element_type'] == 'article')} articles"
    )

    return articles


def extract_proposal_articles(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract articles only from the proposal (from Silver's _proposal data).

    This is the most common use case - extracting the proposal structure
    for search and reference.

    Args:
        procedure: Procedure dict with events containing _proposal
        logger: Optional logger

    Returns:
        List of article dicts from the proposal only
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    procedure_id = procedure.get("id", "unknown")

    # Find _proposal in events
    for event in procedure.get("events", []):
        if "_proposal" in event:
            proposal = event["_proposal"]

            # Get document ID
            doc_id = None
            for doc in event.get("documents", []):
                if doc.get("id", "").startswith("COM("):
                    doc_id = doc["id"]
                    break
            if not doc_id:
                doc_id = "proposal"

            return _extract_from_structure(
                structure=proposal,
                procedure_id=procedure_id,
                document_source=doc_id,
                document_version="proposal",
                logger=logger,
            )

    _log(f"No proposal structure found for {procedure_id}", "debug")
    return []


# Legacy function names for backward compatibility
def extract_articles_from_procedure(
    procedure: Dict[str, Any],
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Legacy: Use extract_articles_from_silver instead."""
    return extract_articles_from_silver(procedure, logger)
