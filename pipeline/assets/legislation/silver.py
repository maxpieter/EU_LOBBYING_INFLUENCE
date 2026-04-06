"""Silver layer: Data merging, enrichment, and document structure extraction."""

from typing import Any, Dict, List

from dagster import AssetExecutionContext, asset

from pipeline.models.legislation import Procedure
from pipeline.partitions.definitions import weekly_partitions
from pipeline.resources.http_client import HttpClientResource
from pipeline.resources.selenium import SeleniumResource


def normalize_document_url_to_html(url: str) -> str:
    """Convert document URLs to HTML format.

    Converts:
    - Europarl RegData COM PDFs to EUR-Lex HTML (using COM:YYYY:NNNN:FIN format)
    - Europarl Doceo PDFs to HTML
    - EUR-Lex PDF URLs to HTML
    - Leaves HTML and DOCX URLs unchanged (DOCX is preferred for A-10 documents)

    Args:
        url: Document URL (PDF, HTML, or DOCX)

    Returns:
        HTML or DOCX URL

    Examples:
        >>> normalize_document_url_to_html("https://www.europarl.europa.eu/RegData/.../com/2025/0784/...")
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=COM:2025:784:FIN"
    """
    if not url:
        return url

    # DOCX files are preferred for A-10 documents - leave them unchanged
    if url.endswith(".docx"):
        return url

    # Europarl RegData COM documents (PDF -> EUR-Lex HTML with CELEX format)
    # Example: https://www.europarl.europa.eu/RegData/docs_autres_institutions/commission_europeenne/com/2025/0784/COM_COM(2025)0784_EN.pdf
    # Convert to CELEX format (canonical per robots.txt): https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52025PC0784
    if "europarl.europa.eu/RegData" in url and "/com/" in url.lower() and url.endswith(".pdf"):
        import re

        # Extract year and number from URL path: .../com/YYYY/NNNN/...
        match = re.search(r"/com/(\d{4})/(\d+)/", url, re.IGNORECASE)
        if match:
            year = match.group(1)
            number = match.group(2).lstrip("0")  # Remove leading zeros
            # Use CELEX format: 5YYYYPCNNNN (5=proposal, PC=Commission proposal)
            # This is the canonical format per EUR-Lex robots.txt
            celex = f"5{year}PC{number.zfill(4)}"
            return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
        # If extraction fails, return original URL
        return url

    # Europarl Doceo documents (PDF -> HTML)
    # Example: https://www.europarl.europa.eu/doceo/document/TA-10-2025-0295_EN.pdf
    if "europarl.europa.eu/doceo" in url and url.endswith(".pdf"):
        return url.replace(".pdf", ".html")

    # EUR-Lex documents (PDF -> HTML)
    # Example: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52025PC0784
    if "eur-lex.europa.eu" in url and "/PDF/" in url:
        return url.replace("/PDF/", "/HTML/")

    # EUR-Lex SWD/SEC documents using LexUriServ (PDF -> HTML)
    # Example: https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=SWD:2025:0565:FIN:EN:PDF
    # Convert to: https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=SWD:2025:0565:FIN:EN:HTML
    if "eur-lex.europa.eu/LexUriServ" in url and ":PDF" in url:
        return url.replace(":PDF", ":HTML")

    # Direct PDF file links on EUR-Lex or Europarl (try replacing .pdf extension)
    # This catches any remaining PDF URLs that weren't caught by specific patterns
    if url.endswith(".pdf") and ("eur-lex.europa.eu" in url or "europarl.europa.eu" in url):
        return url.replace(".pdf", ".html")

    # Already HTML or unknown format
    return url


@asset(
    group_name="eu_silver",
    compute_kind="transformation",
    partitions_def=weekly_partitions,
    description=(
        "Merge OEIL and v2 API data, normalise document URLs (PDF→HTML), deduplicate events, "
        "and enrich procedure records with computed fields. Prepares clean procedure dicts "
        "for diamond-layer upload."
    ),
)
def eu_legislation_silver(
    context: AssetExecutionContext,
    eu_legislation_bronze: List[Dict[str, Any]],
    http_client: HttpClientResource,
    selenium: SeleniumResource,
) -> List[Dict[str, Any]]:
    """Silver layer: Merge OEIL+v2 data, add computed fields, normalize data, and extract document structures.

    Transforms Bronze data with:
    1. Merge OEIL events + v2 events into unified timeline
    2. Merge OEIL actors + v2 participations into unified actor list
    3. Momentum computation based on last activity date
    4. Date type conversions
    5. URL normalization (PDF -> HTML)
    6. Document structure extraction (recitals, articles, hierarchy)
    7. Stage/status inference from events
    8. Event provenance (source field)
    9. Actor name resolution status
    10. Final validation

    Note: All data extraction (including OEIL HTML scraping) happens in Bronze layer.
    Silver layer only performs transformations and computations.
    """
    from pipeline.config.models import DocumentCacheConfig

    from .silver_finalization import finalize_silver_enrichment
    from .structured_documents import enrich_events_with_document_structures
    from .v2_enrichment import enrich_procedure_with_v2_data

    context.log.info(f"Processing {len(eu_legislation_bronze)} legislation items")

    # STEP 1: Merge OEIL and v2 data (events + actors/participations)
    context.log.info("Merging OEIL and v2 API data...")
    for item in eu_legislation_bronze:
        enrich_procedure_with_v2_data(item, logger=context.log)

    context.log.info("Completed OEIL + v2 data merge")

    # STEP 2: Normalize and enrich
    enriched = []
    for item in eu_legislation_bronze:
        # Convert to Procedure model first (validates structure)
        # Note: stage field is now populated from v2 API in bronze layer
        proc = Procedure(**item)

        # Convert back to dict for modifications
        proc_dict = proc.model_dump()

        # Normalize document URLs to HTML format (events and background_documents)
        # MUST do this on proc_dict after model_dump() to ensure modifications persist
        for event in proc_dict.get("events", []):
            for doc in event.get("documents", []):
                if "url" in doc:
                    doc["url"] = normalize_document_url_to_html(doc["url"])

        # IMPORTANT: Also normalize background_documents URLs (SWD, SEC impact assessments)
        for bg_doc in proc_dict.get("background_documents", []):
            if "url" in bg_doc:
                bg_doc["url"] = normalize_document_url_to_html(bg_doc["url"])

        # Extract document structures and anchor them to events (in-place)
        # Note: proc_dict already created above with normalized URLs
        enrich_events_with_document_structures(
            proc_dict,
            logger=context.log,
            http_client=http_client,
            selenium_resource=selenium,
            partition_key=context.partition_key,
            cache_dir=DocumentCacheConfig.CACHE_DIR if DocumentCacheConfig.ENABLED else None,
            cache_ttl_days=DocumentCacheConfig.TTL_DAYS,
        )

        # STEP 3: Finalize Silver enrichment
        # - Infer stage/status from events
        # - Add event provenance
        # - Mark actor name resolution status
        finalize_silver_enrichment(proc_dict, logger=context.log)

        enriched.append(proc_dict)

    context.add_output_metadata(
        {
            "record_count": len(enriched),
            "in_progress": sum(1 for item in enriched if item.get("status") == "in_progress"),
            "completed": sum(1 for item in enriched if item.get("status") == "completed"),
            "with_proposal_events": sum(
                1 for p in enriched if any(e.get("_proposal") for e in p.get("events", []))
            ),
            "with_amendment_events": sum(
                1 for p in enriched if any(e.get("_amendments") for e in p.get("events", []))
            ),
            "with_final_text_events": sum(
                1 for p in enriched if any(e.get("_final_text") for e in p.get("events", []))
            ),
        }
    )

    return enriched
