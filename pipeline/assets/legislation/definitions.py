"""Legislation Assets - Bronze → Silver → Diamond.

OEIL scraping + v2 API enrichment → Silver transformation → Upload to Supabase.

Bronze also includes amendment and document scrapers that pull PDFs from
europarl.europa.eu/doceo/ and store parsed content in Supabase.
"""

from typing import Any, Dict, List

from dagster import AssetExecutionContext, AssetIn, Config, asset

from pipeline.partitions.definitions import weekly_partitions
from pipeline.resources.supabase import SupabaseResource

from .bronze import eu_legislation_bronze
from .silver import eu_legislation_silver


# ---------------------------------------------------------------------------
# Config classes for scraper assets
# ---------------------------------------------------------------------------


class AmendmentScraperConfig(Config):
    """Configuration for the amendment scraper asset."""

    procedure_id: str | None = None
    """Single procedure reference, e.g. '2021/0106(COD)'. None = all COD procedures."""

    skip_existing: bool = True
    """Skip procedures that already have amendments in Supabase."""


class DocumentScraperConfig(Config):
    """Configuration for the document scraper asset."""

    procedure_id: str | None = None
    """Single procedure reference. None = all COD procedures."""

    skip_existing: bool = True
    """Skip procedures that already have documents in Supabase."""


# ---------------------------------------------------------------------------
# Bronze: Amendment scraper
# ---------------------------------------------------------------------------


@asset(
    name="eu_amendments_bronze",
    group_name="eu_bronze",
    compute_kind="scraper",
    required_resource_keys={"supabase", "http_client"},
    description=(
        "Scrape amendment PDFs from OEIL (europarl.europa.eu), parse with pdftotext, "
        "and upsert to the procedure_amendments table."
    ),
)
def eu_amendments_bronze(context: AssetExecutionContext, config: AmendmentScraperConfig):
    from .amendment_scraper import (
        fetch_all_cod_procedures,
        fetch_procedures_with_amendments,
        scrape_procedure_amendments,
    )

    supabase: SupabaseResource = context.resources.supabase
    http_client = context.resources.http_client
    client = supabase.get_client()
    session = http_client.session

    if config.procedure_id:
        procedures = [config.procedure_id]
        context.log.info(f"Single-procedure mode: {config.procedure_id}")
    else:
        procedures = fetch_all_cod_procedures(client)
        context.log.info(f"Batch mode: {len(procedures)} COD procedures found")
        if config.skip_existing:
            existing = fetch_procedures_with_amendments(client)
            before = len(procedures)
            procedures = [p for p in procedures if p not in existing]
            context.log.info(f"Skipping {before - len(procedures)} with existing amendments")

    succeeded = 0
    failed = 0
    no_amendments = 0
    total_amendments = 0

    for i, proc_id in enumerate(procedures, 1):
        context.log.info(f"[{i}/{len(procedures)}] {proc_id}")
        try:
            count = scrape_procedure_amendments(
                procedure_id=proc_id, client=client, session=session, logger=context.log,
            )
            if count and count > 0:
                total_amendments += count
                succeeded += 1
            else:
                no_amendments += 1
        except Exception as exc:
            context.log.error(f"Failed for {proc_id}: {exc}")
            failed += 1

    context.add_output_metadata({
        "procedures_with_amendments": succeeded,
        "procedures_no_amendments": no_amendments,
        "procedures_failed": failed,
        "total_amendments_uploaded": total_amendments,
    })
    return {"total_amendments": total_amendments, "succeeded": succeeded, "failed": failed}


# ---------------------------------------------------------------------------
# Bronze: Document scraper
# ---------------------------------------------------------------------------


@asset(
    name="eu_documents_bronze",
    group_name="eu_bronze",
    compute_kind="scraper",
    required_resource_keys={"supabase", "http_client"},
    description=(
        "Scrape non-amendment legislative documents (draft reports, opinions, committee "
        "reports, texts adopted, COM proposals) from OEIL, extract text with pdftotext, "
        "and upsert to the procedure_documents table."
    ),
)
def eu_documents_bronze(context: AssetExecutionContext, config: DocumentScraperConfig):
    from .document_scraper import (
        fetch_all_cod_procedures,
        fetch_procedures_with_documents,
        scrape_procedure_documents,
    )

    supabase: SupabaseResource = context.resources.supabase
    http_client = context.resources.http_client
    client = supabase.get_client()
    session = http_client.session

    if config.procedure_id:
        procedures = [config.procedure_id]
        context.log.info(f"Single-procedure mode: {config.procedure_id}")
    else:
        procedures = fetch_all_cod_procedures(client)
        context.log.info(f"Batch mode: {len(procedures)} COD procedures found")
        if config.skip_existing:
            existing = fetch_procedures_with_documents(client)
            before = len(procedures)
            procedures = [p for p in procedures if p not in existing]
            context.log.info(f"Skipping {before - len(procedures)} with existing documents")

    total_uploaded = 0
    succeeded = 0
    failed = 0
    no_docs = 0

    for i, proc_id in enumerate(procedures, 1):
        context.log.info(f"[{i}/{len(procedures)}] {proc_id}")
        try:
            type_counts = scrape_procedure_documents(
                procedure_id=proc_id, client=client, session=session, logger=context.log,
            )
            proc_total = type_counts.get("total", 0)
            total_uploaded += proc_total
            if proc_total > 0:
                succeeded += 1
            else:
                no_docs += 1
        except Exception as exc:
            context.log.error(f"Failed for {proc_id}: {exc}")
            failed += 1

    context.add_output_metadata({
        "procedures_with_documents": succeeded,
        "procedures_no_documents": no_docs,
        "procedures_failed": failed,
        "total_documents_uploaded": total_uploaded,
    })
    return {"total_uploaded": total_uploaded, "succeeded": succeeded, "failed": failed}


# ---------------------------------------------------------------------------
# Diamond: Upload legislation
# ---------------------------------------------------------------------------


@asset(
    name="eu_legislation_diamond",
    group_name="eu_diamond",
    description=(
        "Upsert legislative procedure records to the Supabase procedures table. Generates "
        "deterministic IDs from procedure references, formats timeline events, and writes "
        "with upsert semantics for idempotent re-runs."
    ),
    compute_kind="upload",
    partitions_def=weekly_partitions,
    ins={"silver_data": AssetIn("eu_legislation_silver")},
)
def eu_legislation_diamond(
    context: AssetExecutionContext,
    supabase: SupabaseResource,
    silver_data: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Diamond layer: Upload procedures to Supabase."""
    if not silver_data:
        context.log.info("No data to upload")
        return {"success": 0, "failed": 0}

    from .diamond import prepare_procedure_records, upload_procedures

    records = prepare_procedure_records(silver_data)
    result = upload_procedures(
        procedures=records,
        supabase_resource=supabase,
        logger=context.log,
    )

    context.add_output_metadata({
        "uploaded": result.get("success", 0),
        "failed": result.get("failed", 0),
    })
    return result


legislation_assets = [
    eu_legislation_bronze,
    eu_legislation_silver,
    eu_legislation_diamond,
    eu_amendments_bronze,
    eu_documents_bronze,
]
