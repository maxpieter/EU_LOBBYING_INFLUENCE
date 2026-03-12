"""Diamond layer: Upload EU legislation procedures to Supabase.

This file contains:
- Prepare procedure records for database
- Upload procedures to Supabase
- Upload article-level structure for search
- Upload amendments with MEP matching
- Upload statistics and metadata
"""

import uuid
from typing import Any, Dict, List, Optional

from dagster import AssetExecutionContext, asset

from pipeline.partitions.definitions import weekly_partitions
from pipeline.resources.supabase import SupabaseResource

from .amendment_extractor import (
    extract_amendments_from_procedures,
    strip_amendments_from_events,
)


def generate_article_id(
    procedure_id: str,
    document_version: str,
    element_type: str,
    element_number: str,
) -> str:
    """Generate deterministic UUID for procedure_articles.

    Uses UUID v5 (name-based) with consistent namespace for reproducibility.

    Args:
        procedure_id: Procedure ID (e.g., "2025/0424(COD)")
        document_version: "proposal" or "adopted"
        element_type: "recital" or "article"
        element_number: Element number (e.g., "1", "5(2)")

    Returns:
        Deterministic UUID string
    """
    # Normalize element_number (remove whitespace, make consistent)
    normalized_number = str(element_number).strip()

    # Concatenate components with separator
    components = [procedure_id, document_version, element_type, normalized_number]
    name_string = "::".join(components)

    # Generate deterministic UUID v5 (name-based)
    # Using DNS namespace as base (could use any consistent namespace)
    deterministic_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, name_string)

    return str(deterministic_uuid)


def generate_amendment_id(
    procedure_id: str,
    document_id: str,
    amendment_number: int,
) -> str:
    """Generate deterministic UUID for procedure_amendments.

    Uses UUID v5 (name-based) with composite key matching database constraint.

    CRITICAL: Must use same fields as database UNIQUE constraint:
    (procedure_id, document_id, amendment_number)

    Args:
        procedure_id: Procedure ID (e.g., "2025/0424(COD)")
        document_id: Document ID (e.g., "INTA-AM-772197")
        amendment_number: Amendment number (e.g., 1, 2, 3)

    Returns:
        Deterministic UUID string
    """
    # Concatenate components matching database constraint
    components = [procedure_id, document_id, str(amendment_number)]
    name_string = "::".join(components)

    # Generate deterministic UUID v5
    deterministic_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, name_string)

    return str(deterministic_uuid)


def prepare_procedure_record(procedure: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a procedure record for Supabase upsert.

    Pure data mapping - no business logic.
    Gold layer should ensure all data is in correct format.

    Args:
        procedure: Procedure dict from Gold layer (with AI analysis and embeddings)

    Returns:
        Dict ready for database upsert
    """
    # Build database record matching schema
    # All data should already be in correct format from Gold layer
    record = {
        # Primary identification
        "id": procedure["id"],  # Required: OEIL reference (e.g., "2024/0003(COD)")
        "process_id": procedure["process_id"],  # Required: Process ID
        # Basic information
        "title": procedure["title"],  # Required
        "description": procedure.get("description"),
        "procedure_type": procedure.get("procedure_type"),
        "policy_area": procedure.get("policy_area"),
        "status": procedure.get("status"),
        "stage": procedure.get("stage"),
        # Subjects and legal basis (arrays)
        "subjects": procedure.get("subjects", []),
        "legal_basis": procedure.get("legal_basis", []),
        # Dates
        "proposal_date": procedure.get("proposal_date"),
        "decision_date": procedure.get("decision_date"),
        "last_activity_date": procedure.get("last_activity_date"),
        # Documents and references
        "commission_document": procedure.get("commission_document"),
        "amending_acts": procedure.get("amending_acts", []),
        "background_documents": procedure.get("background_documents", []),
        "celex_number": procedure.get("celex_number"),
        # URLs
        "oeil_url": procedure.get("oeil_url"),
        "eurlex_proposal_url": procedure.get("eurlex_proposal_url"),
        "eurlex_final_act_url": procedure.get("eurlex_final_act_url"),
        # AI-generated content (only fields that exist in our schema)
        "ai_summary": procedure.get("ai_summary"),
        "ai_impact_analysis": procedure.get("ai_impact_analysis"),
        # Structured JSONB data (strip _amendments - moved to procedure_amendments table)
        "events": strip_amendments_from_events(procedure.get("events", [])),
        "actors": procedure.get("actors", []),  # JSONB array
        "foreseen_activities": procedure.get("foreseen_activities", []),
        # Soft delete fields
        "is_deleted": procedure.get("is_deleted", False),
        # AI fields (kept from parl8 compat, nullable)
        "ai_next_steps": procedure.get("ai_next_steps"),
        "embedding_model": procedure.get("embedding_model"),
        "api_uri": procedure.get("api_uri"),
    }

    return record


def prepare_procedure_records(procedures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare multiple procedure records for Supabase upsert.

    Args:
        procedures: List of procedures from Gold layer

    Returns:
        List of records ready for database upsert
    """
    return [prepare_procedure_record(proc) for proc in procedures]


def upload_procedures(
    supabase_resource: Any,
    procedures: List[Dict[str, Any]],
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Upload procedures to Supabase.

    Args:
        supabase_resource: SupabaseResource instance
        procedures: List of procedure records
        logger: Optional logger

    Returns:
        Dict with upload statistics
    """
    if logger:
        logger.info(f"Uploading {len(procedures)} procedures to Supabase")

    result = supabase_resource.batch_upsert(
        table="procedures",
        data=procedures,
        batch_size=50,
        on_conflict="id",
        logger=logger,
    )

    if logger:
        logger.info(f"Upload complete: {result['success']} success, {result['failed']} failed")

    return result


def upload_procedure_articles(
    supabase_resource: Any,
    procedures: List[Dict[str, Any]],
    logger: Optional[Any] = None,
    proposal_only: bool = True,
) -> Dict[str, int]:
    """Extract and upload articles from procedures.

    Args:
        supabase_resource: SupabaseResource instance
        procedures: List of procedures
        logger: Optional logger
        proposal_only: If True, only extract from proposals (faster)

    Returns:
        Dict with upload statistics
    """
    from .article_extractor import (
        extract_articles_from_silver,
        extract_proposal_articles,
    )

    if logger:
        logger.info(f"Extracting articles from {len(procedures)} procedures")

    all_articles = []
    for procedure in procedures:
        try:
            if proposal_only:
                articles = extract_proposal_articles(procedure, logger)
            else:
                articles = extract_articles_from_silver(procedure, logger)

            all_articles.extend(articles)

        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract articles from {procedure.get('id')}: {e}")

    if logger:
        logger.info(f"Extracted {len(all_articles)} articles total")

    # Add deterministic IDs before upload
    for article in all_articles:
        article["id"] = generate_article_id(
            procedure_id=article["procedure_id"],
            document_version=article["document_version"],
            element_type=article["element_type"],
            element_number=article["element_number"],
        )

    # Deduplicate by composite key (database constraint)
    # Note: Must use composite key, not UUID, because database checks this constraint
    seen_keys = set()
    unique_articles = []
    duplicates = 0

    for article in all_articles:
        # Create key matching database UNIQUE constraint
        composite_key = (
            article["procedure_id"],
            article["document_source"],
            article["element_type"],
            article["element_number"],
        )
        if composite_key not in seen_keys:
            seen_keys.add(composite_key)
            unique_articles.append(article)
        else:
            duplicates += 1

    if duplicates > 0 and logger:
        logger.warning(
            f"Removed {duplicates} duplicate articles (same procedure/source/type/number)"
        )

    if logger:
        logger.info(f"Uploading {len(unique_articles)} unique articles")

    # Upload to Supabase
    # Note: Database has unique constraint on (procedure_id, document_source, element_type, element_number)
    # Must specify this for on_conflict to properly handle existing records
    result = supabase_resource.batch_upsert(
        table="procedure_articles",
        data=unique_articles,
        batch_size=100,
        on_conflict="id",  # Changed from composite key to id
        logger=logger,
    )

    if logger:
        logger.info(
            f"Articles upload complete: {result['success']} success, {result['failed']} failed"
        )

    return result


def upload_procedure_amendments(
    supabase_resource: Any,
    procedures: List[Dict[str, Any]],
    logger: Optional[Any] = None,
) -> Dict[str, int]:
    """Extract and upload amendments from procedures with MEP matching.

    Args:
        supabase_resource: SupabaseResource instance
        procedures: List of procedures from Gold layer
        logger: Optional logger

    Returns:
        Dict with upload statistics
    """
    if logger:
        logger.info(f"Extracting amendments from {len(procedures)} procedures")

    # Extract amendments with MEP matching
    all_amendments = extract_amendments_from_procedures(
        procedures,
        supabase_resource,
        logger,
    )

    if not all_amendments:
        if logger:
            logger.info("No amendments to upload")
        return {"success": 0, "failed": 0}

    if logger:
        logger.info(f"Uploading {len(all_amendments)} amendments to Supabase")

    # Add deterministic IDs before upload
    for amendment in all_amendments:
        amendment["id"] = generate_amendment_id(
            procedure_id=amendment["procedure_id"],
            document_id=amendment["document_id"],
            amendment_number=amendment["amendment_number"],
        )

    # Deduplicate by composite key (database constraint)
    # Note: Must use composite key, not UUID, because database checks this constraint
    seen_keys = set()
    unique_amendments = []
    duplicates = 0

    for amendment in all_amendments:
        # Create key matching database UNIQUE constraint
        composite_key = (
            amendment["procedure_id"],
            amendment["document_id"],
            amendment["amendment_number"],
        )
        if composite_key not in seen_keys:
            seen_keys.add(composite_key)
            unique_amendments.append(amendment)
        else:
            duplicates += 1

    if duplicates > 0 and logger:
        logger.warning(
            f"Removed {duplicates} duplicate amendments (same procedure/document/number)"
        )

    if logger:
        logger.info(f"Uploading {len(unique_amendments)} unique amendments")

    # Upload to Supabase
    # Note: Database has unique constraint on (procedure_id, document_id, amendment_number)
    # Must specify this for on_conflict, even though we also have an 'id' column
    result = supabase_resource.batch_upsert(
        table="procedure_amendments",
        data=unique_amendments,
        batch_size=100,
        on_conflict="id",  # Changed from composite key to id
        logger=logger,
    )

    if logger:
        logger.info(
            f"Amendments upload complete: {result['success']} success, {result['failed']} failed"
        )

    return result


# Asset definition
@asset(
    group_name="eu_diamond",
    compute_kind="upload",
    partitions_def=weekly_partitions,
)
def eu_legislation_diamond(
    context: AssetExecutionContext,
    supabase: SupabaseResource,
    eu_legislation_gold: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Diamond layer: Upload procedures and articles to Supabase."""
    context.log.info(f"Uploading {len(eu_legislation_gold)} procedures to Supabase")

    # Prepare records
    records = prepare_procedure_records(eu_legislation_gold)

    # Upload procedures (pass resource, not client)
    proc_stats = upload_procedures(supabase, records, logger=context.log)

    # Upload articles (both proposal and final act)
    article_stats = upload_procedure_articles(
        supabase,
        eu_legislation_gold,
        logger=context.log,
        proposal_only=False,  # Extract both proposal and final act articles
    )

    # Upload amendments with MEP matching
    amendment_stats = upload_procedure_amendments(
        supabase,
        eu_legislation_gold,
        logger=context.log,
    )

    # Add metadata
    context.add_output_metadata(
        {
            "procedures_uploaded": proc_stats["success"],
            "procedures_failed": proc_stats["failed"],
            "articles_uploaded": article_stats.get("success", 0),
            "articles_failed": article_stats.get("failed", 0),
            "amendments_uploaded": amendment_stats.get("success", 0),
            "amendments_failed": amendment_stats.get("failed", 0),
        }
    )

    # Check for critical failures and raise exception
    total_procedures = proc_stats["success"] + proc_stats["failed"]
    total_articles = article_stats.get("success", 0) + article_stats.get("failed", 0)
    total_amendments = amendment_stats.get("success", 0) + amendment_stats.get("failed", 0)

    errors = []

    # Check procedures (critical - must succeed)
    if proc_stats["failed"] > 0:
        failure_rate = (
            (proc_stats["failed"] / total_procedures * 100) if total_procedures > 0 else 0
        )
        errors.append(
            f"Procedures: {proc_stats['failed']}/{total_procedures} failed ({failure_rate:.1f}%)"
        )

    # Check articles (critical if any exist)
    if total_articles > 0 and article_stats.get("failed", 0) > 0:
        failure_rate = (article_stats["failed"] / total_articles * 100) if total_articles > 0 else 0
        errors.append(
            f"Articles: {article_stats['failed']}/{total_articles} failed ({failure_rate:.1f}%)"
        )

    # Check amendments (warning only if partial failure, error if total failure)
    if total_amendments > 0 and amendment_stats.get("failed", 0) > 0:
        failure_rate = (
            (amendment_stats["failed"] / total_amendments * 100) if total_amendments > 0 else 0
        )
        if failure_rate == 100:
            errors.append(
                f"Amendments: {amendment_stats['failed']}/{total_amendments} failed (100%)"
            )
        else:
            context.log.warning(
                f"Amendments: {amendment_stats['failed']}/{total_amendments} failed "
                f"({failure_rate:.1f}%)"
            )

    # Raise exception if any critical failures
    if errors:
        error_msg = "Diamond layer upload failed:\n" + "\n".join(f"  - {e}" for e in errors)
        context.log.error(error_msg)
        raise ValueError(error_msg)

    context.log.info("All uploads completed successfully")

    return {
        "procedures": proc_stats,
        "articles": article_stats,
        "amendments": amendment_stats,
    }
