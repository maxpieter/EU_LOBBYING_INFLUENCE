"""Legislation Assets - Bronze → Silver → Diamond (no Gold/AI layer).

OEIL scraping + v2 API enrichment → Silver transformation → Upload to Supabase.
"""

from typing import Any, Dict, List

from dagster import AssetExecutionContext, AssetIn, asset

from pipeline.partitions.definitions import weekly_partitions
from pipeline.resources.supabase import SupabaseResource

from .bronze import eu_legislation_bronze
from .silver import eu_legislation_silver


@asset(
    name="eu_legislation_diamond",
    group_name="eu_diamond",
    description="Upload legislation to Supabase procedures table.",
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

    context.add_output_metadata(
        {
            "uploaded": result.get("success", 0),
            "failed": result.get("failed", 0),
        }
    )

    return result


legislation_assets = [
    eu_legislation_bronze,
    eu_legislation_silver,
    eu_legislation_diamond,
]
