"""Analysis Assets — the influence analysis pipeline.

This is the top-level analytical asset that sits outside the medallion
architecture. It depends on bronze data (amendments, documents) being
already materialized in Supabase and runs the 9-step evidence-dossier
influence pipeline for a given procedure.
"""

from dagster import AssetExecutionContext, Config, asset

from pipeline.resources.supabase import SupabaseResource


class InfluenceAnalysisConfig(Config):
    """Configuration for the influence analysis asset."""

    procedure_id: str
    """Required: EU procedure reference, e.g. '2023/0212(COD)'."""

    regen_taxonomy: bool = False
    """When True, delete the cached taxonomy and regenerate via AI."""


@asset(
    name="eu_influence_analysis",
    group_name="analysis",
    compute_kind="ai+python",
    required_resource_keys={"supabase"},
    description=(
        "9-step evidence-dossier lobbying analysis pipeline. "
        "Combines Supabase data (amendments, meetings, procedures) with "
        "Claude AI calls to extract positions, classify themes, and "
        "assemble evidence dossiers for journalist review."
    ),
)
def eu_influence_analysis(context: AssetExecutionContext, config: InfluenceAnalysisConfig):
    from .influence import run_influence_pipeline

    if not config.procedure_id or not config.procedure_id.strip():
        raise ValueError(
            "procedure_id is required. Set it in the Dagster launchpad config, "
            "e.g. procedure_id: '2021/0106(COD)'"
        )

    procedure_id = config.procedure_id.strip()
    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    context.log.info(
        f"Starting influence analysis for {procedure_id} "
        f"(regen_taxonomy={config.regen_taxonomy})"
    )

    report = run_influence_pipeline(
        procedure_id=procedure_id,
        client=client,
        regen_taxonomy=config.regen_taxonomy,
        logger=context.log,
    )

    stats = report.get("summary_stats", {})

    context.add_output_metadata({
        "procedure_id": procedure_id,
        "title": report.get("title", ""),
        "ai_provider": report.get("ai_provider") or "None (regex-only)",
        "total_amendments_parsed": stats.get("total_amendments_parsed", 0),
        "total_lobbying_meetings": stats.get("total_lobbying_meetings", 0),
        "commission_meetings_with_notes": stats.get("commission_meetings_with_notes", 0),
        "total_organisations": stats.get("total_organisations", 0),
        "themes_generated": len(report.get("taxonomy", {})),
        "meps_analysed": len(report.get("mep_crossref", {})),
        "commission_evidence_dossiers": len(report.get("commission_evidence", [])),
        "amendment_evidence_dossiers": len(report.get("amendment_evidence", [])),
    })

    return report


analysis_assets = [
    eu_influence_analysis,
]
