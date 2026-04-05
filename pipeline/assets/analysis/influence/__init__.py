"""EU lobbying influence analysis pipeline — evidence dossier approach.

Steps
-----
1.  step1_collect_data             — fetch procedure data from Supabase
2.  step2_generate_taxonomy        — AI-assisted theme taxonomy (cached to disk)
3.  step3_parse_amendments         — parse amendments from Supabase or local PDFs
4.  step4_classify_amendments      — AI theme classification (diff-aware)
5.  step5_extract_positions        — AI position extraction from meetings
6.  step6_quantitative_analysis    — org influence, theme indicators, match density
7.  step7_commission_evidence      — commission-level evidence assembly (AI summaries)
8.  step8_amendment_evidence       — amendment-level evidence assembly (deterministic)
9.  step9_generate_report          — assemble JSON report and generate one-pager
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Re-exports for backward compatibility
from ._ai import configure_ai_provider
from ._helpers import _classify_by_regex, compile_taxonomy_patterns
from ._supabase import fetch_all

# Step imports
from . import _config
from .step1_collect import step1_collect_data
from .step2_taxonomy import step2_generate_taxonomy
from .step3_amendments import step3_parse_amendments
from .step4_classify import step4_classify_amendments
from .step5_positions import step5_extract_positions
from .step6_quant import step6_quantitative_analysis
from .step7_commission_evidence import step7_commission_evidence
from .step8_amendment_evidence import step8_amendment_evidence
from .step9_report import step9_generate_report


def run_influence_pipeline(
    procedure_id: str,
    client: Any,
    regen_taxonomy: bool = False,
    output_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run all 9 pipeline steps and return the report dict.

    This is the single entry-point called from the Dagster asset.

    Parameters
    ----------
    procedure_id:
        EU procedure reference, e.g. ``2023/0212(COD)``.
    client:
        Raw Supabase client (from ``SupabaseResource.get_client()``).
    regen_taxonomy:
        When True, the cached taxonomy file is deleted and regenerated.
    output_dir:
        Directory for the JSON report. Defaults to ``ANALYSIS_OUTPUT_DIR``.
    logger:
        Optional logger (e.g. ``context.log`` from Dagster).
    """
    _log = logger.info if logger else print

    _log(f"Starting influence pipeline for {procedure_id}")

    configure_ai_provider()
    if _config.AI_PROVIDER is None:
        raise RuntimeError(
            "AI provider could not be configured. "
            "Ensure the 'claude' CLI is installed and available on PATH."
        )
    _log(f"AI provider: {_config.AI_PROVIDER}")

    # Step 1 — collect all data from Supabase
    data = step1_collect_data(procedure_id, client, logger=logger)

    # Steps 2 + 3 in parallel (taxonomy generation + amendment fetching)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_taxonomy: Future = pool.submit(
            step2_generate_taxonomy,
            procedure_id, data, regen_taxonomy, logger,
        )
        future_amendments: Future = pool.submit(
            step3_parse_amendments,
            procedure_id, client, logger,
        )
        taxonomy = future_taxonomy.result()
        amendments = future_amendments.result()

    compiled_patterns = compile_taxonomy_patterns(taxonomy)

    # Steps 4 + 5 in parallel (classify amendments + extract positions)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_classified: Future = pool.submit(
            step4_classify_amendments,
            amendments, taxonomy, logger,
        )
        future_positions: Future = pool.submit(
            step5_extract_positions,
            data["commission"], taxonomy, compiled_patterns, logger,
        )
        amendments = future_classified.result()
        positions = future_positions.result()

    # Step 6 — quantitative analysis (needs classified amendments + positions)
    quant = step6_quantitative_analysis(
        data, amendments, positions, taxonomy, compiled_patterns, logger=logger
    )

    # Steps 7 + 8 in parallel (commission evidence + amendment evidence)
    documents = data.get("documents", {})
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_commission_ev: Future = pool.submit(
            step7_commission_evidence,
            taxonomy, positions, compiled_patterns, documents,
            quant.get("theme_lobbying_density", []), logger,
        )
        future_amendment_ev: Future = pool.submit(
            step8_amendment_evidence,
            amendments, positions, taxonomy, quant, logger,
        )
        commission_evidence = future_commission_ev.result()
        amendment_evidence = future_amendment_ev.result()

    # Step 9 — assemble and write report
    report = step9_generate_report(
        procedure_id, data, taxonomy, amendments, positions, quant,
        commission_evidence, amendment_evidence,
        output_dir=output_dir, logger=logger,
    )

    _log(f"Influence pipeline complete for {procedure_id}")
    return report
