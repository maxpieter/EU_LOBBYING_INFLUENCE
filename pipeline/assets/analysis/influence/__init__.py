"""EU lobbying influence analysis pipeline.

Steps
-----
1.  step1_collect_data          — fetch procedure data from Supabase
2.  step2_generate_taxonomy     — AI-assisted theme taxonomy (cached to disk)
3.  step3_parse_amendments      — parse amendments from Supabase or local PDFs
4.  step4_classify_amendments   — AI theme classification (diff-aware)
5.  step5_extract_positions     — AI position extraction from meetings
6.  step6_quantitative_analysis — LEI / ALAS / ICI / Fisher's exact
7.  step7_directional_alignment — AI-scored amendment-to-lobby alignment
8.  step8_proposal_alignment    — lobby positions vs. commission proposal text
9.  step9_text_evolution        — track how provisions changed across doc stages
10. step10_lifecycle_score      — Lifecycle Influence Index (LII) per theme
11. step11_generate_report      — assemble JSON report and generate one-pager
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
from .step7_alignment import step7_directional_alignment
from .step8_proposal import step8_proposal_alignment
from .step9_evolution import step9_text_evolution
from .step10_lifecycle import step10_lifecycle_score
from .step11_report import step11_generate_report


def run_influence_pipeline(
    procedure_id: str,
    client: Any,
    regen_taxonomy: bool = False,
    output_dir: Path | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Run all 11 pipeline steps and return the report dict.

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
        data, amendments, taxonomy, compiled_patterns, logger=logger
    )

    # Step 7 — directional alignment (needs quant + positions + amendments)
    alignment = step7_directional_alignment(
        quant, positions, amendments, taxonomy, logger=logger
    )

    # Steps 8 + 9 in parallel (both are independent of each other)
    documents = data.get("documents", {})
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_proposal_align: Future = pool.submit(
            step8_proposal_alignment,
            positions, taxonomy, compiled_patterns, documents, logger,
        )
        future_text_evo: Future = pool.submit(
            step9_text_evolution,
            taxonomy, compiled_patterns, documents, logger,
        )
        proposal_alignment = future_proposal_align.result()
        text_evolution = future_text_evo.result()

    # Step 10 — lifecycle score (depends on step 8 output)
    lifecycle_scores = step10_lifecycle_score(
        positions, amendments, taxonomy, compiled_patterns, documents,
        proposal_alignment, alignment, logger=logger,
    )

    # Step 11 — assemble and write report
    report = step11_generate_report(
        procedure_id, data, taxonomy, amendments, positions, quant, alignment,
        output_dir=output_dir, logger=logger,
        proposal_alignment=proposal_alignment,
        text_evolution=text_evolution,
        lifecycle_scores=lifecycle_scores,
    )

    _log(f"Influence pipeline complete for {procedure_id}")
    return report
