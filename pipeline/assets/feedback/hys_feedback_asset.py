"""Dagster asset definition for HYS feedback scraping.

Imports all logic from hys_feedback_scraper.py (no Dagster dependency there),
so tests can import the logic without triggering the @asset decorator.
"""

from dagster import AssetExecutionContext, Config, asset, AssetDep

from .hys_feedback_scraper import (
    _make_session,
    build_chunk_records,
    fetch_procedures_with_com_numbers,
    scrape_hys_feedback_for_procedure,
    upsert_chunk_rows,
    BATCH_SIZE,
)


class HYSFeedbackConfig(Config):
    """Configuration for the HYS feedback scraper asset."""

    procedure_id: str = ""
    """Single procedure OEIL reference to scrape (e.g. '2025/0005(COD)').
    Empty string = all procedures with a COM number in Supabase."""

    com_number_override: str = ""
    """Manually provide the COM number instead of reading from Supabase.
    Only used when procedure_id is also set. Useful for testing."""

    skip_existing: bool = True
    """Skip (initiative, procedure) pairs that already have rows in Supabase."""


@asset(
    name="hys_feedback_bronze",
    group_name="eu_bronze",
    compute_kind="scraper",
    required_resource_keys={"supabase"},
    description=(
        "Scrape Have Your Say feedback for EU legislation procedures. "
        "Matches procedures to HYS initiatives via COM number, paginates all "
        "non-citizen feedback, and stores raw rows + text chunks in Supabase."
    ),
)
def hys_feedback_bronze(
    context: AssetExecutionContext,
    config: HYSFeedbackConfig,
) -> dict[str, int]:
    """Bronze asset: HYS feedback scraping."""
    from pipeline.resources.supabase import SupabaseResource

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    # --- Resolve which procedures to process ---
    if config.procedure_id and config.com_number_override:
        procedures = [
            {
                "id": config.procedure_id,
                "commission_document": config.com_number_override,
            }
        ]
        context.log.info(
            f"Manual mode: {config.procedure_id} / {config.com_number_override}"
        )
    elif config.procedure_id:
        procedures = fetch_procedures_with_com_numbers(
            client, procedure_ids=[config.procedure_id]
        )
        if not procedures:
            context.log.warning(
                f"Procedure {config.procedure_id} not found or has no COM number"
            )
            return {"feedback_upserted": 0, "chunks_upserted": 0}
    else:
        procedures = fetch_procedures_with_com_numbers(client)
        context.log.info(f"Batch mode: {len(procedures)} procedures with COM numbers")

    total_feedback = 0
    total_chunks = 0
    succeeded = 0
    no_initiative = 0
    failed = 0

    for i, proc in enumerate(procedures, 1):
        procedure_id = proc["id"]
        com_number = proc["commission_document"]
        context.log.info(f"[{i}/{len(procedures)}] {procedure_id} ({com_number})")
        # Create a fresh session per procedure to avoid stale keep-alive
        # connections in the pool hanging for the OS TCP retransmission timeout
        # (~10 min on macOS) when the EC server closes idle connections.
        session = _make_session()
        try:
            result = scrape_hys_feedback_for_procedure(
                procedure_id=procedure_id,
                com_number=com_number,
                session=session,
                client=client,
                logger=context.log,
                skip_existing=config.skip_existing,
            )
            if result["initiatives_found"] == 0:
                no_initiative += 1
            elif result["feedback_upserted"] > 0:
                succeeded += 1
                total_feedback += result["feedback_upserted"]
                total_chunks += result["chunks_upserted"]
            else:
                no_initiative += 1
        except Exception as exc:
            context.log.error(f"  Failed for {procedure_id}: {exc}")
            failed += 1
        finally:
            session.close()

    context.add_output_metadata(
        {
            "procedures_processed": len(procedures),
            "procedures_with_feedback": succeeded,
            "procedures_no_initiative": no_initiative,
            "procedures_failed": failed,
            "total_feedback_rows": total_feedback,
            "total_chunk_rows": total_chunks,
        }
    )
    return {
        "feedback_upserted": total_feedback,
        "chunks_upserted": total_chunks,
        "succeeded": succeeded,
        "failed": failed,
    }


class HYSChunksConfig(Config):
    """Configuration for the HYS feedback chunks asset."""

    procedure_id: str = ""
    """Limit chunking to a single procedure (e.g. '2022/0272(COD)').
    Empty string = all procedures. Use this to test before running on all rows."""


@asset(
    name="hys_feedback_chunks",
    group_name="eu_bronze",
    compute_kind="transform",
    required_resource_keys={"supabase"},
    deps=["hys_feedback_bronze"],
    description=(
        "Build text chunks from hys_feedback_bronze rows. "
        "Reads feedback_text from Supabase, skips already-chunked rows, "
        "and upserts chunks one row at a time. Downstream of hys_feedback_bronze."
    ),
)
def hys_feedback_chunks(
    context: AssetExecutionContext,
    config: HYSChunksConfig,
) -> dict[str, int]:
    """Downstream asset: chunk feedback_text from bronze into hys_feedback_chunks."""
    from pipeline.resources.supabase import SupabaseResource

    supabase: SupabaseResource = context.resources.supabase
    client = supabase.get_client()

    total_chunks = 0
    rows_processed = 0
    last_id = 0  # cursor-based pagination — no offset drift

    context.log.info("Starting chunking — fetching unchunked rows from hys_feedback_bronze...")
    while True:
        # Fetch IDs only — never load multiple large feedback_texts into memory at once.
        # PDF texts can be several MB each; fetching 50 at once causes OOM.
        id_query = (
            client.table("hys_feedback_bronze")
            .select("feedback_id")
            .not_.is_("feedback_text", "null")
            .gt("feedback_id", last_id)
            .order("feedback_id")
            .limit(50)
        )
        if config.procedure_id:
            id_query = id_query.eq("procedure_id", config.procedure_id)

        id_rows = (id_query.execute()).data or []
        context.log.info(f"  Page cursor={last_id}: got {len(id_rows)} IDs")
        if not id_rows:
            break

        ids = [r["feedback_id"] for r in id_rows]
        last_id = ids[-1]

        # One query to find which are already chunked
        already_chunked = {
            r["feedback_id"]
            for r in (
                client.table("hys_feedback_chunks")
                .select("feedback_id")
                .in_("feedback_id", ids)
                .execute()
            ).data or []
        }
        context.log.info(
            f"  {len(already_chunked)} already chunked, {len(ids) - len(already_chunked)} to process"
        )

        for fid in ids:
            if fid in already_chunked:
                continue

            # Fetch one row at a time — keeps memory bounded to one PDF text at a time
            context.log.info(f"  Processing feedback_id={fid}")
            row_resp = (
                client.table("hys_feedback_bronze")
                .select(
                    "feedback_id, initiative_id, procedure_id, com_number, "
                    "feedback_text, organisation_name, transparency_reg_id, date_feedback"
                )
                .eq("feedback_id", fid)
                .maybe_single()
                .execute()
            )
            row = row_resp.data if row_resp else None
            if not row:
                continue

            text = row.get("feedback_text") or ""
            if len(text) <= 100:
                continue

            context.log.info(f"    text={len(text):,} chars, building chunks...")
            chunk_records = build_chunk_records(
                feedback_id=row["feedback_id"],
                initiative_id=row["initiative_id"],
                procedure_id=row["procedure_id"],
                com_number=row["com_number"],
                text=text,
                organisation_name=row.get("organisation_name"),
                transparency_reg_id=row.get("transparency_reg_id"),
                date_feedback=row.get("date_feedback"),
            )
            context.log.info(f"    built {len(chunk_records)} chunks, upserting...")
            if chunk_records:
                total_chunks += upsert_chunk_rows(chunk_records, client, logger=context.log)
            context.log.info(f"    upsert done, total_chunks={total_chunks}")
            rows_processed += 1
            if rows_processed % 10 == 0:
                context.log.info(f"  Chunked {rows_processed} rows, {total_chunks} chunks so far")

    context.add_output_metadata({
        "rows_chunked": rows_processed,
        "total_chunks": total_chunks,
    })
    context.log.info(f"Done: {rows_processed} rows → {total_chunks} chunks")
    return {"rows_chunked": rows_processed, "chunks_upserted": total_chunks}
