#!/usr/bin/env python3
"""Chunk HYS feedback for one or more procedures (no Dagster required).

Usage:
    python scripts/chunk_hys_feedback.py "2023/0284(COD)"
    python scripts/chunk_hys_feedback.py "2023/0284(COD)" "2022/0051(COD)" "2020/0374(COD)"
    python scripts/chunk_hys_feedback.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

from supabase import create_client

from pipeline.assets.feedback.hys_feedback_scraper import (
    build_chunk_records,
    upsert_chunk_rows,
)


def chunk_procedure(client, procedure_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Chunking: {procedure_id}")
    print(f"{'='*60}")

    total_chunks = 0
    rows_processed = 0
    rows_skipped = 0
    last_id = 0

    while True:
        id_rows = (
            client.table("hys_feedback_bronze")
            .select("feedback_id")
            .eq("procedure_id", procedure_id)
            .not_.is_("feedback_text", "null")
            .gt("feedback_id", last_id)
            .order("feedback_id")
            .limit(50)
            .execute()
        ).data or []

        if not id_rows:
            break

        ids = [r["feedback_id"] for r in id_rows]
        last_id = ids[-1]

        already_chunked = {
            r["feedback_id"]
            for r in (
                client.table("hys_feedback_chunks")
                .select("feedback_id")
                .in_("feedback_id", ids)
                .execute()
            ).data or []
        }

        for fid in ids:
            if fid in already_chunked:
                rows_skipped += 1
                continue

            row = (
                client.table("hys_feedback_bronze")
                .select(
                    "feedback_id, initiative_id, procedure_id, com_number, "
                    "feedback_text, organisation_name, transparency_reg_id, date_feedback"
                )
                .eq("feedback_id", fid)
                .maybe_single()
                .execute()
            ).data
            if not row:
                continue

            text = row.get("feedback_text") or ""
            if len(text) <= 100:
                continue

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
            if chunk_records:
                total_chunks += upsert_chunk_rows(chunk_records, client)
            rows_processed += 1

            if rows_processed % 25 == 0:
                print(f"  [{procedure_id}] {rows_processed} rows → {total_chunks} chunks")

    print(f"  Done: {rows_processed} chunked, {rows_skipped} skipped, {total_chunks} chunks total")
    return {"procedure_id": procedure_id, "rows": rows_processed, "chunks": total_chunks}


def main():
    parser = argparse.ArgumentParser(description="Chunk HYS feedback")
    parser.add_argument("procedures", nargs="*", help="Procedure IDs to chunk")
    parser.add_argument("--all", action="store_true", help="Chunk all unchunked procedures")
    args = parser.parse_args()

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    if args.all:
        bronze = {}
        offset = 0
        while True:
            batch = (
                client.table("hys_feedback_bronze")
                .select("procedure_id")
                .not_.is_("feedback_text", "null")
                .range(offset, offset + 999)
                .execute()
            ).data or []
            if not batch:
                break
            for r in batch:
                bronze.setdefault(r["procedure_id"], 0)
                bronze[r["procedure_id"]] += 1
            offset += 1000
            if len(batch) < 1000:
                break
        procedures = sorted(bronze.keys())
        print(f"All mode: {len(procedures)} procedures with bronze feedback")
    else:
        procedures = args.procedures

    if not procedures:
        parser.print_help()
        return

    results = []
    for pid in procedures:
        results.append(chunk_procedure(client, pid))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['procedure_id']:25s}  {r['rows']:>4d} rows → {r['chunks']:>5d} chunks")


if __name__ == "__main__":
    main()
