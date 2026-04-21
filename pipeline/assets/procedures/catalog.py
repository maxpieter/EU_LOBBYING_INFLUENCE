"""Procedure catalog: scrape ALL EU procedures (minimal fields).

Uses the EP Open Data Portal /procedures endpoint to discover all ~4,900
procedures. Only upserts {id, process_id, title, procedure_type} — existing
detailed rows keep their events/actors/docs because PostgREST only updates
columns present in the payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import re

from pipeline.assets.legislation.bronze import _process_id_to_oeil_ref
from pipeline.assets.legislation.v2_feed import fetch_all_procedure_ids


def fetch_procedure_catalog(logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Fetch all procedures from EP Open Data Portal.

    Returns list of minimal procedure dicts ready for upsert:
    {id, process_id, title, procedure_type}
    """
    _log = logger.info if logger else print

    raw = fetch_all_procedure_ids(logger=logger)
    _log(f"Fetched {len(raw)} procedure IDs from EP Open Data Portal")

    procedures = []
    for item in raw:
        process_id = item.get("process_id", "")
        process_type = item.get("process_type", "")
        label = item.get("label", "")
        if isinstance(label, list):
            label = next((x for x in label if isinstance(x, str) and x.strip()), "")

        if not process_id or not process_type:
            continue

        oeil_ref = _process_id_to_oeil_ref(process_id, process_type)
        if not oeil_ref:
            continue

        # Title has a NOT NULL constraint. Prefer the real label when the API
        # provides one; fall back to the reference ID so the insert succeeds.
        if label and not re.match(r"^\d{4}/\d{4}\(", label):
            title = label
        else:
            title = oeil_ref

        record = {
            "id": oeil_ref,
            "process_id": process_id,
            "procedure_type": process_type,
            "title": title,
        }

        procedures.append(record)

    _log(f"Prepared {len(procedures)} procedures for upsert")
    return procedures
