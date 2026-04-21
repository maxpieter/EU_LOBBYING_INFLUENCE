"""Step 1: Data collection — fetch all procedure data from Supabase."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._supabase import fetch_all


def step1_collect_data(
    procedure_id: str,
    client: Any,
    logger: Any = None,
) -> dict[str, Any]:
    """Fetch all data for the procedure from Supabase.

    Returns a dict with keys: ``procedure``, ``articles``, ``lobbying``,
    ``commission``.
    """
    _log = logger.info if logger else print

    _log(f"STEP 1: Fetching procedure {procedure_id!r} ...")

    proc_resp = (
        client.table("procedures")
        .select("id,title,description,events,actors")
        .eq("id", procedure_id)
        .limit(1)
        .execute()
    )
    if not proc_resp.data:
        raise ValueError(f"Procedure {procedure_id!r} not found in database.")
    procedure = proc_resp.data[0]
    _log(f"Title: {procedure.get('title', 'N/A')}")

    # Fetch all document types from procedure_documents
    _DOC_TYPES = [
        "commission_proposal",
        "draft_report",
        "opinion",
        "committee_report",
        "text_adopted",
    ]
    all_proc_docs = fetch_all(
        client,
        "procedure_documents",
        "procedure_id,document_id,document_type,content_text",
        {"procedure_id": procedure_id},
    )

    _DOC_TRUNCATE = 80000

    def _first_doc_text(doc_type: str) -> str:
        for doc in all_proc_docs:
            if doc.get("document_type") == doc_type:
                return (doc.get("content_text") or "")[:_DOC_TRUNCATE]
        return ""

    def _all_doc_texts(doc_type: str) -> list[str]:
        return [
            (doc.get("content_text") or "")[:_DOC_TRUNCATE]
            for doc in all_proc_docs
            if doc.get("document_type") == doc_type
            and (doc.get("content_text") or "").strip()
        ]

    proposal_text = _first_doc_text("commission_proposal")
    draft_report_text = _first_doc_text("draft_report")
    opinion_texts = _all_doc_texts("opinion")
    committee_report_text = _first_doc_text("committee_report")
    text_adopted_text = _first_doc_text("text_adopted")

    if proposal_text:
        _log(f"Commission proposal text: {len(proposal_text)} chars")
    else:
        _log("Commission proposal text: not found")
    if draft_report_text:
        _log(f"Draft report text: {len(draft_report_text)} chars")
    if opinion_texts:
        _log(f"Opinions: {len(opinion_texts)} document(s)")
    if committee_report_text:
        _log(f"Committee report text: {len(committee_report_text)} chars")
    if text_adopted_text:
        _log(f"Text adopted: {len(text_adopted_text)} chars")

    documents: dict[str, Any] = {
        "commission_proposal": proposal_text,
        "draft_report": draft_report_text,
        "opinions": opinion_texts,
        "committee_report": committee_report_text,
        "text_adopted": text_adopted_text,
    }

    commission_meetings = _fetch_commission_meetings(client, procedure_id)
    commission_meetings = _enrich_commission_meetings(client, commission_meetings)
    with_notes = sum(1 for m in commission_meetings if m.get("points_raised"))
    _log(
        f"Commission meetings: {len(commission_meetings)} ({with_notes} with points_raised)"
    )

    lobbying_meetings = _fetch_lobbying_meetings(client, procedure_id)
    if lobbying_meetings:
        lobbying_meetings = _enrich_lobbying_meetings(client, lobbying_meetings)
    _log(f"EP lobbying meetings: {len(lobbying_meetings)}")

    unique_orgs = {m.get("org_name", "") for m in lobbying_meetings if m.get("org_name")}
    unique_meps = {m.get("mep_name", "") for m in lobbying_meetings if m.get("mep_name")}
    _log(f"Unique lobbying organisations: {len(unique_orgs)}")
    _log(f"Unique MEPs with meetings: {len(unique_meps)}")

    return {
        "procedure": procedure,
        "proposal_text": proposal_text,
        "documents": documents,
        "lobbying": lobbying_meetings,
        "commission": commission_meetings,
    }


def _fetch_commission_meetings(client: Any, procedure_id: str) -> list[dict[str, Any]]:
    # All matches now live in meeting_procedure_links (legacy matched_procedure_id dropped)
    link_rows = fetch_all(
        client,
        "meeting_procedure_links",
        "commission_meeting_id",
        {"procedure_id": procedure_id},
    )
    linked_ids = [r["commission_meeting_id"] for r in link_rows if r.get("commission_meeting_id")]

    meetings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for mid in linked_ids:
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        resp = (
            client.table("commission_meetings")
            .select(
                "id,commissioner_name,meeting_date,subject,organizations_raw,"
                "points_raised,conclusions"
            )
            .eq("id", mid)
            .limit(1)
            .execute()
        )
        if resp.data:
            meetings.extend(resp.data)
    return meetings


def _fetch_lobbying_meetings(client: Any, procedure_id: str) -> list[dict[str, Any]]:
    link_rows = fetch_all(
        client,
        "meeting_procedure_links",
        "lobbying_meeting_id",
        {"procedure_id": procedure_id},
    )
    linked_ids = [r["lobbying_meeting_id"] for r in link_rows if r.get("lobbying_meeting_id")]
    direct = fetch_all(
        client,
        "lobbying_meetings",
        "id,mep_id,organization_id,meeting_date,title,capacity,related_procedure",
        {"related_procedure": procedure_id},
    )
    direct_ids = {m["id"] for m in direct}
    for mid in linked_ids:
        if mid not in direct_ids:
            resp = (
                client.table("lobbying_meetings")
                .select("id,mep_id,organization_id,meeting_date,title,capacity,related_procedure")
                .eq("id", mid)
                .limit(1)
                .execute()
            )
            if resp.data:
                direct.extend(resp.data)
                direct_ids.add(mid)
    return direct


def _enrich_commission_meetings(
    client: Any,
    meetings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not meetings:
        return meetings
    meeting_ids = [m["id"] for m in meetings]
    all_links: list[dict[str, Any]] = []
    for i in range(0, len(meeting_ids), 200):
        chunk = meeting_ids[i : i + 200]
        resp = (
            client.table("commission_meeting_organizations")
            .select("meeting_id,organization_id,organization_name")
            .in_("meeting_id", chunk)
            .limit(10000)
            .execute()
        )
        if resp.data:
            all_links.extend(resp.data)

    org_ids = list({lnk["organization_id"] for lnk in all_links if lnk.get("organization_id")})
    org_ir_map: dict[str, str] = {}
    for i in range(0, len(org_ids), 200):
        chunk = org_ids[i : i + 200]
        resp = (
            client.table("organizations")
            .select("id,interests_represented")
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            if row.get("interests_represented"):
                org_ir_map[row["id"]] = row["interests_represented"]

    meeting_orgs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lnk in all_links:
        meeting_orgs[lnk["meeting_id"]].append(
            {
                "name": lnk.get("organization_name", ""),
                "interests_represented": org_ir_map.get(lnk.get("organization_id", ""), "Unknown"),
            }
        )
    for m in meetings:
        m["resolved_orgs"] = meeting_orgs.get(m["id"], [])
    return meetings


def _enrich_lobbying_meetings(
    client: Any,
    meetings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    org_ids = list({m["organization_id"] for m in meetings if m.get("organization_id")})
    org_map: dict[str, dict[str, Any]] = {}
    for i in range(0, len(org_ids), 200):
        chunk = org_ids[i : i + 200]
        resp = (
            client.table("organizations")
            .select("id,name,interests_represented")
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            org_map[row["id"]] = row

    mep_ids = list({m["mep_id"] for m in meetings if m.get("mep_id")})
    mep_map: dict[int, str] = {}
    for i in range(0, len(mep_ids), 200):
        chunk = mep_ids[i : i + 200]
        resp = (
            client.table("meps")
            .select('id,"fullName"')
            .in_("id", chunk)
            .limit(10000)
            .execute()
        )
        for row in resp.data or []:
            mep_map[row["id"]] = row.get("fullName", f"MEP {row['id']}")

    for m in meetings:
        org_data = org_map.get(m.get("organization_id", ""), {})
        m["org_name"] = org_data.get("name", "")
        m["interests_represented"] = org_data.get("interests_represented") or "Unknown"
        m["mep_name"] = mep_map.get(m.get("mep_id"), "Unknown MEP")
    return meetings
