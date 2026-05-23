#!/usr/bin/env python3
"""Preproposal alignment pipeline.

For a given procedure, embeds pre-proposal lobbying texts against the Commission
proposal articles using intfloat/multilingual-e5-large, then classifies each
surviving reciprocal match with Claude (Prompt D).

Outputs a flat CSV + JSON under analysis/<proc_slug>/preproposal_<date>.*

Sources used (pre-proposal window only):
  - HYS feedback chunks       (hys_feedback_chunks, date_feedback < proposal_date)
  - Commissioner meeting notes (commission_meetings via meeting_procedure_links,
                                meeting_date < proposal_date)

Usage:
    python scripts/run_preproposal_pipeline.py --procedure "2021/0106(COD)"
    python scripts/run_preproposal_pipeline.py --procedure "2025/0102(COD)" --output-dir analysis/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

# ── Pipeline constants ─────────────────────────────────────────────────────────
MODEL_ID        = "intfloat/multilingual-e5-large"
QUERY_PREFIX    = "query: "
PASSAGE_PREFIX  = "passage: "

K             = 5     # top-k articles per source text before reciprocal filter
MIN_SCORE     = 0.84  # minimum cosine similarity to include a pair
K_RECIP       = 10    # reverse top-K for reciprocal filter
MIN_CHUNK_LEN = 80    # skip texts shorter than this

LLM_MODEL   = "claude-sonnet-4-6"
MAX_TOKENS  = 800
TEMPERATURE = 0.0
LABELS      = ["ALIGNED", "OPPOSING", "UNDETECTABLE", "NOISE"]

# ── Procedure aliases ─────────────────────────────────────────────────────────
# Alternative names / references for the procedure to search for in commission
# meeting texts (subject, points_raised, conclusions).
# Catches meetings that discuss this procedure but weren't formally linked via
# meeting_procedure_links.  These count as real access evidence, so alias-matched
# meetings are included in both the source pool and access counts.
#
# Add short names, acronyms, COM document numbers, and common titles.
# Matching is case-insensitive substring search.
# Leave empty ([]) to disable.
PROCEDURE_ALIASES: list[str] = [
    # Fill in before running, e.g.:
    "AI act",
    "artificial intelligence act",
    # "2021/0106",
    # "COM(2021) 206",
]


# ── Citation stripping (mirrors build_annotation_candidates.py) ────────────────
_NUM     = r'\(?\d+[a-z]?\)?'
_SEP     = r'(?:\s*(?:,|and|to|-)\s*' + _NUM + r')*'
_ART     = r'(?:articles?|art\.?)'
_REC     = r'(?:recitals?)'
STRIP_RE = re.compile(
    _ART + r'\s*' + _NUM + r'(?:,\s*paragraph\s*\d+[a-z]?)?' + _SEP
    + r'|' + _REC + r'\s*' + _NUM + _SEP,
    re.IGNORECASE,
)


def clean_citations(text: str) -> str:
    out = STRIP_RE.sub('', text)
    out = re.sub(r'[ \t]{2,}', ' ', out)
    out = re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()


# ── Supabase helpers ───────────────────────────────────────────────────────────
def paginate(supabase, table: str, select: str, filter_fn=None, page_size: int = 1000):
    rows, offset = [], 0
    while True:
        q = supabase.table(table).select(select)
        if filter_fn:
            q = filter_fn(q)
        page = q.range(offset, offset + page_size - 1).execute().data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def batch_in(supabase, table: str, select: str, column: str, values: list, batch_size: int = 100):
    rows = []
    for i in range(0, len(values), batch_size):
        batch = values[i : i + batch_size]
        rows.extend(
            supabase.table(table).select(select).in_(column, batch).execute().data or []
        )
    return rows


# ── Embedding ──────────────────────────────────────────────────────────────────
def embed(model, texts: list[str], prefix: str = "", batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    prefixed = [prefix + t for t in texts] if prefix else texts
    return model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def load_or_embed(
    model,
    texts: list[str],
    prefix: str,
    cache_dir: Path,
    role: str,
) -> np.ndarray:
    """Return embeddings from disk cache if the texts are unchanged, else re-embed and save."""
    digest    = hashlib.sha256((prefix + "\n".join(texts)).encode()).hexdigest()[:20]
    cache_file = cache_dir / f"emb_{role}_{digest}.npy"
    if cache_file.exists():
        arr = np.load(cache_file)
        print(f"  Loaded cached {role} embeddings ({arr.shape[0]} × {arr.shape[1]}) ← {cache_file.name}")
        return arr
    print(f"  Embedding {len(texts)} {role} texts...")
    arr = embed(model, texts, prefix=prefix)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    print(f"  Cached {role} embeddings → {cache_file.name}")
    return arr


# ── Reciprocal matching ────────────────────────────────────────────────────────
def reciprocal_match(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_meta: list[dict],
    k: int,
    min_score: float,
    k_recip: int | None,
) -> list[list[dict]]:
    """Return up to k corpus matches per query, filtered by score and mutual ranking.

    A pair (query_i, corpus_j) survives only if:
      - corpus_j is in query_i's top-k by cosine similarity (>= min_score)
      - query_i is in corpus_j's top-k_recip queries (reverse direction)
    """
    if query_embs is None or len(query_embs) == 0:
        return []

    sim   = query_embs @ corpus_embs.T   # (n_queries, n_corpus)
    n_q, n_c = sim.shape

    if k_recip is not None and n_q >= k_recip:
        k_r      = min(k_recip, n_q)
        rev_top  = np.argpartition(sim, -k_r, axis=0)[-k_r:, :]   # (k_r, n_c)
        rev_mask = np.zeros((n_q, n_c), dtype=bool)
        for aj in range(n_c):
            rev_mask[rev_top[:, aj], aj] = True
    else:
        rev_mask = None

    results = []
    for qi in range(n_q):
        scores = sim[qi]
        ranked = np.argsort(scores)[::-1]
        matches: list[dict] = []
        for ci in ranked:
            s = float(scores[ci])
            if s < min_score:
                break
            if len(matches) >= k:
                break
            if rev_mask is not None and not rev_mask[qi, ci]:
                continue
            matches.append({**corpus_meta[ci], "score": round(s, 6), "rank": len(matches) + 1})
        results.append(matches)
    return results


# ── LLM prompt (Prompt G — chain-of-thought + explicit NOISE gate) ────────────
CLASSIFY_TOOL = {
    "name": "classify_match",
    "description": (
        "Classify whether a lobbying organisation's pre-proposal text aligns with, "
        "opposes, is undetectably related to, or is noise relative to a legislative "
        "provision in the Commission proposal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "provision_effect": {
                "type": "string",
                "description": (
                    "One sentence describing what the legislative provision establishes "
                    "or requires (e.g. 'establishes mandatory safety stock requirements')."
                ),
            },
            "label": {"type": "string", "enum": LABELS},
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the classification.",
            },
        },
        "required": ["provision_effect", "label", "reasoning"],
    },
}

_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal text and a
provision in a European Commission legislative proposal (article or recital).

Your task is to assess whether the provision responds to what the organisation
advocated for — not merely whether both discuss the same topic.
Base your classification solely on the text provided. Do not infer positions or
connections beyond what is explicitly stated.

You are given:
  • LEGISLATIVE PROVISION — the article or recital as it appears in the proposal
  • ORG POSITION          — what the organisation expressed before the proposal
                            was tabled (feedback submission or commissioner meeting)

Classify via the tool:
  ALIGNED       — there is a clear tie between what the org advocates for and what the
                  provision does. Two paths to ALIGNED:
                  (a) The org makes a specific ask and the provision delivers it (or more
                      of it in the same direction).
                  (b) The org advocates for action/initiative in a specific direction, and
                      the provision takes concrete action in that direction — not merely
                      defines or neutrally describes the area. If you can argue a clear
                      connection between the org's position and the provision's action,
                      classify ALIGNED.
  OPPOSING      — the org explicitly argues against what the provision establishes,
                  or the provision does the opposite of what the org asked for.
  UNDETECTABLE  — the org has a specific advocacy position related to the provision's
                  subject, but it is genuinely unclear whether the provision satisfies,
                  contradicts, or ignores it. Use this when you can see a real connection
                  but cannot confidently determine direction. If uncertain → fall back here.
  NOISE         — the org text contains no substantive advocacy position on this topic:
                  boilerplate, background descriptions of existing law, administrative text,
                  general endorsements without a specific stance
                  (e.g. "we support EU action on medicines", "we welcome the initiative"),
                  OR the subjects are unrelated.

Critical distinctions:
  • An org that argues a mechanism should NOT EXIST is OPPOSING the provision that
    creates it, even if the provision also includes safeguards the org likes.
  • A provision that merely defines a term, lists background context, or describes
    the current state of affairs is NOT taking action — path (b) for ALIGNED does not
    apply to purely definitional or recital provisions that take no concrete stance.
  • General values ("we support supply chain resilience") with no specific direction
    → NOISE. A position needs to be specific enough that you can evaluate whether the
    provision responds to it.
  • UNDETECTABLE is the fallback when a genuine connection exists but direction is
    ambiguous — not when the org text is simply vague or general (that is NOISE).\
"""

_COT = (
    "\n\nBefore classifying, think step by step:\n"
    "  1. What does the provision specifically establish, require, or do? "
    "State the concrete action — or note if it only defines/describes.\n"
    "  2. Does the org text express a specific advocacy position on this subject "
    "(not just a general value or endorsement)?\n"
    "     If only general support/concern with no directional stance → NOISE (stop here).\n"
    "  3. Counterfactual test: could an org with the OPPOSITE general stance on this topic "
    "produce a text that would also match this provision in the same direction?\n"
    "     If yes — the org's position is too broad to constitute specific alignment → NOISE.\n"
    "     Example: 'we support supply chain resilience' would match any supply chain provision "
    "regardless of the org's actual position, so it fails this test.\n"
    "  4. Is there a clear and specific tie between the org's position and the provision's action?\n"
    "     • Clear specific tie, provision acts in the direction the org advocated → ALIGNED\n"
    "     • The org opposes this mechanism or asks for the opposite → OPPOSING\n"
    "     • Real connection exists but direction is genuinely ambiguous → UNDETECTABLE"
)


def build_prompt_G(organisation: str, article_text: str, source_text: str) -> str:
    body = (
        "\n---"
        f"\n\nLEGISLATIVE PROVISION:\n{article_text}"
        f"\n\nORG POSITION:\n{source_text}"
    )
    return f"Organisation: {organisation}\n\n{_INTRO}{_COT}{body}"


def classify_pair(client: anthropic.Anthropic, prompt: str) -> dict | None:
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0,
        )
        tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
        return tool_block.input if tool_block else None
    except (anthropic.APIError, anthropic.APITimeoutError) as exc:
        print(f"    API error: {exc}")
        return None


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_proposal_articles(supabase, proc_id: str) -> list[dict]:
    rows = paginate(
        supabase, "procedure_articles",
        "element_type, element_number, title, content, sort_order",
        lambda q: (
            q.eq("procedure_id", proc_id)
             .eq("document_version", "proposal")
             .neq("element_type", "recital")
             .order("sort_order")
        ),
    )
    print(f"  Proposal articles:        {len(rows)}  (recitals excluded)")
    return rows


def load_hys_chunks(supabase, proc_id: str, proposal_date: str) -> list[dict]:
    rows = paginate(
        supabase, "hys_feedback_chunks",
        "id, feedback_id, chunk_index, chunk_text, organisation_name, "
        "transparency_reg_id, date_feedback",
        lambda q: q.eq("procedure_id", proc_id).lt("date_feedback", proposal_date),
    )
    rows.sort(key=lambda r: r["id"])
    print(f"  HYS chunks (raw):         {len(rows)}")
    return rows


def load_commission_meetings(
    supabase, proc_id: str, proposal_date: str
) -> tuple[list[dict], dict[str, list[dict]]]:
    CM_COLS = "id, commissioner_name, meeting_date, subject, points_raised, conclusions"
    meetings_by_id: dict[str, dict] = {}
    offset, page_size = 0, 1000

    while True:
        page = (
            supabase.table("meeting_procedure_links")
            .select(f"commission_meetings({CM_COLS})")
            .eq("procedure_id", proc_id)
            .not_.is_("commission_meeting_id", "null")
            .range(offset, offset + page_size - 1)
            .execute()
            .data or []
        )
        for row in page:
            m = row.get("commission_meetings")
            if m and m.get("id"):
                meetings_by_id[m["id"]] = m
        if len(page) < page_size:
            break
        offset += page_size

    cm_all      = sorted(meetings_by_id.values(), key=lambda m: m["id"])
    cm_filtered = [
        m for m in cm_all
        if m.get("meeting_date") and m["meeting_date"] < proposal_date
    ]
    print(f"  Commission meetings:       {len(cm_filtered)} pre-proposal / {len(cm_all)} total")

    cm_ids      = [m["id"] for m in cm_filtered]
    cm_org_rows = batch_in(
        supabase, "commission_meeting_organizations",
        "meeting_id, organization_name, eu_transparency_register_id",
        "meeting_id", cm_ids,
    ) if cm_ids else []

    org_map: dict[str, list[dict]] = defaultdict(list)
    for o in cm_org_rows:
        org_map[o["meeting_id"]].append({
            "name":  o.get("organization_name") or "",
            "tr_id": o.get("eu_transparency_register_id"),
        })

    return cm_filtered, dict(org_map)


def load_lobbying_meetings(
    supabase, proc_id: str, proposal_date: str
) -> list[dict]:
    """Load EP lobbying meetings (MEP meetings) pre-proposal, with org name resolved."""
    LM_COLS = "id, organization_id, meeting_date"
    lm_by_id: dict[str, dict] = {}
    offset, page_size = 0, 1000

    while True:
        page = (
            supabase.table("meeting_procedure_links")
            .select(f"lobbying_meetings({LM_COLS})")
            .eq("procedure_id", proc_id)
            .not_.is_("lobbying_meeting_id", "null")
            .range(offset, offset + page_size - 1)
            .execute()
            .data or []
        )
        for row in page:
            m = row.get("lobbying_meetings")
            if m and m.get("id"):
                lm_by_id[m["id"]] = m
        if len(page) < page_size:
            break
        offset += page_size

    lm_filtered = sorted(
        (m for m in lm_by_id.values() if m.get("meeting_date") and m["meeting_date"] < proposal_date),
        key=lambda m: m["id"],
    )
    print(f"  EP lobbying meetings:      {len(lm_filtered)} pre-proposal / {len(lm_by_id)} total")

    org_ids = list({m["organization_id"] for m in lm_filtered if m.get("organization_id")})
    orgs    = batch_in(
        supabase, "organizations",
        "id, name, eu_transparency_register_id",
        "id", org_ids,
    ) if org_ids else []
    org_map = {o["id"]: o for o in orgs}

    enriched = []
    for m in lm_filtered:
        org = org_map.get(m.get("organization_id"), {})
        enriched.append({
            "organisation": org.get("name"),
            "tr_id":        org.get("eu_transparency_register_id"),
        })
    return enriched


def find_procedure_alias_meetings(
    supabase,
    proposal_date: str,
    already_linked_ids: set[str],
    aliases: list[str],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Find commission meetings that mention procedure aliases but weren't formally linked.

    Loads all commission meetings before proposal_date, searches their text
    (subject, points_raised, conclusions) for any alias string, and returns the
    ones not already in already_linked_ids together with their org associations.

    These are included in both the source pool and access counts — they represent
    real engagement with this procedure that wasn't captured in meeting_procedure_links.
    """
    if not aliases:
        return [], {}

    lower_aliases = [a.lower() for a in aliases]

    CM_COLS = "id, commissioner_name, meeting_date, subject, points_raised, conclusions"
    all_meetings: list[dict] = []
    offset, page_size = 0, 1000

    print(f"  Loading all commission meetings before {proposal_date} for alias search...")
    while True:
        page = (
            supabase.table("commission_meetings")
            .select(CM_COLS)
            .lt("meeting_date", proposal_date)
            .range(offset, offset + page_size - 1)
            .execute()
            .data or []
        )
        all_meetings.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    print(f"  Total meetings scanned: {len(all_meetings)}")

    matched: list[dict] = []
    for m in sorted(all_meetings, key=lambda x: x["id"]):
        if m["id"] in already_linked_ids:
            continue
        haystack = " ".join(filter(None, [
            m.get("subject") or "",
            m.get("points_raised") or "",
            m.get("conclusions") or "",
        ])).lower()
        if any(alias in haystack for alias in lower_aliases):
            m["alias_match"] = True
            matched.append(m)

    print(f"  Alias-matched (unlinked) meetings: {len(matched)}")
    if not matched:
        return [], {}

    # Load org associations for newly found meetings
    matched_ids = [m["id"] for m in matched]
    org_rows    = batch_in(
        supabase, "commission_meeting_organizations",
        "meeting_id, organization_name, eu_transparency_register_id",
        "meeting_id", matched_ids,
    )
    org_map: dict[str, list[dict]] = defaultdict(list)
    for o in org_rows:
        org_map[o["meeting_id"]].append({
            "name":  o.get("organization_name") or "",
            "tr_id": o.get("eu_transparency_register_id"),
        })

    return matched, dict(org_map)


def build_access_counts(
    cm_rows: list[dict],
    cm_org_map: dict[str, list[dict]],
    lm_enriched: list[dict],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, str]]:
    """Return (total_count, cm_count, ep_count) per org key.

    Org key is the TR ID when available, falling back to lowercased name.
    Also returns a tr_id → canonical_name map for label resolution.
    """
    cm_counts:    dict[str, int] = defaultdict(int)
    ep_counts:    dict[str, int] = defaultdict(int)
    key_to_name:  dict[str, str] = {}

    def org_key(name: str | None, tr_id: str | None) -> str | None:
        if tr_id:
            return tr_id
        if name:
            return name.lower().strip()
        return None

    for m in cm_rows:
        for o in cm_org_map.get(m["id"], []):
            k = org_key(o.get("name"), o.get("tr_id"))
            if k:
                cm_counts[k] += 1
                if o.get("name"):
                    key_to_name[k] = o["name"]

    for o in lm_enriched:
        k = org_key(o.get("organisation"), o.get("tr_id"))
        if k:
            ep_counts[k] += 1
            if o.get("organisation"):
                key_to_name[k] = o["organisation"]

    all_keys   = set(cm_counts) | set(ep_counts)
    total      = {k: cm_counts.get(k, 0) + ep_counts.get(k, 0) for k in all_keys}
    return total, dict(cm_counts), dict(ep_counts), key_to_name


# ── Source pool builder ────────────────────────────────────────────────────────
def build_source_pool(
    hys_rows: list[dict],
    cm_rows: list[dict],
    cm_org_map: dict[str, list[dict]],
) -> list[dict]:
    """Flatten all pre-proposal sources into a unified list of dicts.

    cm_rows may include both formally linked and procedure-alias-matched meetings;
    any meeting with alias_match=True is flagged in the output.
    """
    pool: list[dict] = []

    chunk_lookup = {
        (r["feedback_id"], r["chunk_index"]): r["chunk_text"]
        for r in hys_rows
    }

    for r in hys_rows:
        text = clean_citations(r.get("chunk_text") or "")
        if len(text) < MIN_CHUNK_LEN:
            continue
        fi, ci = r["feedback_id"], r["chunk_index"]
        pool.append({
            "text":                text,
            "source_type":         "hys_feedback",
            "organisation":        r.get("organisation_name"),
            "transparency_reg_id": r.get("transparency_reg_id"),
            "source_date":         str(r.get("date_feedback") or "")[:10] or None,
            "commissioner":        None,
            "context_before":      chunk_lookup.get((fi, ci - 1)) if ci > 0 else None,
            "context_after":       chunk_lookup.get((fi, ci + 1)),
            "chunk_id":            r.get("id"),
            "alias_match":         False,
        })

    for m in cm_rows:
        parts: list[str] = []
        if m.get("subject"):
            parts.append(f"Subject: {m['subject']}")
        if m.get("points_raised"):
            parts.append(m["points_raised"])
        if m.get("conclusions"):
            parts.append(f"Conclusions: {m['conclusions']}")
        text = " ".join(parts).strip()
        if len(text) < MIN_CHUNK_LEN:
            continue

        orgs = cm_org_map.get(m["id"]) or [{"name": None, "tr_id": None}]
        for org in orgs:
            pool.append({
                "text":                text,
                "source_type":         "commission_meeting",
                "organisation":        org.get("name"),
                "transparency_reg_id": org.get("tr_id"),
                "source_date":         str(m.get("meeting_date") or "")[:10] or None,
                "commissioner":        m.get("commissioner_name"),
                "context_before":      None,
                "context_after":       None,
                "chunk_id":            None,
                "alias_match":         bool(m.get("alias_match")),
            })

    hys_n   = sum(1 for p in pool if p["source_type"] == "hys_feedback")
    cm_n    = sum(1 for p in pool if p["source_type"] == "commission_meeting" and not p["alias_match"])
    alias_n = sum(1 for p in pool if p.get("alias_match"))
    print(f"  Source pool:              {len(pool)} ({hys_n} HYS, {cm_n} commission, {alias_n} procedure-alias)")
    return pool


def article_to_passage_text(a: dict) -> str:
    header = f"{a['element_type']} {a['element_number']}"
    if a.get("title"):
        header += f": {a['title']}"
    return f"{header}\n{a.get('content', '')}".strip()


def article_to_llm_text(a: dict) -> str:
    """Full article text for the LLM prompt, including element header."""
    header = f"{a['element_type'].title()} {a['element_number']}"
    if a.get("title"):
        header += f" — {a['title']}"
    return f"{header}\n{a.get('content', '')}".strip()


# ── Cache key ──────────────────────────────────────────────────────────────────
def make_cache_key(source_text: str, element_type: str, element_number: str) -> str:
    digest = hashlib.sha256(source_text.encode()).hexdigest()[:20]
    return f"{digest}_{element_type}_{element_number}"


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_preproposal_pipeline(
    proc_id: str,
    output_dir: Path,
    rate_limit_sleep: float = 0.4,
) -> pd.DataFrame:
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Procedure metadata ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Procedure: {proc_id}")
    print(f"{'='*60}")

    proc = (
        supabase.table("procedures")
        .select("title, proposal_date")
        .eq("id", proc_id)
        .single()
        .execute()
        .data
    )
    if not proc:
        raise ValueError(f"Procedure {proc_id!r} not found.")

    proposal_date = proc["proposal_date"]
    print(f"  Title:         {proc['title']}")
    print(f"  Proposal date: {proposal_date}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data from database...")
    art_rows            = load_proposal_articles(supabase, proc_id)
    hys_rows            = load_hys_chunks(supabase, proc_id, proposal_date)
    cm_rows, cm_org_map = load_commission_meetings(supabase, proc_id, proposal_date)

    if not art_rows:
        raise ValueError(
            f"No proposal articles found for {proc_id!r}.\n"
            "Run scripts/chunk_procedure_articles.py first."
        )

    lm_enriched = load_lobbying_meetings(supabase, proc_id, proposal_date)

    if PROCEDURE_ALIASES:
        print("\nSearching for procedure-alias meetings (unlinked)...")
        already_linked = {m["id"] for m in cm_rows}
        alias_cm_rows, alias_cm_org_map = find_procedure_alias_meetings(
            supabase, proposal_date, already_linked, PROCEDURE_ALIASES
        )
        cm_rows    = cm_rows + alias_cm_rows
        cm_org_map = {**cm_org_map, **alias_cm_org_map}
    else:
        print("  Procedure aliases: none configured — skipping alias search.")

    total_counts, cm_counts, ep_counts, key_to_name = build_access_counts(
        cm_rows, cm_org_map, lm_enriched
    )
    n_orgs_with_meetings = len(total_counts)
    print(f"  Orgs with any pre-proposal meetings: {n_orgs_with_meetings}")

    pool = build_source_pool(hys_rows, cm_rows, cm_org_map)
    if not pool:
        raise ValueError("No pre-proposal source texts found.")

    # ── Embed (with disk cache) ───────────────────────────────────────────────
    proc_slug = proc_id.replace("/", ":").replace("(", "").replace(")", "")
    emb_cache_dir = output_dir / proc_slug / "emb_cache"

    print(f"\nLoading {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)

    art_passage_texts = [article_to_passage_text(a) for a in art_rows]
    art_llm_texts     = [article_to_llm_text(a) for a in art_rows]
    art_meta = [
        {
            "element_type":   a["element_type"],
            "element_number": a["element_number"],
            "article_title":  a.get("title"),
            "article_text":   art_llm_texts[i],
        }
        for i, a in enumerate(art_rows)
    ]

    art_embs = load_or_embed(model, art_passage_texts, PASSAGE_PREFIX, emb_cache_dir, "articles")

    src_texts = [p["text"] for p in pool]
    src_embs  = load_or_embed(model, src_texts, QUERY_PREFIX, emb_cache_dir, "sources")

    # ── Reciprocal matching ───────────────────────────────────────────────────
    print(
        f"\nRunning reciprocal matching "
        f"(K={K}, min_score={MIN_SCORE}, K_recip={K_RECIP})..."
    )
    all_matches = reciprocal_match(src_embs, art_embs, art_meta, K, MIN_SCORE, K_RECIP)
    n_pairs     = sum(len(m) for m in all_matches)
    print(f"  Surviving pairs: {n_pairs}")

    # ── Build flat pair list ──────────────────────────────────────────────────
    def lookup_counts(tr_id: str | None, name: str | None) -> tuple[int, int, int]:
        k = tr_id if tr_id else (name.lower().strip() if name else None)
        if k and k in total_counts:
            return total_counts[k], cm_counts.get(k, 0), ep_counts.get(k, 0)
        return 0, 0, 0

    pairs: list[dict] = []
    for src_i, matches in enumerate(all_matches):
        src = pool[src_i]
        total, cm_n, ep_n = lookup_counts(src["transparency_reg_id"], src["organisation"])
        for cand in matches:
            pairs.append({
                "id":                       str(uuid.uuid4()),
                "procedure_id":             proc_id,
                "source_type":              src["source_type"],
                "organisation":             src["organisation"],
                "transparency_reg_id":      src["transparency_reg_id"],
                "source_date":              src["source_date"],
                "commissioner":             src["commissioner"],
                "source_text":              src["text"],
                "context_before":           src["context_before"],
                "context_after":            src["context_after"],
                "chunk_id":                 src["chunk_id"],
                "preproposal_meetings_total":       total,
                "preproposal_meetings_commission":  cm_n,
                "preproposal_meetings_ep":          ep_n,
                "alias_match":              src.get("alias_match", False),
                "article_type":             cand["element_type"],
                "article_number":           cand["element_number"],
                "article_title":            cand["article_title"],
                "article_text":             cand["article_text"],
                "similarity_score":         cand["score"],
                "provision_effect":         None,
                "label":                    None,
                "reasoning":               None,
            })

    # ── LLM classification ────────────────────────────────────────────────────
    cache_path = output_dir / proc_slug / "preproposal_llm_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"  Loaded {len(cache)} cached LLM responses.")
        except json.JSONDecodeError:
            print("  Cache corrupted — starting fresh.")

    def save_cache() -> None:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    print(f"\nClassifying {len(pairs)} pairs with {LLM_MODEL} (Prompt G, temp={TEMPERATURE})...")
    for i, pair in enumerate(pairs):
        cache_key = make_cache_key(
            pair["source_text"], pair["article_type"], pair["article_number"]
        )

        if cache_key in cache:
            result = cache[cache_key]
        else:
            prompt = build_prompt_G(
                organisation=pair["organisation"] or "Unknown Organisation",
                article_text=pair["article_text"],
                source_text=pair["source_text"],
            )
            result = classify_pair(client, prompt)
            if result is not None:
                cache[cache_key] = result
                save_cache()
            time.sleep(rate_limit_sleep)

        if result:
            pair["label"]            = (result.get("label") or "NOISE").upper()
            pair["provision_effect"] = result.get("provision_effect")
            pair["reasoning"]        = result.get("reasoning")
        else:
            pair["label"] = "NOISE"

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            done = i + 1
            cached = sum(1 for p in pairs[:done] if p["label"] is not None)
            print(f"  [{done}/{len(pairs)}]  last={pair['label']}")

    # ── Save output ───────────────────────────────────────────────────────────
    out_dir  = output_dir / proc_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    today    = date.today().isoformat()

    df       = pd.DataFrame(pairs)
    csv_path = out_dir / f"preproposal_{today}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # ── Deduplicated dyad table ───────────────────────────────────────────────
    # Unit of analysis: (org, article) dyad — one org cannot align with or oppose
    # the same article more than once, regardless of how many source chunks matched it.
    # Resolution: most specific label wins (ALIGNED/OPPOSING > UNDETECTABLE > NOISE).
    # Conflict: if the same dyad has both ALIGNED and OPPOSING, flag it.
    LABEL_RANK = {"ALIGNED": 0, "OPPOSING": 1, "UNDETECTABLE": 2, "NOISE": 3}

    dyad_rows = []
    dyad_key  = ["organisation", "transparency_reg_id", "article_type", "article_number"]
    for key, grp in df.groupby(dyad_key, dropna=False):
        labels   = set(grp["label"].dropna())
        conflict = "ALIGNED" in labels and "OPPOSING" in labels
        # Pick most specific label; prefer the row with highest similarity score
        best_label = sorted(labels, key=lambda l: LABEL_RANK.get(l, 99))[0]
        best_row   = grp[grp["label"] == best_label].sort_values(
            "similarity_score", ascending=False
        ).iloc[0]
        dyad_rows.append({
            **best_row[["organisation", "transparency_reg_id",
                         "source_type", "preproposal_meetings_total",
                         "preproposal_meetings_commission", "preproposal_meetings_ep",
                         "article_type", "article_number",
                         "article_title", "similarity_score",
                         "provision_effect", "label", "reasoning"]].to_dict(),
            "n_chunks":  len(grp),
            "conflict":  conflict,
        })

    dyad_df      = pd.DataFrame(dyad_rows)
    dyad_csv     = out_dir / f"preproposal_{today}_dyads.csv"
    dyad_df.to_csv(dyad_csv, index=False)
    print(f"Saved dyad CSV → {dyad_csv}  ({len(dyad_df)} dyads from {len(df)} raw pairs)")

    conflicts = dyad_df[dyad_df["conflict"]]
    if len(conflicts):
        print(f"  ⚠ Conflict dyads (ALIGNED + OPPOSING same org×article): {len(conflicts)}")
        for _, r in conflicts.iterrows():
            print(f"    {r['organisation']} × {r['article_type']} {r['article_number']}")

    meta = {
        "procedure_id":          proc_id,
        "title":                 proc["title"],
        "proposal_date":         proposal_date,
        "stage":                 "preproposal",
        "embedding_model":       MODEL_ID,
        "llm_model":             LLM_MODEL,
        "llm_temperature":       TEMPERATURE,
        "k":                     K,
        "k_recip":               K_RECIP,
        "min_score":             MIN_SCORE,
        "generated":             today,
        "total_pairs":           len(pairs),
        "total_dyads":           len(dyad_df),
        "conflict_dyads":        int(conflicts.shape[0]),
        "orgs_with_meetings":    n_orgs_with_meetings,
        "label_distribution":    df["label"].value_counts().to_dict(),
        "dyad_label_distribution": dyad_df["label"].value_counts().to_dict(),
        "source_distribution":   df["source_type"].value_counts().to_dict(),
    }
    json_path = out_dir / f"preproposal_{today}.json"
    json_path.write_text(
        json.dumps({"metadata": meta, "pairs": pairs}, indent=2, ensure_ascii=False)
    )
    print(f"Saved JSON → {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {proc_id}")
    print(f"{'='*60}")
    print(f"  Raw pairs:    {len(df)}")
    print(f"  Dyads:        {len(dyad_df)}  (unique org × article)")
    for label, cnt in df["label"].value_counts().items():
        d = dyad_df["label"].value_counts().get(label, 0)
        print(f"    {label:<14} {cnt:>4} raw  →  {d:>4} dyads")

    signal = dyad_df[dyad_df["label"].isin(["ALIGNED", "OPPOSING"])].copy()
    print(f"\n  Signal dyads (ALIGNED + OPPOSING): {len(signal)}")

    aligned_dyads = dyad_df[dyad_df["label"] == "ALIGNED"]
    if len(aligned_dyads):
        print("\n  Top orgs by ALIGNED dyads:")
        for org, cnt in aligned_dyads["organisation"].value_counts().head(10).items():
            print(f"    {cnt:>4}  {org}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preproposal alignment pipeline — embed, match, classify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--procedure", required=True,
        help="OEIL procedure ID, e.g. '2021/0106(COD)'",
    )
    parser.add_argument(
        "--output-dir", default="analysis",
        help="Root output directory (default: analysis/)",
    )
    parser.add_argument(
        "--rate-limit-sleep", type=float, default=0.4,
        help="Seconds to sleep between LLM API calls (default: 0.4)",
    )
    args = parser.parse_args()

    run_preproposal_pipeline(
        proc_id=args.procedure,
        output_dir=Path(args.output_dir),
        rate_limit_sleep=args.rate_limit_sleep,
    )


if __name__ == "__main__":
    main()
