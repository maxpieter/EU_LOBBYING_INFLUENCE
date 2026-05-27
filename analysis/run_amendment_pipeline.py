#!/usr/bin/env python3
"""Post-proposal amendment alignment pipeline.

For a given procedure, embeds post-proposal HYS feedback against EP amendments
using intfloat/multilingual-e5-large, then classifies each surviving reciprocal
match with Claude (amendment direction-first CoT).

Source window: HYS feedback submitted on or after proposal_date.
Corpus:        procedure_amendments (amended_text, falling back to original_text).

Outputs under analysis/<proc_slug>/:
  amendment_<date>.csv              — flat pair-level results
  amendment_<date>_dyads.csv        — deduplicated (org × amendment) dyads
  amendment_<date>.json             — metadata + all pairs
  amendment_llm_cache.json          — LLM response cache

Usage:
    python scripts/run_amendment_pipeline.py --procedure "2021/0106(COD)"
    python scripts/run_amendment_pipeline.py --procedure "2025/0102(COD)" --output-dir analysis/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
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

K             = 5     # top-k amendments per source text before reciprocal filter
MIN_SCORE     = 0.84  # minimum cosine similarity
K_RECIP       = 10    # reverse top-K for reciprocal filter
MIN_CHUNK_LEN = 80    # skip texts shorter than this

LLM_MODEL   = "claude-sonnet-4-6"
MAX_TOKENS  = 800
TEMPERATURE = 0.0
LABELS      = ["ALIGNED", "OPPOSING", "UNDETECTABLE", "NOISE"]


# ── Citation stripping ─────────────────────────────────────────────────────────
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


def load_or_embed(model, texts: list[str], prefix: str, cache_dir: Path, role: str) -> np.ndarray:
    digest     = hashlib.sha256((prefix + "\n".join(texts)).encode()).hexdigest()[:20]
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
    if query_embs is None or len(query_embs) == 0:
        return []

    sim    = query_embs @ corpus_embs.T
    n_q, n_c = sim.shape

    if k_recip is not None and n_q >= k_recip:
        k_r      = min(k_recip, n_q)
        rev_top  = np.argpartition(sim, -k_r, axis=0)[-k_r:, :]
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
            matches.append({**corpus_meta[ci], "score": round(s, 6)})
        results.append(matches)
    return results


# ── LLM prompt (amendment direction-first CoT) ────────────────────────────────
CLASSIFY_TOOL = {
    "name": "classify_match",
    "description": (
        "Classify whether a lobbying organisation's text aligns with, opposes, "
        "is undetectably related to, or is noise relative to a parliamentary amendment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "provision_effect": {
                "type": "string",
                "description": "One sentence describing what the amendment changes relative to the original.",
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

_AMD_INTRO = """\
You are given a match between a lobbying organisation's pre-proposal text and a
parliamentary amendment to a European Commission legislative proposal.

Your task is to assess whether the organisation's expressed position advocates
for the change the amendment makes — not merely whether they discuss the same topic.

You are given:
  • ORIGINAL TEXT  — the provision as it appeared in the Commission proposal
                     (omitted if the amendment is a new insertion)
  • AMENDED TO     — the text as revised by the amendment
  • JUSTIFICATION  — the amendment's stated rationale (may be absent)
  • ORG POSITION   — what the organisation expressed before the proposal was tabled

The organisation does not need to mention the specific article. What matters is
whether their expressed position advocates for the direction of change the amendment
makes to the original text.

Classify via the tool:
  ALIGNED       — the amendment moves the original text in the direction the org
                  advocated for. The org's position must be specific enough that
                  you can say the amendment responds to it.
  OPPOSING      — the amendment moves the original text in the opposite direction
                  from what the org advocated for, or the org explicitly opposes
                  what the amendment introduces.
  UNDETECTABLE  — the org has a specific position related to this area, but it is
                  genuinely unclear whether the amendment's direction satisfies,
                  contradicts, or ignores it. Use when a real connection exists
                  but direction is ambiguous. Fall back here when uncertain.
  NOISE         — the org text contains no substantive advocacy position on this
                  topic: boilerplate, background descriptions, general endorsements
                  without a specific directional stance
                  (e.g. "we support this initiative", "we welcome EU action"),
                  OR the subjects are unrelated.

Critical distinctions:
  • The direction of the amendment is what matters — not whether both texts discuss
    the same subject. Determine first what the amendment ADDS, REMOVES, STRENGTHENS,
    or WEAKENS relative to the original, then evaluate the org's position against that.
  • General support for the policy area → NOISE, not UNDETECTABLE.
  • Counterfactual test: could an org with the OPPOSITE stance produce text that
    also matches this amendment in the same direction? If yes, the org's position
    is too broad → NOISE.
  • UNDETECTABLE is the fallback when a genuine directional connection exists but
    the relationship is ambiguous. It is NOT the default when the org text is vague.\
"""

_AMD_COT = (
    "\n\nBefore classifying, think step by step:\n"
    "  1. What does the ORIGINAL TEXT establish? If no original, note this is a new insertion.\n"
    "  2. What does the AMENDMENT change it to? What specifically does it add, remove, "
    "strengthen, weaken, or clarify?\n"
    "  3. State the DIRECTION of change in one sentence "
    "(e.g. 'the amendment makes X mandatory where it was optional', "
    "'the amendment removes the obligation to Y').\n"
    "  4. Does the org text express a specific advocacy position on this subject — "
    "not just general support or concern?\n"
    "     If only general endorsement with no directional stance → NOISE (stop here).\n"
    "  5. Counterfactual: could an org with the opposite position also match this amendment "
    "in the same direction? If yes → NOISE.\n"
    "  6. Does the org's position advocate for pushing the original text in this direction?\n"
    "     • Yes, the amendment goes where the org wanted → ALIGNED\n"
    "     • The org wanted the opposite direction → OPPOSING\n"
    "     • Real connection but genuinely unclear whether this direction satisfies the org → UNDETECTABLE"
)


def build_amendment_prompt(
    organisation: str,
    original_text: str | None,
    amended_text: str,
    justification: str | None,
    source_text: str,
    context_before: str | None,
    context_after: str | None,
) -> str:
    parts = ["\n---"]
    if original_text and original_text.strip():
        parts.append(f"\nORIGINAL TEXT:\n{original_text.strip()}")
    else:
        parts.append("\nORIGINAL TEXT:\n(new insertion — no original text)")
    parts.append(f"\nAMENDED TO:\n{amended_text.strip()}")
    if justification and justification.strip():
        parts.append(f"\nJUSTIFICATION:\n{justification.strip()}")
    if context_before and context_before.strip():
        parts.append(f"\nORG POSITION — PRECEDING CONTEXT:\n{context_before.strip()}")
    parts.append(f"\nORG POSITION — MATCHED CHUNK:\n{source_text.strip()}")
    if context_after and context_after.strip():
        parts.append(f"\nORG POSITION — FOLLOWING CONTEXT:\n{context_after.strip()}")

    body = "\n".join(parts)
    return f"Organisation: {organisation}\n\n{_AMD_INTRO}{_AMD_COT}{body}"


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
def load_amendments(supabase, proc_id: str) -> list[dict]:
    rows = paginate(
        supabase, "procedure_amendments",
        "id, amendment_number, target_element, target_type, "
        "original_text, amended_text, justification, committee",
        lambda q: q.eq("procedure_id", proc_id),
    )
    valid = [
        r for r in rows
        if len((r.get("amended_text") or r.get("original_text") or "").strip()) >= MIN_CHUNK_LEN
    ]
    print(f"  Amendments (raw):          {len(rows)}")
    print(f"  Amendments (with text):    {len(valid)}")
    return valid


def load_post_meetings(
    supabase, proc_id: str, proposal_date: str
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, str]]:
    """Load post-proposal EP lobbying + Commission meetings.

    Returns (total_count, cm_count, ep_count, key_to_name) per org key.
    Org key is TR ID when available, else lowercased name.
    Only orgs with ≥1 meeting in total pass through to alignment classification.
    """
    from collections import defaultdict

    def org_key_name(tr: str | None, name: str | None) -> tuple[str | None, str | None]:
        key = tr if tr else (name.lower().strip() if name else None)
        return key, name

    cm_counts:   dict[str, int] = defaultdict(int)
    ep_counts:   dict[str, int] = defaultdict(int)
    key_to_name: dict[str, str] = {}

    # ── Commission meetings ───────────────────────────────────────────────────
    CM_COLS = "id, meeting_date"
    cm_by_id: dict[str, dict] = {}
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
                cm_by_id[m["id"]] = m
        if len(page) < page_size:
            break
        offset += page_size

    cm_post = [m for m in cm_by_id.values()
               if m.get("meeting_date") and m["meeting_date"] >= proposal_date]
    print(f"  Commission meetings:        {len(cm_post)} post-proposal / {len(cm_by_id)} total")

    if cm_post:
        cm_ids   = [m["id"] for m in cm_post]
        org_rows = []
        for i in range(0, len(cm_ids), 100):
            org_rows.extend(
                supabase.table("commission_meeting_organizations")
                .select("meeting_id, organization_name, eu_transparency_register_id")
                .in_("meeting_id", cm_ids[i : i + 100])
                .execute()
                .data or []
            )
        for o in org_rows:
            key, name = org_key_name(o.get("eu_transparency_register_id"), o.get("organization_name"))
            if key:
                cm_counts[key] += 1
                if name:
                    key_to_name[key] = name

    # ── EP lobbying meetings ──────────────────────────────────────────────────
    LM_COLS = "id, organization_id, meeting_date"
    lm_by_id: dict[str, dict] = {}
    offset = 0
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

    lm_post = [m for m in lm_by_id.values()
               if m.get("meeting_date") and m["meeting_date"] >= proposal_date]
    print(f"  EP lobbying meetings:       {len(lm_post)} post-proposal / {len(lm_by_id)} total")

    if lm_post:
        org_ids = list({m["organization_id"] for m in lm_post if m.get("organization_id")})
        orgs    = []
        for i in range(0, len(org_ids), 100):
            orgs.extend(
                supabase.table("organizations")
                .select("id, name, eu_transparency_register_id")
                .in_("id", org_ids[i : i + 100])
                .execute()
                .data or []
            )
        org_lookup = {o["id"]: o for o in orgs}
        for m in lm_post:
            o        = org_lookup.get(m.get("organization_id"), {})
            key, name = org_key_name(o.get("eu_transparency_register_id"), o.get("name"))
            if key:
                ep_counts[key] += 1
                if name:
                    key_to_name[key] = name

    all_keys    = set(cm_counts) | set(ep_counts)
    total       = {k: cm_counts.get(k, 0) + ep_counts.get(k, 0) for k in all_keys}
    print(f"  Unique orgs with any post-proposal meeting: {len(total)}")
    return total, dict(cm_counts), dict(ep_counts), key_to_name


def load_hys_chunks_post(supabase, proc_id: str, proposal_date: str) -> list[dict]:
    rows = paginate(
        supabase, "hys_feedback_chunks",
        "id, feedback_id, chunk_index, chunk_text, organisation_name, "
        "transparency_reg_id, date_feedback",
        lambda q: q.eq("procedure_id", proc_id).gte("date_feedback", proposal_date),
    )
    rows.sort(key=lambda r: r["id"])
    print(f"  HYS chunks post-proposal:  {len(rows)}")
    return rows


def build_source_pool(
    hys_rows: list[dict],
    total_counts: dict[str, int],
    cm_counts:    dict[str, int],
    ep_counts:    dict[str, int],
) -> list[dict]:
    """Build source pool restricted to orgs with ≥1 post-proposal meeting (any type).

    Org matching: TR ID first, fallback to lowercased name.
    """
    chunk_lookup = {
        (r["feedback_id"], r["chunk_index"]): r["chunk_text"]
        for r in hys_rows
    }

    def org_key(tr_id: str | None, name: str | None) -> str | None:
        if tr_id and tr_id in total_counts:
            return tr_id
        if name and name.lower().strip() in total_counts:
            return name.lower().strip()
        return None

    pool: list[dict] = []
    skipped_no_meeting = 0

    for r in hys_rows:
        text = clean_citations(r.get("chunk_text") or "")
        if len(text) < MIN_CHUNK_LEN:
            continue

        key = org_key(r.get("transparency_reg_id"), r.get("organisation_name"))
        if key is None:
            skipped_no_meeting += 1
            continue

        fi, ci = r["feedback_id"], r["chunk_index"]
        pool.append({
            "text":                text,
            "organisation":        r.get("organisation_name"),
            "transparency_reg_id": r.get("transparency_reg_id"),
            "source_date":         str(r.get("date_feedback") or "")[:10] or None,
            "context_before":      chunk_lookup.get((fi, ci - 1)) if ci > 0 else None,
            "context_after":       chunk_lookup.get((fi, ci + 1)),
            "chunk_id":            r.get("id"),
            "meetings_total":      total_counts[key],
            "meetings_commission": cm_counts.get(key, 0),
            "meetings_ep":         ep_counts.get(key, 0),
        })

    print(f"  Source pool (org had meeting):    {len(pool)}  (skipped {skipped_no_meeting} chunks — no meeting)")
    return pool


def amendment_to_passage_text(a: dict) -> str:
    text = (a.get("amended_text") or a.get("original_text") or "").strip()
    target = (a.get("target_element") or "").strip()
    return f"{target}\n{text}".strip() if target else text


def make_cache_key(source_text: str, amendment_number: int | str) -> str:
    digest = hashlib.sha256(source_text.encode()).hexdigest()[:20]
    return f"{digest}_amd_{amendment_number}"


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_amendment_pipeline(
    proc_id: str,
    output_dir: Path,
    rate_limit_sleep: float = 0.4,
    k: int = K,
    min_score: float = MIN_SCORE,
    k_recip: int = K_RECIP,
) -> pd.DataFrame:
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"\n{'='*60}")
    print(f"  Procedure: {proc_id}")
    print(f"{'='*60}")

    _proc_res = (
        supabase.table("procedures")
        .select("title, proposal_date")
        .eq("id", proc_id)
        .execute()
    )
    proc = _proc_res.data[0] if _proc_res.data else None
    if not proc:
        raise ValueError(f"Procedure {proc_id!r} not found.")

    proposal_date = proc["proposal_date"]
    print(f"  Title:         {proc['title']}")
    print(f"  Proposal date: {proposal_date}")

    print("\nLoading data from database...")
    amd_rows                                   = load_amendments(supabase, proc_id)
    total_counts, cm_counts, ep_counts, _names = load_post_meetings(supabase, proc_id, proposal_date)
    hys_rows                                   = load_hys_chunks_post(supabase, proc_id, proposal_date)

    if not amd_rows:
        raise ValueError(f"No amendments found for {proc_id!r}.")
    if not total_counts:
        raise ValueError(f"No post-proposal meetings found for {proc_id!r}.")

    pool = build_source_pool(hys_rows, total_counts, cm_counts, ep_counts)
    if not pool:
        raise ValueError("No valid source chunks from orgs with post-proposal meetings.")

    # ── Embed ─────────────────────────────────────────────────────────────────
    proc_slug     = proc_id.replace("/", ":").replace("(", "").replace(")", "")
    emb_cache_dir = output_dir / proc_slug / "emb_cache"

    print(f"\nLoading {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)

    amd_passage_texts = [amendment_to_passage_text(a) for a in amd_rows]
    amd_meta = [
        {
            "amendment_id":     a["id"],
            "amendment_number": a["amendment_number"],
            "target_element":   a.get("target_element"),
            "target_type":      a.get("target_type"),
            "committee":        a.get("committee"),
            "original_text":    a.get("original_text"),
            "amended_text":     (a.get("amended_text") or a.get("original_text") or "").strip(),
            "justification":    a.get("justification"),
        }
        for a in amd_rows
    ]

    amd_embs  = load_or_embed(model, amd_passage_texts, PASSAGE_PREFIX, emb_cache_dir, "amendments")
    src_texts = [p["text"] for p in pool]
    src_embs  = load_or_embed(model, src_texts, QUERY_PREFIX, emb_cache_dir, "sources_post")

    # ── Reciprocal matching ───────────────────────────────────────────────────
    print(f"\nRunning reciprocal matching (K={k}, min_score={min_score}, K_recip={k_recip})...")
    all_matches = reciprocal_match(src_embs, amd_embs, amd_meta, k, min_score, k_recip)
    n_pairs     = sum(len(m) for m in all_matches)
    print(f"  Surviving pairs: {n_pairs}")

    # ── Build flat pair list ──────────────────────────────────────────────────
    pairs: list[dict] = []
    for src_i, matches in enumerate(all_matches):
        src = pool[src_i]
        for cand in matches:
            pairs.append({
                "id":                  str(uuid.uuid4()),
                "procedure_id":        proc_id,
                "source_type":         "hys_feedback",
                "organisation":        src["organisation"],
                "transparency_reg_id": src["transparency_reg_id"],
                "source_date":         src["source_date"],
                "source_text":         src["text"],
                "context_before":      src["context_before"],
                "context_after":       src["context_after"],
                "chunk_id":            src["chunk_id"],
                "meetings_total":      src["meetings_total"],
                "meetings_commission": src["meetings_commission"],
                "meetings_ep":         src["meetings_ep"],
                "amendment_id":        cand["amendment_id"],
                "amendment_number":    cand["amendment_number"],
                "target_element":      cand["target_element"],
                "target_type":         cand["target_type"],
                "committee":           cand["committee"],
                "original_text":       cand["original_text"],
                "amended_text":        cand["amended_text"],
                "justification":       cand["justification"],
                "similarity_score":    cand["score"],
                "provision_effect":    None,
                "label":               None,
                "reasoning":           None,
            })

    # ── Pre-LLM dyad deduplication ────────────────────────────────────────────
    # Keep only the highest-scoring source chunk per (org × amendment).
    # One org cannot align with the same amendment via multiple chunks —
    # deduplicating before LLM avoids paying for redundant API calls.
    dyad_seen: dict[tuple, dict] = {}
    for pair in pairs:
        key = (pair["organisation"], pair["transparency_reg_id"], pair["amendment_number"])
        if key not in dyad_seen or pair["similarity_score"] > dyad_seen[key]["similarity_score"]:
            dyad_seen[key] = pair
    pairs_deduped = list(dyad_seen.values())
    print(f"  After pre-LLM dyad dedup: {len(pairs_deduped)} pairs  (was {len(pairs)})")
    pairs = pairs_deduped

    # ── LLM classification ────────────────────────────────────────────────────
    out_dir    = output_dir / proc_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "amendment_llm_cache.json"

    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            print(f"  Loaded {len(cache)} cached LLM responses.")
        except json.JSONDecodeError:
            print("  Cache corrupted — starting fresh.")

    def save_cache() -> None:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        tmp.replace(cache_path)

    n_cached = sum(
        1 for p in pairs
        if make_cache_key(p["source_text"], p["amendment_number"]) in cache
    )
    print(f"\nClassifying {len(pairs)} pairs with {LLM_MODEL} (amendment CoT, temp={TEMPERATURE})...")
    print(f"  Cache hits pre-run: {n_cached}/{len(pairs)}")
    api_errors = 0
    for i, pair in enumerate(pairs):
        cache_key = make_cache_key(pair["source_text"], pair["amendment_number"])

        if cache_key in cache:
            result = cache[cache_key]
        else:
            prompt = build_amendment_prompt(
                organisation=pair["organisation"] or "Unknown Organisation",
                original_text=pair["original_text"],
                amended_text=pair["amended_text"],
                justification=pair["justification"],
                source_text=pair["source_text"],
                context_before=pair["context_before"],
                context_after=pair["context_after"],
            )
            result = classify_pair(client, prompt)
            if result is not None:
                cache[cache_key] = result
                save_cache()
            else:
                api_errors += 1
            time.sleep(rate_limit_sleep)  # rate-limit: only on actual API calls

        if result:
            pair["label"]            = (result.get("label") or "NOISE").upper()
            pair["provision_effect"] = result.get("provision_effect")
            pair["reasoning"]        = result.get("reasoning")
        else:
            pair["label"] = None  # API failure — exclude from output

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            print(f"  [{i + 1}/{len(pairs)}]  last={pair.get('label', 'ERROR')}  api_errors={api_errors}")

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    if api_errors:
        print(f"\n  ⚠ {api_errors} pairs had API errors and are excluded from output.")
        print(f"  Re-run the pipeline once your quota resets to classify them.")
    today    = date.today().isoformat()
    df       = pd.DataFrame([p for p in pairs if p.get("label") is not None])
    csv_path = out_dir / f"amendment_{today}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # ── Dyad table (already one row per org × amendment after pre-LLM dedup) ──
    # Pairs are already deduplicated; this step just renames/selects columns
    # and flags any residual conflicts (shouldn't occur post-dedup).
    dyad_df = df[[
        "organisation", "transparency_reg_id", "source_date",
        "meetings_total", "meetings_commission", "meetings_ep",
        "amendment_number", "target_element", "target_type",
        "committee", "similarity_score",
        "provision_effect", "label", "reasoning",
    ]].copy()
    dyad_df["n_chunks"] = 1
    dyad_df["conflict"] = False
    dyad_csv = out_dir / f"amendment_{today}_dyads.csv"
    dyad_df.to_csv(dyad_csv, index=False)
    print(f"Saved dyad CSV → {dyad_csv}  ({len(dyad_df)} dyads from {len(df)} raw pairs)")

    conflicts = dyad_df[dyad_df["conflict"]]
    if len(conflicts):
        print(f"  ⚠ Conflict dyads (ALIGNED + OPPOSING same org×amendment): {len(conflicts)}")

    # ── JSON output ───────────────────────────────────────────────────────────
    meta = {
        "procedure_id":            proc_id,
        "title":                   proc["title"],
        "proposal_date":           proposal_date,
        "stage":                   "amendment",
        "embedding_model":         MODEL_ID,
        "llm_model":               LLM_MODEL,
        "llm_temperature":         TEMPERATURE,
        "k":                       k,
        "k_recip":                 k_recip,
        "min_score":               min_score,
        "generated":               today,
        "orgs_with_meetings":       len(total_counts),
        "total_pairs":             len(pairs),
        "total_dyads":             len(dyad_df),
        "conflict_dyads":          int(conflicts.shape[0]),
        "label_distribution":      df["label"].value_counts().to_dict(),
        "dyad_label_distribution": dyad_df["label"].value_counts().to_dict(),
    }
    json_path = out_dir / f"amendment_{today}.json"
    json_path.write_text(
        json.dumps({"metadata": meta, "pairs": pairs}, indent=2, ensure_ascii=False)
    )
    print(f"Saved JSON → {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {proc_id}")
    print(f"{'='*60}")
    print(f"  Raw pairs:    {len(df)}")
    print(f"  Dyads:        {len(dyad_df)}  (unique org × amendment)")
    for label, cnt in df["label"].value_counts().items():
        d = dyad_df["label"].value_counts().get(label, 0)
        print(f"    {label:<14} {cnt:>4} raw  →  {d:>4} dyads")

    signal = dyad_df[dyad_df["label"].isin(["ALIGNED", "OPPOSING"])]
    print(f"\n  Signal dyads (ALIGNED + OPPOSING): {len(signal)}")

    aligned_dyads = dyad_df[dyad_df["label"] == "ALIGNED"]
    if len(aligned_dyads):
        print("\n  Top orgs by ALIGNED dyads:")
        for org, cnt in aligned_dyads["organisation"].value_counts().head(10).items():
            print(f"    {cnt:>4}  {org}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-proposal amendment alignment pipeline — embed, match, classify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--procedure", required=True,
        help="OEIL procedure ID, e.g. '2021/0106(COD)'",
    )
    parser.add_argument(
        "--output-dir", default="analysis_results",
        help="Root output directory (default: analysis_results/)",
    )
    parser.add_argument(
        "--rate-limit-sleep", type=float, default=0.4,
        help="Seconds to sleep between LLM API calls (default: 0.4)",
    )
    parser.add_argument(
        "--k", type=int, default=K,
        help=f"Top-K amendments per source text in forward retrieval (default: {K}). "
             f"Lower → fewer matches.",
    )
    parser.add_argument(
        "--min-score", type=float, default=MIN_SCORE,
        help=f"Minimum cosine similarity for a pair to survive (default: {MIN_SCORE}). "
             f"Higher → fewer matches.",
    )
    parser.add_argument(
        "--k-recip", type=int, default=K_RECIP,
        help=f"Top-K sources per amendment in reverse retrieval (default: {K_RECIP}). "
             f"Lower → fewer matches (stricter reciprocity).",
    )
    args = parser.parse_args()

    run_amendment_pipeline(
        proc_id=args.procedure,
        output_dir=Path(args.output_dir),
        rate_limit_sleep=args.rate_limit_sleep,
        k=args.k,
        min_score=args.min_score,
        k_recip=args.k_recip,
    )


if __name__ == "__main__":
    main()
