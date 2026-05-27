"""
Embedding Model Benchmark — EU Legislative Retrieval
=====================================================

Runs both backward (feedback → article) and forward (feedback → amendment)
retrieval benchmarks on a single EU procedure using citation-derived gold
standards.

Usage:
    .venv/bin/python evaluation/embedding_benchmark.py [--procedure 2021/0106(COD)]
                                                       [--task backward|forward|both]
                                                       [--models MODEL1,MODEL2,...]
                                                       [--device mps|cpu|cuda]

Models are specified by short name (see MODELS dict below).
Default runs all models on both tasks.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Model registry ────────────────────────────────────────────────────────────
# Short name → config.  Add new models here.
MODELS = {
    # ── Already benchmarked (baselines) ──────────────────────────────────
    'mpnet-base-v2': {
        'hf_id':          'sentence-transformers/all-mpnet-base-v2',
        'query_prefix':   '',
        'passage_prefix': '',
    },
    'bge-large-en-v1.5': {
        'hf_id':          'BAAI/bge-large-en-v1.5',
        'query_prefix':   'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
    },
    'e5-large-v2': {
        'hf_id':          'intfloat/e5-large-v2',
        'query_prefix':   'query: ',
        'passage_prefix': 'passage: ',
    },
    'multilingual-e5-large': {
        'hf_id':          'intfloat/multilingual-e5-large',
        'query_prefix':   'query: ',
        'passage_prefix': 'passage: ',
    },
    'multilingual-e5-large-instruct': {
        'hf_id':          'intfloat/multilingual-e5-large-instruct',
        'query_prefix':   (
            'Instruct: Given a stakeholder lobbying position on EU legislation, '
            'retrieve the most relevant legislative article or recital\nQuery: '
        ),
        'passage_prefix': '',
    },
    'multilingual-e5-large-instruct-generic': {
        'hf_id':          'intfloat/multilingual-e5-large-instruct',
        'query_prefix':   'Instruct: Retrieve the most semantically similar document\nQuery: ',
        'passage_prefix': '',
    },
    'multilingual-e5-large-instruct-consult': {
        'hf_id':          'intfloat/multilingual-e5-large-instruct',
        'query_prefix':   (
            'Instruct: Given a public consultation response about EU policy, '
            'retrieve the legislative provision it discusses\nQuery: '
        ),
        'passage_prefix': '',
    },
    'multilingual-e5-large-instruct-bare': {
        'hf_id':          'intfloat/multilingual-e5-large-instruct',
        'query_prefix':   'query: ',
        'passage_prefix': 'passage: ',
    },
    'sbert-legal-xlm-roberta': {
        'hf_id':          'Stern5497/sbert-legal-xlm-roberta-base',
        'query_prefix':   '',
        'passage_prefix': '',
    },

    # ── New models to benchmark ──────────────────────────────────────────

    # BGE-M3: state-of-the-art multilingual retrieval (568M params).
    # Handles 100+ languages, dense+sparse+colbert multi-vector.
    # We use dense mode only for apples-to-apples comparison.
    'bge-m3': {
        'hf_id':          'BAAI/bge-m3',
        'query_prefix':   '',
        'passage_prefix': '',
    },

    # GTE-large-en v1.5: Alibaba, strong English retrieval (~434M params).
    'gte-large-en-v1.5': {
        'hf_id':          'Alibaba-NLP/gte-large-en-v1.5',
        'query_prefix':   '',
        'passage_prefix': '',
        'trust_remote_code': True,
    },

    # GTE-Qwen2-1.5B-instruct: larger Alibaba model, instruction-tuned.
    # 1.5B params — fits in 48GB M4 Pro but will be slower.
    'gte-qwen2-1.5b': {
        'hf_id':          'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'query_prefix':   (
            'Instruct: Given a stakeholder lobbying position on EU legislation, '
            'retrieve the most relevant legislative article or recital\nQuery: '
        ),
        'passage_prefix': '',
        'trust_remote_code': True,
    },

    # Nomic-embed-text v1.5: 137M params, matryoshka embeddings,
    # surprisingly competitive for its size.
    'nomic-embed-v1.5': {
        'hf_id':          'nomic-ai/nomic-embed-text-v1.5',
        'query_prefix':   'search_query: ',
        'passage_prefix': 'search_document: ',
        'trust_remote_code': True,
    },

    # mxbai-embed-large-v1: mixedbread.ai, top MTEB performer (~335M params).
    'mxbai-embed-large': {
        'hf_id':          'mixedbread-ai/mxbai-embed-large-v1',
        'query_prefix':   'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
    },

    # Snowflake Arctic Embed L v2.0: top MTEB, English-focused (~335M params).
    'arctic-embed-l-v2': {
        'hf_id':          'Snowflake/snowflake-arctic-embed-l-v2.0',
        'query_prefix':   'Represent this sentence for searching relevant passages: ',
        'passage_prefix': '',
        'trust_remote_code': True,
    },

    # E5-mistral-7b-instruct: 7B param model, very high quality but slow.
    # Only include if you have patience — ~15 min per embedding run on M4 Pro.
    'e5-mistral-7b': {
        'hf_id':          'intfloat/e5-mistral-7b-instruct',
        'query_prefix':   'Instruct: Given a stakeholder lobbying position on EU legislation, retrieve the most relevant legislative article or recital\nQuery: ',
        'passage_prefix': '',
        'batch_size':     4,
    },
    'e5-mistral-7b-generic': {
        'hf_id':          'intfloat/e5-mistral-7b-instruct',
        'query_prefix':   'Instruct: Retrieve the most semantically similar document\nQuery: ',
        'passage_prefix': '',
        'batch_size':     4,
    },
    'e5-mistral-7b-consult': {
        'hf_id':          'intfloat/e5-mistral-7b-instruct',
        'query_prefix':   (
            'Instruct: Given a public consultation response about EU policy, '
            'retrieve the legislative provision it discusses\nQuery: '
        ),
        'passage_prefix': '',
        'batch_size':     4,
    },
    'e5-mistral-7b-bare': {
        'hf_id':          'intfloat/e5-mistral-7b-instruct',
        'query_prefix':   'query: ',
        'passage_prefix': 'passage: ',
        'batch_size':     4,
    },

    # Jina-embeddings-v3: multilingual, matryoshka, task-specific (~570M).
    'jina-v3': {
        'hf_id':          'jinaai/jina-embeddings-v3',
        'query_prefix':   '',
        'passage_prefix': '',
        'trust_remote_code': True,
    },
}

# Default set: original models + best new candidates (skip 7B by default)
DEFAULT_MODELS = [
    'multilingual-e5-large',
    'multilingual-e5-large-instruct',
    'e5-large-v2',
    'bge-large-en-v1.5',
    'bge-m3',
    'gte-large-en-v1.5',
    'gte-qwen2-1.5b',
    'nomic-embed-v1.5',
    'mxbai-embed-large',
    'arctic-embed-l-v2',
    'jina-v3',
]

K_VALUES             = [1, 3, 5, 10]
MIN_CLEANED_LEN      = 80
PARAPHRASE_THRESHOLD = 0.30
MIN_GOLD_PAIRS       = 20


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_supabase():
    return create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_SERVICE_ROLE_KEY'],
    )


def paginate(supabase, table, select, filter_fn=None, page_size=1000):
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


# ── Citation helpers ──────────────────────────────────────────────────────────

_NUM  = r'\(?\d+[a-z]?\)?'
_SEP  = r'(?:\s*(?:,|and|to|-)\s*' + _NUM + r')*'
_ART  = r'(?:articles?|art\.?)'
_REC  = r'(?:recitals?)'

STRIP_RE = re.compile(
    _ART + r'\s*' + _NUM + r'(?:,\s*paragraph\s*\d+[a-z]?)?' + _SEP
    + r'|' +
    _REC + r'\s*' + _NUM + _SEP,
    re.IGNORECASE,
)

_CAP_ART = re.compile(_ART + r'\s*\(?(\d+[a-z]?)\)?', re.IGNORECASE)
_CAP_REC = re.compile(_REC + r'\s*\(?(\d+)\)?',       re.IGNORECASE)

_AMT_ART = re.compile(r'article\s*\(?(\d+[a-z]?)\)?', re.IGNORECASE)
_AMT_REC = re.compile(r'recital\s*\(?(\d+[a-z]?)\)?', re.IGNORECASE)


def clean_citations(text):
    out = STRIP_RE.sub('', text)
    out = re.sub(r'[ \t]{2,}', ' ', out)
    out = re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()


def citation_window(text, match, window=900):
    start = max(0, match.start() - window)
    end   = min(len(text), match.end() + window)
    return text[start:end]


def _lcs_len(a, b):
    a, b = a[:150], b[:150]
    m, n = len(a), len(b)
    if not m or not n:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            curr[j + 1] = prev[j] + 1 if a[i] == b[j] else max(prev[j + 1], curr[j])
        prev = curr
    return prev[n]


def rouge_l_f1(hyp, ref):
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    lcs  = _lcs_len(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def parse_amd_target(target_element):
    if not target_element:
        return None, None
    m = _AMT_ART.search(target_element)
    if m:
        return 'article', m.group(1).lower().lstrip('0') or '0'
    m = _AMT_REC.search(target_element)
    if m:
        return 'recital', m.group(1).lower().lstrip('0') or '0'
    return None, None


# ── Embedding helper ──────────────────────────────────────────────────────────

def embed_texts(model, texts, prefix='', batch_size=32, device=None):
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)
    prefixed = [prefix + t for t in texts] if prefix else texts
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if device:
        kwargs['device'] = device
    return model.encode(prefixed, **kwargs)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_backward_data(supabase, procedure_id):
    """Load articles + post-proposal feedback; build gold pairs for backward benchmark."""
    proc = (
        supabase.table('procedures')
        .select('title, proposal_date')
        .eq('id', procedure_id)
        .single()
        .execute()
        .data
    )
    proposal_date = proc['proposal_date']
    print(f'Procedure:     {procedure_id}')
    print(f'Title:         {proc["title"]}')
    print(f'Proposal date: {proposal_date}')

    art_rows = paginate(
        supabase,
        'procedure_articles',
        'element_type, element_number, title, content, sort_order',
        lambda q: (
            q.eq('procedure_id', procedure_id)
             .eq('document_version', 'proposal')
             .order('sort_order')
        ),
    )
    if not art_rows:
        raise ValueError(f'No proposal articles for {procedure_id}')

    article_lookup = {
        (a['element_type'], a['element_number'].lstrip('0') or '0'): a
        for a in art_rows
    }

    fb_rows = paginate(
        supabase,
        'hys_feedback_chunks',
        'id, feedback_id, chunk_index, chunk_text, organisation_name, transparency_reg_id, date_feedback',
        lambda q: (
            q.eq('procedure_id', procedure_id)
             .gte('date_feedback', proposal_date)
        ),
    )
    print(f'Proposal elements: {len(art_rows)}')
    print(f'Post-proposal feedback chunks: {len(fb_rows)}')

    chunk_lookup = {
        (r['feedback_id'], r['chunk_index']): r['chunk_text']
        for r in fb_rows
    }

    gold_pairs = []
    skipped    = defaultdict(int)
    seen_keys  = set()

    for row in fb_rows:
        current = (row.get('chunk_text') or '').strip()
        if not current:
            skipped['empty_chunk'] += 1
            continue

        nxt      = chunk_lookup.get((row['feedback_id'], row['chunk_index'] + 1), '')
        combined = (current + ' ' + nxt).strip() if nxt else current

        matches = (
            [('article', m) for m in _CAP_ART.finditer(combined)] +
            [('recital', m) for m in _CAP_REC.finditer(combined)]
        )
        if not matches:
            skipped['no_citation'] += 1
            continue

        for etype, match in matches:
            enum = match.group(1).lower().lstrip('0') or '0'
            if (etype, enum) not in article_lookup:
                skipped['article_not_found'] += 1
                continue

            key = (row['id'], etype, enum)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            window_text = citation_window(combined, match, window=900)
            cleaned     = clean_citations(window_text)

            if len(cleaned) < MIN_CLEANED_LEN:
                skipped['too_short_after_clean'] += 1
                continue

            art = article_lookup[(etype, enum)]
            gold_pairs.append({
                'chunk_id':       row['id'],
                'org':            row.get('organisation_name'),
                'feedback_raw':   window_text,
                'feedback_clean': cleaned,
                'element_type':   etype,
                'element_number': enum,
                'art_content':    art['content'],
                'art_title':      art.get('title'),
            })

    # Density filter
    cites_per_chunk = Counter(p['chunk_id'] for p in gold_pairs)
    noisy_chunks    = {cid for cid, n in cites_per_chunk.items() if n >= 5}
    gold_pairs      = [p for p in gold_pairs if p['chunk_id'] not in noisy_chunks]

    # ROUGE-L
    gold_df = pd.DataFrame(gold_pairs)
    gold_df['rouge_l']       = [rouge_l_f1(p['feedback_clean'], p['art_content']) for p in gold_pairs]
    gold_df['is_paraphrase'] = gold_df['rouge_l'] >= PARAPHRASE_THRESHOLD

    print(f'Gold pairs: {len(gold_pairs)} ({gold_df["is_paraphrase"].sum()} near-paraphrases)')
    print(f'Skipped: {dict(skipped)}')

    if len(gold_pairs) < MIN_GOLD_PAIRS:
        raise ValueError(f'Only {len(gold_pairs)} gold pairs (need >= {MIN_GOLD_PAIRS})')

    # Build corpus arrays
    def art_header(a):
        h = f"{a['element_type']} {a['element_number']}"
        return h + (f": {a['title']}" if a.get('title') else '')

    art_raw   = [art_header(a) + '\n' + a['content'] for a in art_rows]
    art_clean = [a['content'] for a in art_rows]

    article_keys   = [(a['element_type'], a['element_number'].lstrip('0') or '0') for a in art_rows]
    art_key_to_idx = {k: i for i, k in enumerate(article_keys)}

    feedback_clean = [p['feedback_clean'] for p in gold_pairs]
    gold_art_idxs  = [art_key_to_idx[(p['element_type'], p['element_number'])] for p in gold_pairs]

    return {
        'art_rows':       art_rows,
        'art_raw':        art_raw,
        'art_clean':      art_clean,
        'feedback_clean': feedback_clean,
        'gold_art_idxs':  gold_art_idxs,
        'gold_df':        gold_df,
        'gold_pairs':     gold_pairs,
    }


def load_forward_data(supabase, procedure_id):
    """Load amendments + post-proposal feedback; build gold pairs for forward benchmark."""
    proc = (
        supabase.table('procedures')
        .select('title, proposal_date')
        .eq('id', procedure_id)
        .single()
        .execute()
        .data
    )
    proposal_date = proc['proposal_date']

    amd_rows = paginate(
        supabase,
        'procedure_amendments',
        'id, amendment_number, target_element, target_type, original_text, amended_text, justification, committee',
        lambda q: q.eq('procedure_id', procedure_id),
    )
    amd_rows = [
        a for a in amd_rows
        if a.get('target_type') in ('article', 'recital')
        and len((a.get('amended_text') or a.get('original_text') or '').strip()) >= 80
    ]
    print(f'Amendments with article/recital target: {len(amd_rows)}')

    fb_rows = paginate(
        supabase,
        'hys_feedback_chunks',
        'id, feedback_id, chunk_index, chunk_text, organisation_name, transparency_reg_id, date_feedback',
        lambda q: (
            q.eq('procedure_id', procedure_id)
             .gte('date_feedback', proposal_date)
        ),
    )
    print(f'Post-proposal feedback chunks: {len(fb_rows)}')

    amd_key_to_idxs = defaultdict(list)
    for idx, a in enumerate(amd_rows):
        etype, enum = parse_amd_target(a.get('target_element', ''))
        if etype and enum:
            amd_key_to_idxs[(etype, enum)].append(idx)

    chunk_lookup = {
        (r['feedback_id'], r['chunk_index']): r['chunk_text']
        for r in fb_rows
    }

    gold_pairs = []
    skipped    = defaultdict(int)
    seen_keys  = set()

    for row in fb_rows:
        current = (row.get('chunk_text') or '').strip()
        if not current:
            skipped['empty_chunk'] += 1
            continue

        nxt      = chunk_lookup.get((row['feedback_id'], row['chunk_index'] + 1), '')
        combined = (current + ' ' + nxt).strip() if nxt else current

        matches = (
            [('article', m) for m in _CAP_ART.finditer(combined)] +
            [('recital', m) for m in _CAP_REC.finditer(combined)]
        )
        if not matches:
            skipped['no_citation'] += 1
            continue

        cited = set()
        for etype, match in matches:
            enum = match.group(1).lower().lstrip('0') or '0'
            cited.add((etype, enum))

        if len(cited) > 1:
            skipped['multi_citation'] += 1
            continue

        etype, enum = next(iter(cited))
        gold_idxs = amd_key_to_idxs.get((etype, enum), [])
        if not gold_idxs:
            skipped['no_amendment_for_article'] += 1
            continue

        key = (row['id'], etype, enum)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        first_match = matches[0][1]
        window_text = citation_window(combined, first_match, window=900)
        cleaned     = clean_citations(window_text)

        if len(cleaned) < MIN_CLEANED_LEN:
            skipped['too_short_after_clean'] += 1
            continue

        gold_pairs.append({
            'chunk_id':       row['id'],
            'org':            row.get('organisation_name'),
            'feedback_raw':   window_text,
            'feedback_clean': cleaned,
            'element_type':   etype,
            'element_number': enum,
            'gold_amd_idxs':  gold_idxs,
            'n_gold':         len(gold_idxs),
        })

    # Density filter
    cites_per_chunk = Counter(p['chunk_id'] for p in gold_pairs)
    noisy_chunks    = {cid for cid, n in cites_per_chunk.items() if n >= 5}
    gold_pairs      = [p for p in gold_pairs if p['chunk_id'] not in noisy_chunks]

    def max_rouge(fb_clean, gold_idxs):
        best = 0.0
        for idx in gold_idxs:
            amd_text = (amd_rows[idx].get('amended_text') or amd_rows[idx].get('original_text') or '')
            best = max(best, rouge_l_f1(fb_clean, amd_text))
        return best

    gold_df = pd.DataFrame(gold_pairs)
    gold_df['rouge_l']       = [max_rouge(p['feedback_clean'], p['gold_amd_idxs']) for p in gold_pairs]
    gold_df['is_paraphrase'] = gold_df['rouge_l'] >= PARAPHRASE_THRESHOLD

    print(f'Gold pairs: {len(gold_pairs)} ({gold_df["is_paraphrase"].sum()} near-paraphrases)')
    print(f'Skipped: {dict(skipped)}')

    if len(gold_pairs) < MIN_GOLD_PAIRS:
        raise ValueError(f'Only {len(gold_pairs)} gold pairs (need >= {MIN_GOLD_PAIRS})')

    amd_text_only = [
        (a.get('amended_text') or a.get('original_text') or '').strip()
        for a in amd_rows
    ]
    amd_with_header = [
        (a.get('target_element') or '') + '\n' + (a.get('amended_text') or a.get('original_text') or '').strip()
        for a in amd_rows
    ]

    feedback_clean = [p['feedback_clean'] for p in gold_pairs]
    gold_amd_idxs  = [p['gold_amd_idxs'] for p in gold_pairs]

    return {
        'amd_rows':        amd_rows,
        'amd_text_only':   amd_text_only,
        'amd_with_header': amd_with_header,
        'feedback_clean':  feedback_clean,
        'gold_amd_idxs':   gold_amd_idxs,
        'gold_df':         gold_df,
        'gold_pairs':      gold_pairs,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_backward(data, model_configs, device=None):
    """Run backward benchmark: feedback → article retrieval."""
    from sentence_transformers import SentenceTransformer

    VARIANTS = [
        ('actual', 'fb_clean', 'art_raw',   'Clean query + raw article'),
        ('strict', 'fb_clean', 'art_clean', 'Both stripped — semantic lower bound'),
    ]

    all_results  = []
    per_query_db = {}

    for name, cfg in model_configs.items():
        hf_id = cfg['hf_id']
        qpfx  = cfg['query_prefix']
        ppfx  = cfg['passage_prefix']
        bs    = cfg.get('batch_size', 32)
        trust = cfg.get('trust_remote_code', False)

        print(f'\n{"="*62}')
        print(f'  BACKWARD: {name}  ({hf_id})')
        print(f'{"="*62}')

        try:
            t0 = time.time()
            mdl = SentenceTransformer(hf_id, trust_remote_code=trust)

            try:
                emb_art_raw   = embed_texts(mdl, data['art_raw'],        prefix=ppfx, batch_size=bs, device=device)
                emb_art_clean = embed_texts(mdl, data['art_clean'],      prefix=ppfx, batch_size=bs, device=device)
                emb_fb_clean  = embed_texts(mdl, data['feedback_clean'], prefix=qpfx, batch_size=bs, device=device)
            except Exception as e:
                if device == 'mps':
                    print(f'  MPS failed ({e}), retrying on CPU...')
                    emb_art_raw   = embed_texts(mdl, data['art_raw'],        prefix=ppfx, batch_size=bs, device='cpu')
                    emb_art_clean = embed_texts(mdl, data['art_clean'],      prefix=ppfx, batch_size=bs, device='cpu')
                    emb_fb_clean  = embed_texts(mdl, data['feedback_clean'], prefix=qpfx, batch_size=bs, device='cpu')
                else:
                    raise

            elapsed = time.time() - t0
            print(f'  Embedding time: {elapsed:.1f}s')

            del mdl; gc.collect()

            pool = {
                'art_raw':   emb_art_raw,
                'art_clean': emb_art_clean,
                'fb_clean':  emb_fb_clean,
            }

            for v_name, q_key, a_key, desc in VARIANTS:
                sim = pool[q_key] @ pool[a_key].T
                per_q, rr = [], []

                for qi, true_idx in enumerate(data['gold_art_idxs']):
                    scores = sim[qi]
                    ranked = np.argsort(scores)[::-1]
                    rank   = int(np.where(ranked == true_idx)[0][0]) + 1
                    rr.append(1.0 / rank)
                    per_q.append({
                        'rank':          rank,
                        'score_correct': float(scores[true_idx]),
                        'score_top1':    float(scores[ranked[0]]),
                        'rouge_l':       data['gold_df'].iloc[qi]['rouge_l'],
                        'is_paraphrase': bool(data['gold_df'].iloc[qi]['is_paraphrase']),
                    })

                per_query_db[(name, v_name)] = per_q

                row = {'model': name, 'variant': v_name, 'mrr': float(np.mean(rr)),
                       'embed_time_s': elapsed}
                for k in K_VALUES:
                    row[f'R@{k}'] = sum(1 for p in per_q if p['rank'] <= k) / len(per_q)
                all_results.append(row)

                print(f'  [{v_name:8s}] MRR={row["mrr"]:.3f}  ' +
                      '  '.join(f'R@{k}={row[f"R@{k}"]:.1%}' for k in K_VALUES))

            sem = [p for p in per_query_db[(name, 'actual')] if not p['is_paraphrase']]
            if sem:
                sem_mrr = float(np.mean([1/p['rank'] for p in sem]))
                sem_r5  = sum(1 for p in sem if p['rank'] <= 5) / len(sem)
                print(f'  [sem-only] MRR={sem_mrr:.3f}  R@5={sem_r5:.1%}  (n={len(sem)})')

        except Exception as e:
            print(f'  FAILED: {e}')
            continue

    return pd.DataFrame(all_results), per_query_db


def evaluate_forward(data, model_configs, device=None):
    """Run forward benchmark: feedback → amendment retrieval."""
    from sentence_transformers import SentenceTransformer

    VARIANTS = [
        ('pipeline',    'fb_clean', 'amd_text',   'Text only — real pipeline proxy'),
        ('with_header', 'fb_clean', 'amd_header', 'target_element header — upper bound'),
    ]

    all_results  = []
    per_query_db = {}

    for name, cfg in model_configs.items():
        hf_id = cfg['hf_id']
        qpfx  = cfg['query_prefix']
        ppfx  = cfg['passage_prefix']
        bs    = cfg.get('batch_size', 32)
        trust = cfg.get('trust_remote_code', False)

        print(f'\n{"="*62}')
        print(f'  FORWARD: {name}  ({hf_id})')
        print(f'{"="*62}')

        try:
            t0 = time.time()
            mdl = SentenceTransformer(hf_id, trust_remote_code=trust)

            try:
                emb_amd_text   = embed_texts(mdl, data['amd_text_only'],   prefix=ppfx, batch_size=bs, device=device)
                emb_amd_header = embed_texts(mdl, data['amd_with_header'], prefix=ppfx, batch_size=bs, device=device)
                emb_fb_clean   = embed_texts(mdl, data['feedback_clean'],  prefix=qpfx, batch_size=bs, device=device)
            except Exception as e:
                if device == 'mps':
                    print(f'  MPS failed ({e}), retrying on CPU...')
                    emb_amd_text   = embed_texts(mdl, data['amd_text_only'],   prefix=ppfx, batch_size=bs, device='cpu')
                    emb_amd_header = embed_texts(mdl, data['amd_with_header'], prefix=ppfx, batch_size=bs, device='cpu')
                    emb_fb_clean   = embed_texts(mdl, data['feedback_clean'],  prefix=qpfx, batch_size=bs, device='cpu')
                else:
                    raise

            elapsed = time.time() - t0
            print(f'  Embedding time: {elapsed:.1f}s')

            del mdl; gc.collect()

            pool = {
                'amd_text':   emb_amd_text,
                'amd_header': emb_amd_header,
                'fb_clean':   emb_fb_clean,
            }

            for v_name, q_key, a_key, desc in VARIANTS:
                sim = pool[q_key] @ pool[a_key].T
                per_q, rr = [], []

                for qi, gold_set in enumerate(data['gold_amd_idxs']):
                    scores = sim[qi]
                    ranked = np.argsort(scores)[::-1]

                    first_hit_rank = None
                    for pos, amd_idx in enumerate(ranked):
                        if amd_idx in gold_set:
                            first_hit_rank = pos + 1
                            break
                    rank = first_hit_rank if first_hit_rank is not None else len(ranked) + 1
                    rr.append(1.0 / rank)

                    best_gold_score = float(max(scores[i] for i in gold_set))
                    per_q.append({
                        'rank':            rank,
                        'n_gold':          len(gold_set),
                        'score_best_gold': best_gold_score,
                        'score_top1':      float(scores[ranked[0]]),
                        'rouge_l':         data['gold_df'].iloc[qi]['rouge_l'],
                        'is_paraphrase':   bool(data['gold_df'].iloc[qi]['is_paraphrase']),
                    })

                per_query_db[(name, v_name)] = per_q

                row = {'model': name, 'variant': v_name, 'mrr': float(np.mean(rr)),
                       'embed_time_s': elapsed}
                for k in K_VALUES:
                    row[f'R@{k}'] = sum(
                        1 for qi, gold_set in enumerate(data['gold_amd_idxs'])
                        if any(idx in gold_set for idx in np.argsort(sim[qi])[::-1][:k])
                    ) / len(data['gold_amd_idxs'])
                all_results.append(row)

                print(f'  [{v_name:12s}] MRR={row["mrr"]:.3f}  ' +
                      '  '.join(f'R@{k}={row[f"R@{k}"]:.1%}' for k in K_VALUES))

        except Exception as e:
            print(f'  FAILED: {e}')
            continue

    return pd.DataFrame(all_results), per_query_db


# ── Output ────────────────────────────────────────────────────────────────────

def print_leaderboard(df, variant_col, main_variant):
    if df.empty:
        print('\n  No results to display (all models failed).')
        return
    sub = df[df['variant'] == main_variant].sort_values('mrr', ascending=False)
    print(f'\n{"="*70}')
    print(f'  LEADERBOARD — {main_variant} variant')
    print(f'{"="*70}')
    print(f'  {"Rank":<5} {"Model":<32} {"MRR":>7} {"R@1":>7} {"R@3":>7} {"R@5":>7} {"R@10":>7} {"Time":>7}')
    print(f'  {"─"*5} {"─"*32} {"─"*7} {"─"*7} {"─"*7} {"─"*7} {"─"*7} {"─"*7}')
    for rank, (_, row) in enumerate(sub.iterrows(), 1):
        print(f'  {rank:<5} {row["model"]:<32} {row["mrr"]:>6.1%} {row["R@1"]:>6.1%} '
              f'{row["R@3"]:>6.1%} {row["R@5"]:>6.1%} {row["R@10"]:>6.1%} {row["embed_time_s"]:>5.0f}s')


def save_results(results_df, per_query_db, task, procedure_id, out_dir):
    proc_slug = procedure_id.replace('/', '_').replace('(', '').replace(')', '')
    suffix    = f'{task}_{proc_slug}'

    csv_path  = out_dir / f'embedding_benchmark_{suffix}.csv'
    json_path = out_dir / f'embedding_benchmark_{suffix}.json'

    results_df.to_csv(csv_path, index=False)

    export = {
        'procedure_id': procedure_id,
        'task':         task,
        'aggregate':    results_df.to_dict(orient='records'),
        'per_query':    {
            f'{m}__{v}': pq
            for (m, v), pq in per_query_db.items()
        },
    }
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(export, fh, indent=2)

    print(f'\nSaved: {csv_path}')
    print(f'Saved: {json_path}')
    return csv_path, json_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Embedding model benchmark for EU legislative retrieval')
    parser.add_argument('--procedure', default='2021/0106(COD)',
                        help='Procedure ID to benchmark (default: AI Act)')
    parser.add_argument('--task', default='both', choices=['backward', 'forward', 'both'],
                        help='Which benchmark to run')
    parser.add_argument('--models', default=None,
                        help='Comma-separated model short names (default: all non-7B models)')
    parser.add_argument('--device', default=None, choices=['mps', 'cpu', 'cuda'],
                        help='Device for encoding (default: auto-detect)')
    parser.add_argument('--list-models', action='store_true',
                        help='Print available models and exit')
    args = parser.parse_args()

    if args.list_models:
        print('Available models:')
        for name, cfg in MODELS.items():
            marker = ' *' if name in DEFAULT_MODELS else ''
            print(f'  {name:<35} {cfg["hf_id"]}{marker}')
        print('\n* = included in default run')
        return

    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        for m in model_names:
            if m not in MODELS:
                print(f'Unknown model: {m}')
                print(f'Available: {", ".join(MODELS.keys())}')
                sys.exit(1)
    else:
        model_names = DEFAULT_MODELS

    model_configs = {name: MODELS[name] for name in model_names}

    device = args.device
    if device is None:
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            print('Auto-detected MPS (Apple Silicon)')
        elif torch.cuda.is_available():
            device = 'cuda'
            print('Auto-detected CUDA')
        else:
            device = 'cpu'
            print('Using CPU')

    print(f'\nRunning {args.task} benchmark on {args.procedure}')
    print(f'Models: {", ".join(model_names)}')
    print(f'Device: {device}\n')

    supabase = get_supabase()
    out_dir  = Path('evaluation/results')
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ('backward', 'both'):
        print('\n' + '▓' * 70)
        print('  BACKWARD BENCHMARK: Feedback → Article')
        print('▓' * 70)
        bw_data = load_backward_data(supabase, args.procedure)
        bw_results, bw_pq = evaluate_backward(bw_data, model_configs, device=device)
        print_leaderboard(bw_results, 'variant', 'actual')
        save_results(bw_results, bw_pq, 'backward', args.procedure, out_dir)

    if args.task in ('forward', 'both'):
        print('\n' + '▓' * 70)
        print('  FORWARD BENCHMARK: Feedback → Amendment')
        print('▓' * 70)
        fw_data = load_forward_data(supabase, args.procedure)
        fw_results, fw_pq = evaluate_forward(fw_data, model_configs, device=device)
        print_leaderboard(fw_results, 'variant', 'pipeline')
        save_results(fw_results, fw_pq, 'forward', args.procedure, out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
