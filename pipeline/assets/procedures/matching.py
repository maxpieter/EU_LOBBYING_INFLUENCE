"""Procedure matching: link meetings to legislative procedures.

5-step cascade (deterministic first, AI last):
1. Exact ID match (lobbying meetings with related_procedure)
2. Alias exact match (lowercased meeting text → procedure_aliases)
3. Trigram similarity (rapidfuzz against procedure titles)
4. AI classification (Claude Haiku, batches)
5. Unmatchable (no candidates → match_status = 'no_match')

Temporal filtering: meetings only match procedures active at the meeting date.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Temporal filtering
# ---------------------------------------------------------------------------

def _is_temporally_valid(
    meeting_date: Optional[str],
    procedure: dict,
    buffer_months: int = 6,
) -> bool:
    """Check if a meeting date falls within a procedure's active window.

    Window: proposal_date → decision_date + buffer (or last_activity + buffer).
    If no dates available, skip the check (catalog-only rows).
    """
    if not meeting_date:
        return True

    try:
        m_date = date.fromisoformat(meeting_date)
    except (ValueError, TypeError):
        return True

    proposal = procedure.get("proposal_date")
    decision = procedure.get("decision_date")
    last_activity = procedure.get("last_activity_date")

    # No dates at all → skip temporal check
    if not proposal and not decision and not last_activity:
        return True

    buffer = timedelta(days=buffer_months * 30)

    # Start: proposal date (or 2 years before first known date as fallback)
    if proposal:
        try:
            start = date.fromisoformat(str(proposal))
        except (ValueError, TypeError):
            start = date(2000, 1, 1)
    else:
        start = date(2000, 1, 1)

    # End: decision + buffer, or last_activity + buffer
    end = date(2099, 12, 31)  # default: open-ended
    if decision:
        try:
            end = date.fromisoformat(str(decision)) + buffer
        except (ValueError, TypeError):
            pass
    elif last_activity:
        try:
            end = date.fromisoformat(str(last_activity)) + buffer
        except (ValueError, TypeError):
            pass

    return start <= m_date <= end


# ---------------------------------------------------------------------------
# ProcedureMatcher
# ---------------------------------------------------------------------------

# Generic meeting titles that cannot meaningfully match a procedure
_GENERIC_TITLES = {
    "exchange of views", "general exchange of views", "meeting",
    "introductory meeting", "introduction", "introductory",
    "various", "austausch", "möte", "rencontre", "incontro",
    "general exchange of view", "bilateral meeting", "courtesy visit",
    "courtesy call", "phone call", "video call", "working lunch",
    "working dinner", "breakfast meeting", "lunch meeting",
    "dinner meeting", "informal meeting", "formal meeting",
}


class ProcedureMatcher:
    """Match meetings to procedures using a 5-step cascade.

    Initialize with procedures and aliases from DB, then call match() on
    individual meetings or match_batch() for bulk processing.
    """

    def __init__(
        self,
        procedures: list[dict],
        aliases: list[dict],
    ):
        # Index procedures by ID
        self._by_id: dict[str, dict] = {p["id"]: p for p in procedures}

        # Build alias → [procedure_ids] index (lowercased)
        self._alias_to_procs: dict[str, list[str]] = {}
        for a in aliases:
            key = a["alias"].strip().lower()
            proc_id = a["procedure_id"]
            if proc_id in self._by_id:
                self._alias_to_procs.setdefault(key, []).append(proc_id)

        # Also index procedure titles as aliases
        for p in procedures:
            title_lower = p["title"].strip().lower()
            self._alias_to_procs.setdefault(title_lower, []).append(p["id"])

        # Pre-build alias substring index for fast lookup
        # Strategy: build a dict of first-word → [(alias, proc_ids)] for quick filtering
        self._alias_first_word: dict[str, list[tuple[str, list[str]]]] = {}
        for alias, proc_ids in self._alias_to_procs.items():
            if len(alias) >= 4:
                first_word = alias.split()[0] if " " in alias else alias
                self._alias_first_word.setdefault(first_word, []).append((alias, proc_ids))
        # Also keep single-word aliases indexed by themselves
        self._single_word_aliases: dict[str, list[str]] = {}
        for alias, proc_ids in self._alias_to_procs.items():
            if len(alias) >= 4 and " " not in alias:
                self._single_word_aliases[alias] = proc_ids

        # Build rapidfuzz lookup
        self._proc_titles = {p["id"]: p["title"] for p in procedures}
        self._title_list = list(self._proc_titles.values())
        self._title_id_map: dict[str, str] = {}
        for pid, title in self._proc_titles.items():
            norm = title.strip().lower()
            self._title_id_map[norm] = pid

    def match(
        self,
        meeting_text: str,
        meeting_date: Optional[str] = None,
        related_procedure: Optional[str] = None,
        org_name: Optional[str] = None,
    ) -> list[dict]:
        """Run the 3-step cascade on a single meeting.

        Cascade (high → low reliability):
        1. exact_id:        related_procedure field matches procedures.id
        2. alias_exact:     full meeting text (after filler stripping) = known alias
        3. alias_substring: known alias appears within the text → send to AI

        No trigram/rapidfuzz — empirically ~10% true-positive rate, not worth it.

        Returns list of {procedure_id, match_method, confidence} dicts,
        or [{_needs_ai, candidates, matched_alias}] for AI confirmation.
        """
        results = []

        # Step 1: Exact ID match (lobbying meetings with related_procedure)
        if related_procedure:
            proc_id = related_procedure.strip()
            if proc_id in self._by_id:
                proc = self._by_id[proc_id]
                if _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": proc_id,
                        "match_method": "exact_id",
                        "confidence": 1.0,
                    })
                    return results

        if not meeting_text or not meeting_text.strip():
            return []

        text_lower = meeting_text.strip().lower()
        text_normalized = re.sub(r"\s+", " ", text_lower)

        # Skip generic / too-short meeting titles
        if text_normalized in _GENERIC_TITLES or len(text_normalized) < 3:
            return []

        # Step 2a: Alias exact match (full text = alias)
        # No word-count filter here — "digital euro", "NIS2" etc. are valid aliases
        if text_normalized in self._alias_to_procs:
            proc_ids = self._alias_to_procs[text_normalized]
            for pid in proc_ids:
                proc = self._by_id.get(pid)
                if proc and _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": pid,
                        "match_method": "alias_exact",
                        "confidence": 0.95,
                    })
            if results:
                return results

        # Strip common filler prefixes
        stripped = re.sub(
            r"^(?:general\s+)?(?:exchange\s+of\s+views?\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"meeting\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"discussion\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"briefing\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"presentation\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"update\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
            r"debate\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?)",
            "", text_normalized, flags=re.IGNORECASE,
        ).strip()

        # Step 2b: Alias exact on stripped text
        if stripped != text_normalized and stripped in self._alias_to_procs:
            proc_ids = self._alias_to_procs[stripped]
            for pid in proc_ids:
                proc = self._by_id.get(pid)
                if proc and _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": pid,
                        "match_method": "alias_exact",
                        "confidence": 0.90,
                    })
            if results:
                return results

        fuzzy_text = stripped if stripped and len(stripped) >= 5 else text_normalized

        # Minimum word count for substring matching — single-word titles
        # like "Energy" or "Trade" produce too many false alias hits.
        # alias_exact already handled short titles that ARE exact aliases.
        word_count = len(fuzzy_text.split())
        if word_count < 4:
            return []

        # Step 3: Alias substring → route to AI for confirmation
        text_words = set(fuzzy_text.split())
        best_match: tuple[str, list[str]] | None = None
        # Single-word aliases (CBAM, CSDDD, EPBD, etc.)
        for word in text_words:
            if word in self._single_word_aliases:
                if best_match is None or len(word) > len(best_match[0]):
                    best_match = (word, self._single_word_aliases[word])
        # Multi-word aliases
        for word in text_words:
            if word in self._alias_first_word:
                for alias, proc_ids in self._alias_first_word[word]:
                    if " " in alias and alias in fuzzy_text:
                        if best_match is None or len(alias) > len(best_match[0]):
                            best_match = (alias, proc_ids)

        if best_match:
            matched_alias, proc_ids = best_match
            # Build single-candidate list for AI confirmation
            candidates = []
            for pid in proc_ids:
                proc = self._by_id.get(pid)
                if proc and _is_temporally_valid(meeting_date, proc):
                    candidates.append({
                        "procedure_id": pid,
                        "title": proc.get("title", ""),
                        "score": 85.0,
                    })
            if candidates:
                return [{
                    "_needs_ai": True,
                    "candidates": candidates,
                    "matched_alias": matched_alias,
                    "org_name": org_name,
                }]

        return []

    def _fuzzy_match(self, text: str, top_n: int = 3) -> list[dict]:
        """Local rapidfuzz matching against procedure titles."""
        from rapidfuzz import fuzz, process

        results = process.extract(
            text, self._title_list, scorer=fuzz.WRatio, limit=top_n,
        )

        matches = []
        for title, score, _ in results:
            title_norm = title.strip().lower()
            proc_id = self._title_id_map.get(title_norm)
            if proc_id and score >= 30:
                matches.append({
                    "procedure_id": proc_id,
                    "title": title,
                    "score": score,
                })
        return matches

    @property
    def procedure_count(self) -> int:
        return len(self._by_id)

    @property
    def alias_count(self) -> int:
        return sum(len(v) for v in self._alias_to_procs.values())


# ---------------------------------------------------------------------------
# AI classification
# ---------------------------------------------------------------------------

def _build_match_prompt(batch: list[dict]) -> str:
    """Build prompt for AI procedure matching.

    Each item has a meeting text, an alias that was found in the text, and
    the candidate procedure(s) that alias maps to. The AI's job is to
    confirm whether the meeting SPECIFICALLY discusses that procedure.
    """
    items = []
    for i, item in enumerate(batch):
        cand_lines = []
        for j, c in enumerate(item["candidates"]):
            proc = item.get("_proc_details", {}).get(c["procedure_id"], {})
            date_range = ""
            if proc.get("proposal_date"):
                date_range = f", active: {proc.get('proposal_date', '?')} → {proc.get('decision_date', 'ongoing')}"
            cand_lines.append(
                f'    {chr(65+j)}. "{c["title"]}" [{c["procedure_id"]}]{date_range}'
            )

        meeting_text = item["text"][:500]
        date_str = f" (date: {item['date']})" if item.get("date") else ""
        org_str = f" (org: {item['org_name']})" if item.get("org_name") else ""
        alias_str = f" [alias matched: \"{item['matched_alias']}\"]" if item.get("matched_alias") else ""

        items.append(
            f'{i+1}. Meeting{date_str}{org_str}: "{meeting_text}"{alias_str}\n'
            f'   Candidates:\n' + "\n".join(cand_lines)
        )

    return (
        "For each meeting below, determine if the meeting SPECIFICALLY discusses "
        "the candidate legislative procedure(s). An alias keyword was found in the "
        "meeting text — your job is to confirm whether the meeting is genuinely about "
        "that procedure, or whether the keyword match is incidental.\n\n"
        "Rules:\n"
        "- Only match 'high' if the meeting clearly and specifically discusses this "
        "legislative file (not just the broad topic area).\n"
        "- A meeting titled just 'Energy policy' does NOT match 'Energy Efficiency "
        "Directive' — it's too broad.\n"
        "- A meeting about 'Fit for 55 package' mentioning multiple files should only "
        "match if the specific candidate procedure is clearly the focus.\n"
        "- If the meeting text is vague or generic (e.g. 'Industry policy', 'Trade'), "
        "return no_match.\n"
        "- Consider the organisation name (if given) as context — e.g. a steel "
        "association meeting about 'emissions' is more likely about ETS than a "
        "generic emissions topic.\n\n"
        + "\n".join(items)
        + "\n\nRespond with a JSON array, one entry per meeting:\n"
        '[{"match": "high"|"no_match", "chosen": "A"|"B"|"C"|"none", '
        '"reasoning": "one sentence"}, ...]\n'
        "Only use 'high'. Do not use 'medium' or 'low' — if unsure, return no_match.\n"
        f"Return exactly {len(batch)} entries in order."
    )


class AIQuotaError(RuntimeError):
    """Raised when AI classification can't proceed globally (credits, auth, org disabled).

    Signals the caller to stop the AI phase entirely. Affected meetings stay
    with match_status=NULL so a future run retries them.
    """


class AIBatchError(RuntimeError):
    """Raised when a single batch couldn't be classified (transient/unknown).

    Signals the caller to skip just this batch. Affected meetings stay with
    match_status=NULL so a future run retries them — they are NOT marked
    no_match, since we never actually heard back from the model.
    """


_QUOTA_SIGNALS = (
    "credit balance",
    "credit_balance",
    "insufficient",
    "quota",
    "billing",
    "authentication_error",
    "invalid x-api-key",
    "invalid api key",
    "organization_disabled",
    "organization_not_approved",
)


def ai_classify_batch(
    batch: list[dict],
    anthropic_client: Any,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Classify meeting→procedure matches via AI.

    Each batch item: {text, date, candidates, _proc_details}
    Returns [{match, chosen_index, reasoning}, ...]

    Raises:
        AIQuotaError: global fatal (credits/auth) — caller stops AI phase.
        AIBatchError: transient/unknown error for this batch — caller skips
            this batch only. Neither error should cause affected meetings to
            be written as no_match.
    """
    if not batch:
        return []

    prompt = _build_match_prompt(batch)

    for attempt in range(5):
        try:
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            msg = str(e).lower()
            if any(sig in msg for sig in _QUOTA_SIGNALS):
                raise AIQuotaError(str(e)) from e
            if "429" in msg or "rate_limit" in msg:
                time.sleep(2 ** attempt * 3)
                continue
            raise AIBatchError(f"API error: {e}") from e
    else:
        raise AIBatchError("Rate limit retries exhausted")

    raw = response.content[0].text
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise AIBatchError("Response contained no JSON array")
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        raise AIBatchError(f"JSON parse error: {e}") from e
    if not isinstance(parsed, list):
        raise AIBatchError(
            f"Response shape mismatch: expected list, got {type(parsed).__name__}"
        )
    # Haiku occasionally miscounts (e.g. 48 or 53 instead of 50).
    # Pad with no_match if too short, truncate if too long.
    if len(parsed) < len(batch):
        parsed.extend(
            [{"match": "no_match", "chosen": "none", "reasoning": "model_undercounted"}]
            * (len(batch) - len(parsed))
        )
    elif len(parsed) > len(batch):
        parsed = parsed[: len(batch)]

    valid = {"high", "medium", "low", "no_match"}
    out = []
    for entry in parsed:
        if isinstance(entry, dict) and entry.get("match") in valid:
            chosen_letter = str(entry.get("chosen", "none")).upper()
            chosen_index = (
                ord(chosen_letter) - 65
                if len(chosen_letter) == 1 and chosen_letter.isalpha() and chosen_letter in "ABC"
                else -1
            )
            out.append({
                "match": entry["match"],
                "chosen_index": chosen_index,
                "reasoning": str(entry.get("reasoning", ""))[:200],
            })
        else:
            out.append({"match": "no_match", "chosen_index": -1, "reasoning": "invalid_entry"})
    return out


# ---------------------------------------------------------------------------
# Bulk matching
# ---------------------------------------------------------------------------

def match_meetings(
    lobbying_meetings: list[dict],
    commission_meetings: list[dict],
    matcher: ProcedureMatcher,
    supabase_client: Any,
    logger: Optional[Any] = None,
    anthropic_client: Optional[Any] = None,
    ai_batch_size: int = 50,
    workers: int = 5,
    dry_run: bool = True,
) -> dict[str, int]:
    """Match all unprocessed meetings to procedures.

    Writes to meeting_procedure_links and updates match_status on source tables.
    Only processes meetings with match_status IS NULL.
    """
    _log = logger.info if logger else print

    stats = {
        "exact_id": 0, "alias_exact": 0,
        "ai_high": 0, "no_match": 0,
        "links_created": 0,
    }

    # Collect all meetings to process
    to_process: list[dict] = []  # {source, id, text, date, related_procedure}

    for m in lobbying_meetings:
        if m.get("match_status") is not None:
            continue
        to_process.append({
            "source": "lobbying",
            "id": m["id"],
            "text": m.get("title") or "",
            "date": m.get("meeting_date"),
            "related_procedure": m.get("related_procedure"),
            "org_name": m.get("org_name"),
        })

    for m in commission_meetings:
        if m.get("match_status") is not None:
            continue
        text = m.get("subject") or ""
        points = m.get("points_raised") or ""
        if points:
            text = f"{text}. {points[:500]}"
        to_process.append({
            "source": "commission",
            "id": m["id"],
            "text": text,
            "date": m.get("meeting_date"),
            "related_procedure": None,
            "org_name": None,
        })

    _log(f"Meetings to process: {len(to_process)} ({sum(1 for m in to_process if m['source'] == 'lobbying')} lobbying, {sum(1 for m in to_process if m['source'] == 'commission')} commission)")

    # Run steps 1-3 on all meetings
    need_ai: list[dict] = []
    link_rows: list[dict] = []
    status_updates: dict[str, list[tuple[str, str]]] = {}  # status -> [(source, id)]

    total = len(to_process)
    for i, meeting in enumerate(to_process):
        results = matcher.match(
            meeting["text"],
            meeting_date=meeting["date"],
            related_procedure=meeting.get("related_procedure"),
            org_name=meeting.get("org_name"),
        )

        if not results:
            status_updates.setdefault("no_match", []).append((meeting["source"], meeting["id"]))
            stats["no_match"] += 1
        elif results[0].get("_needs_ai"):
            need_ai.append({
                "meeting": meeting,
                "candidates": results[0]["candidates"],
                "matched_alias": results[0].get("matched_alias"),
                "org_name": results[0].get("org_name"),
            })
        else:
            # Direct match(es)
            for r in results:
                method = r["match_method"]
                stats[method] = stats.get(method, 0) + 1
                link_rows.append({
                    "source": meeting["source"],
                    "meeting_id": meeting["id"],
                    "procedure_id": r["procedure_id"],
                    "match_method": method,
                    "match_confidence": r["confidence"],
                })
            status_updates.setdefault("matched", []).append((meeting["source"], meeting["id"]))

        if (i + 1) % 10000 == 0:
            _log(
                f"  Progress: {i+1}/{total} — "
                f"exact_id={stats['exact_id']}, alias={stats['alias_exact']}, "
                f"no_match={stats['no_match']}, need_ai={len(need_ai)}"
            )

    _log(
        f"Steps 1-2 complete: {stats['exact_id']} exact ID, {stats['alias_exact']} alias exact, "
        f"{stats['no_match']} no_match, {len(need_ai)} alias_substring → AI"
    )

    # Step 4: AI classification — concurrent with progress + graceful quota skip
    if need_ai and anthropic_client:
        from concurrent.futures import ThreadPoolExecutor

        ai_batches = [need_ai[i:i + ai_batch_size] for i in range(0, len(need_ai), ai_batch_size)]
        _log(
            f"AI classification: {len(need_ai)} meetings in {len(ai_batches)} batches "
            f"(workers={workers})"
        )

        def _prepare(batch: list[dict]) -> list[dict]:
            items = []
            for it in batch:
                items.append({
                    "text": it["meeting"]["text"],
                    "date": it["meeting"]["date"],
                    "org_name": it["meeting"].get("org_name") or it.get("org_name"),
                    "matched_alias": it.get("matched_alias"),
                    "candidates": it["candidates"],
                    "_proc_details": {
                        c["procedure_id"]: matcher._by_id.get(c["procedure_id"], {})
                        for c in it["candidates"]
                    },
                })
            return items

        def _run(batch: list[dict]) -> tuple[list[dict], list[dict]]:
            return batch, ai_classify_batch(_prepare(batch), anthropic_client)

        ai_skipped = False
        report_every = max(1, len(ai_batches) // 20)
        processed_batches = 0
        matched_so_far = 0
        batch_errors = 0
        remaining_batches: list[list[dict]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run, b) for b in ai_batches]
            for idx, fut in enumerate(futures, 1):
                if ai_skipped:
                    remaining_batches.append(ai_batches[idx - 1])
                    continue
                try:
                    batch, ai_results = fut.result()
                except AIQuotaError as e:
                    _log(
                        f"AI quota/auth error at batch {idx}/{len(ai_batches)}: {e}. "
                        f"Skipping remaining AI classification — "
                        f"{len(ai_batches) - idx + 1} batches stay match_status=NULL "
                        f"for a future retry."
                    )
                    ai_skipped = True
                    remaining_batches.append(ai_batches[idx - 1])
                    continue
                except AIBatchError as e:
                    batch_errors += 1
                    if batch_errors <= 5:
                        _log(f"Batch {idx} error ({e}) — leaving as NULL for retry")
                    remaining_batches.append(ai_batches[idx - 1])
                    continue
                except Exception as e:
                    batch_errors += 1
                    if batch_errors <= 5:
                        _log(f"Batch {idx} unexpected error ({e}) — leaving as NULL for retry")
                    remaining_batches.append(ai_batches[idx - 1])
                    continue

                for item, ai in zip(batch, ai_results):
                    meeting = item["meeting"]
                    confidence = ai["match"]
                    chosen_idx = ai["chosen_index"]
                    candidates = item["candidates"]
                    if confidence == "high" and 0 <= chosen_idx < len(candidates):
                        chosen = candidates[chosen_idx]
                        stats[f"ai_{confidence}"] = stats.get(f"ai_{confidence}", 0) + 1
                        matched_so_far += 1
                        link_rows.append({
                            "source": meeting["source"],
                            "meeting_id": meeting["id"],
                            "procedure_id": chosen["procedure_id"],
                            "match_method": f"ai_{confidence}",
                            "match_confidence": 0.9 if confidence == "high" else 0.6,
                        })
                        status_updates.setdefault("matched", []).append((meeting["source"], meeting["id"]))
                    else:
                        stats["no_match"] += 1
                        status_updates.setdefault("no_match", []).append((meeting["source"], meeting["id"]))

                processed_batches += 1
                if processed_batches % report_every == 0:
                    _log(
                        f"  AI progress: {processed_batches}/{len(ai_batches)} batches, "
                        f"{matched_so_far} matches so far"
                    )

        # Meetings whose AI batch was skipped (quota/auth failure) or errored
        # (transient, malformed response, etc.) stay with match_status=NULL so
        # a future run with working AI reprocesses them.
        skipped_meetings = sum(len(b) for b in remaining_batches)
        if skipped_meetings:
            stats["ai_skipped_meetings"] = skipped_meetings
            stats["ai_skipped_batches"] = len(remaining_batches)
            stats["ai_batch_errors"] = batch_errors
            _log(
                f"{skipped_meetings} meetings across {len(remaining_batches)} batches "
                f"left with match_status=NULL "
                f"(ai_skipped={ai_skipped}, batch_errors={batch_errors}) — "
                f"will be retried on next matcher run"
            )

    elif need_ai:
        # No anthropic_client at all: same principle — don't permanently mark
        # these as no_match, so a later run with a key can still classify them.
        _log(
            f"Skipping AI for {len(need_ai)} meetings (no anthropic_client). "
            f"Leaving match_status=NULL so they're retried later."
        )
        stats["ai_skipped_meetings"] = len(need_ai)

    _log(f"Total: {len(link_rows)} links to create")

    # Write results
    if not dry_run and link_rows:
        _write_links(supabase_client, link_rows, logger)
        stats["links_created"] = len(link_rows)

    if not dry_run:
        _update_match_status(supabase_client, status_updates, logger)

    return stats


def _write_links(client: Any, links: list[dict], logger: Optional[Any] = None) -> None:
    """Write meeting_procedure_links rows to Supabase."""
    records = []
    for link in links:
        record: dict[str, Any] = {
            "procedure_id": link["procedure_id"],
            "match_method": link["match_method"],
            "match_confidence": link["match_confidence"],
            "is_primary": True,
            "match_rank": 1,
        }
        if link["source"] == "lobbying":
            record["lobbying_meeting_id"] = link["meeting_id"]
        else:
            record["commission_meeting_id"] = link["meeting_id"]
        records.append(record)

    # Batch insert
    for i in range(0, len(records), 100):
        batch = records[i:i + 100]
        try:
            client.table("meeting_procedure_links").insert(batch).execute()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to insert link batch: {e}")


def _update_match_status(
    client: Any,
    updates: dict[str, list[tuple[str, str]]],
    logger: Optional[Any] = None,
) -> None:
    """Update match_status on lobbying_meetings and commission_meetings."""
    for status, entries in updates.items():
        lobbying_ids = [eid for source, eid in entries if source == "lobbying"]
        commission_ids = [eid for source, eid in entries if source == "commission"]

        for ids, table in [(lobbying_ids, "lobbying_meetings"), (commission_ids, "commission_meetings")]:
            if not ids:
                continue
            # Meeting IDs are sha256 hashes (64 chars each). PostgREST puts
            # the ``.in_`` list in the URL query string, which Supabase caps
            # around 16KB. 50 IDs * ~66 chars keeps us well below the limit.
            failed = 0
            for i in range(0, len(ids), 50):
                batch = ids[i:i + 50]
                try:
                    client.table(table).update(
                        {"match_status": status}
                    ).in_("id", batch).execute()
                except Exception as e:
                    failed += len(batch)
                    if logger and failed <= 5:
                        logger.warning(f"Failed to update match_status on {table}: {e}")
            if failed and logger:
                logger.warning(
                    f"{failed} rows on {table} failed match_status={status} update"
                )
