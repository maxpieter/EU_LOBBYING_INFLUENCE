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

# Precompile hot-path regexes used inside ProcedureMatcher.match()
_RE_WHITESPACE = re.compile(r"\s+")

# Procedure ID format: "2025/0102(COD)", "2024/0123(CNS)", etc.
# Used to extract direct procedure references from meeting text — lobbyists
# sometimes paste the OEIL reference into the meeting title.
_RE_PROCEDURE_ID = re.compile(r"\b(\d{4}/\d{4}\([A-Z]{2,4}\))")

# Common filler prefixes across EU working languages. Stripping these makes
# the alias_exact step fire on titles like "Exchange of views on Critical
# Medicines Act" → "critical medicines act" (which IS an exact alias).
_RE_FILLER_PREFIX = re.compile(
    r"^(?:"
    # English
    r"(?:general\s+)?exchange\s+of\s+views?\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"meeting\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"discussion\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"briefing\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"presentation\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"update\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    r"debate\s+(?:on|about|regarding|concerning)\s+(?:the\s+)?|"
    # French — handles é/e and Unicode/ASCII apostrophes (l' vs l’)
    r"(?:é|e)change\s+(?:de\s+)?vues?\s+sur\s+(?:la\s+|le\s+|les\s+|l[’'])?|"
    r"r(?:é|e)union\s+sur\s+(?:la\s+|le\s+|les\s+|l[’'])?|"
    r"discussion\s+sur\s+(?:la\s+|le\s+|les\s+|l[’'])?|"
    r"pr(?:é|e)sentation\s+sur\s+(?:la\s+|le\s+|les\s+|l[’'])?|"
    # German
    r"austausch\s+(?:zu|über|ueber)\s+(?:der\s+|die\s+|das\s+|den\s+|dem\s+)?|"
    r"sitzung\s+zu\s+(?:der\s+|die\s+|das\s+|den\s+|dem\s+)?|"
    r"treffen\s+zu\s+(?:der\s+|die\s+|das\s+|den\s+|dem\s+)?|"
    r"diskussion\s+(?:über|ueber)\s+(?:der\s+|die\s+|das\s+|den\s+|dem\s+)?|"
    r"pr(?:ä|ae)sentation\s+(?:zu|über|ueber)\s+(?:der\s+|die\s+|das\s+|den\s+|dem\s+)?|"
    # Italian — "su" can contract with the article (sulla, sullo, sulle, sui, …).
    # Apostrophe forms (sull') elide directly into the next word, no \s+ after.
    r"scambio\s+(?:di\s+)?opinioni\s+(?:(?:sul|sullo|sulla|sulle|sui|sugli)\s+|sull[’']|su\s+(?:(?:la|il|lo|le|gli|i)\s+|l[’']))|"
    r"riunione\s+(?:(?:sul|sullo|sulla|sulle|sui|sugli)\s+|sull[’']|su\s+(?:(?:la|il|lo|le|gli|i)\s+|l[’']))|"
    # Spanish
    r"intercambio\s+(?:de\s+)?opiniones\s+sobre\s+(?:el\s+|la\s+|los\s+|las\s+)?|"
    r"reuni(?:ó|o)n\s+sobre\s+(?:el\s+|la\s+|los\s+|las\s+)?|"
    # Dutch
    r"uitwisseling\s+van\s+(?:standpunten|meningen)\s+over\s+(?:de\s+|het\s+)?|"
    r"vergadering\s+over\s+(?:de\s+|het\s+)?"
    r")",
    re.IGNORECASE,
)


def _date_proximity_days(meeting_date: Optional[str], proc: dict) -> int:
    """Days between meeting and the procedure's nearest date anchor.

    Used to rank multiple candidate procedures (e.g. when one alias maps to
    several files) so the temporally-closest one is marked is_primary=True.
    Procedures with no usable date sort last (returns a sentinel large int).
    """
    if not meeting_date:
        return 10**9
    try:
        m = date.fromisoformat(meeting_date)
    except (ValueError, TypeError):
        return 10**9
    anchors = []
    for k in ("proposal_date", "decision_date", "last_activity_date"):
        v = proc.get(k)
        if v:
            try:
                anchors.append(date.fromisoformat(str(v)))
            except (ValueError, TypeError):
                pass
    if not anchors:
        return 10**9
    return min(abs((m - a).days) for a in anchors)


# ---------------------------------------------------------------------------
# Temporal filtering
# ---------------------------------------------------------------------------

def _is_temporally_valid(
    meeting_date: Optional[str],
    procedure: dict,
    buffer_months: int = 6,
    pre_proposal_buffer_months: int = 18,
) -> bool:
    """Check if a meeting date falls within a procedure's active window.

    Window: proposal_date - pre_buffer → decision_date + buffer
    (or last_activity + buffer). The pre-proposal buffer captures the
    agenda-setting / consultation / drafting phase, when lobbying is
    typically most intense — HYS feedback periods, inception impact
    assessments, expert groups, and Commission work programme signalling
    routinely run 6–18 months before formal proposal publication.
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
    pre_buffer = timedelta(days=pre_proposal_buffer_months * 30)

    # Start: proposal date - pre_buffer (or 2 years before as fallback)
    if proposal:
        try:
            start = date.fromisoformat(str(proposal)) - pre_buffer
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

        # Dedupe — same procedure_id can be indexed both via procedure_aliases
        # (e.g. an alias matching the title) AND via the title fallback above,
        # which would otherwise produce duplicate results in alias_exact.
        for k, ids in self._alias_to_procs.items():
            self._alias_to_procs[k] = list(dict.fromkeys(ids))

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
        """Run the cascade on a single meeting.

        Cascade (high → low reliability):
        1a. exact_id:       related_procedure field matches procedures.id
        1b. exact_id:       a procedure_id (e.g. 2025/0102(COD)) appears verbatim
                            in the meeting text
        2.  alias_exact:    full meeting text (after filler stripping) = known alias
        3.  alias_substring: known alias appears within the text → send to AI

        Each result dict carries a `matched_alias` field (empty string for
        exact_id paths) so the caller can persist it as audit trail.
        """
        results = []

        # Step 1a: Exact ID match via the structured related_procedure field
        if related_procedure:
            proc_id = related_procedure.strip()
            if proc_id in self._by_id:
                proc = self._by_id[proc_id]
                if _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": proc_id,
                        "match_method": "exact_id",
                        "matched_alias": "",
                    })
                    return results

        if not meeting_text or not meeting_text.strip():
            return []

        text_lower = meeting_text.strip().lower()
        text_normalized = _RE_WHITESPACE.sub(" ", text_lower)

        # Step 1b: Procedure ID cited verbatim in meeting text
        # (e.g. lobbying titles "2023/0266(COD) Energy Directive") — caught
        # case-insensitive then re-cased to match procedures.id format.
        for raw in _RE_PROCEDURE_ID.findall(meeting_text or ""):
            pid = raw.strip()
            if pid in self._by_id:
                proc = self._by_id[pid]
                if _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": pid,
                        "match_method": "exact_id",
                        "matched_alias": pid,  # the cited reference itself
                    })
        if results:
            return results

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
                        "matched_alias": text_normalized,
                    })
            if results:
                return results

        # Strip common filler prefixes
        stripped = _RE_FILLER_PREFIX.sub("", text_normalized).strip()

        # Step 2b: Alias exact on stripped text
        if stripped != text_normalized and stripped in self._alias_to_procs:
            proc_ids = self._alias_to_procs[stripped]
            for pid in proc_ids:
                proc = self._by_id.get(pid)
                if proc and _is_temporally_valid(meeting_date, proc):
                    results.append({
                        "procedure_id": pid,
                        "match_method": "alias_exact",
                        "matched_alias": stripped,
                    })
            if results:
                return results

        fuzzy_text = stripped if stripped and len(stripped) >= 5 else text_normalized

        # No word-count gate. A 2-word title like "AI Act" or "digital euro"
        # IS the meeting topic; the alias-substring step should fire on it.
        # Protection against false positives lives downstream in the alias
        # index itself (single_word_aliases requires len >= 4 chars; multi-word
        # aliases require " " in the alias) and in the AI confirmation step.

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

# Static prompt prefix — identical across all batches in a run, so it lands
# in the prompt cache (cache_control={"type":"ephemeral"} in the API call).
# Sized to clear Sonnet's 1024-token minimum cacheable prefix; cache hits
# bill at ~10% of normal input rate, dropping per-run AI cost ~10x.
_PROMPT_STATIC_PREFIX = (
    "You are matching EU lobbying / Commission meetings to specific "
    "legislative procedures. A keyword from the candidate file's title or "
    "alias was found in the meeting text — your job is to confirm whether "
    "the meeting discusses that file as a meaningful topic.\n\n"
    "Note: meeting text may be in any EU language (English, French, German, "
    "Italian, Spanish, Dutch, Polish, Portuguese, Romanian, etc.). Subjects "
    "are always listed in English; cross-translate as needed when judging "
    "whether the meeting matches.\n\n"
    "Rules:\n"
    "- Mark 'high' if the meeting clearly discusses the candidate file as a "
    "real topic. It does NOT need to be the only or primary subject — a "
    "meeting covering several files (e.g. 'Fit for 55 package: ETS, CBAM, "
    "RED') matches all of those that are substantively discussed.\n"
    "- Mark 'no_match' only when the keyword hit is truly incidental "
    "(passing mention, generic topic overlap, the file is named but not "
    "actually discussed).\n"
    "- Use the organisation name (if given), procedure subjects, and active "
    "dates as context. A steel association in 2023 talking about "
    "'emissions' is far more likely about the ETS than a 1990s file. "
    "Conversely, a meeting in 2018 cannot be about a 2023 procedure.\n"
    "- When several candidates are closely related (e.g. an original "
    "directive and its recast), pick the one most temporally consistent "
    "with the meeting date and most aligned with the meeting's stated "
    "scope. Don't match an old original when context points to a recent "
    "amendment.\n"
    "- A meeting that ONLY contains the alias keyword as part of a list of "
    "topics, with no further substantive discussion, is no_match.\n\n"
    "Worked examples:\n\n"
    "Example 1 (clear match — file mentioned by name and discussed):\n"
    '  Meeting: "Digital policy: Impact of the AI Act on transparency and trust"\n'
    '  Candidate A. "Artificial Intelligence Act" [2021/0106(COD)] [active: 2021-04-21 → 2024-07-12; subjects: Information and communication technology, Civil rights]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "Meeting explicitly addresses AI Act impact on transparency."}\n\n'
    "Example 2 (incidental keyword, real topic is broader):\n"
    '  Meeting: "Energy and trade roundtable, general industry concerns"\n'
    '  Candidate A. "Energy Efficiency Directive (Recast)" [2021/0203(COD)] [active: 2021-07-14 → 2023-09-13; subjects: Energy efficiency, Buildings]\n'
    '  → {"match": "no_match", "chosen": "none", "reasoning": "Meeting is a generic energy/trade discussion; EED is never specifically referenced."}\n\n'
    "Example 3 (non-English meeting — translate before judging):\n"
    '  Meeting: "Échange de vues sur le règlement sur les médicaments critiques"\n'
    '  Candidate A. "Critical Medicines Act" [2025/0102(COD)] [active: 2025-03-11 → ongoing; subjects: Public health]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "French title directly references the Critical Medicines Act regulation."}\n\n'
    "Example 4 (multiple candidates — pick the temporally closest):\n"
    '  Meeting (date: 2024-05-10): "Discussion on revised Energy Performance of Buildings Directive"\n'
    '  Candidate A. "Energy Performance of Buildings Directive (Recast)" [2021/0426(COD)] [active: 2021-12-15 → 2024-05-08; subjects: Buildings, Energy efficiency]\n'
    '  Candidate B. "Energy Performance of Buildings Directive" [2008/0223(COD)] [active: 2008-11-13 → 2010-06-19]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "Meeting in 2024 about \'revised EPBD\' aligns with the 2021 recast that concluded days before."}\n\n'
    "Example 5 (wrong-version mismatch — meeting context implies a different file):\n"
    '  Meeting (date: 2023-09-15): "Position paper on the new Carcinogens Directive amendment for chemical workers"\n'
    '  Candidate A. "Protection of workers from carcinogens or mutagens (4th amendment)" [2020/0030(COD)] [active: 2020-09-22 → 2022-03-09; subjects: Occupational health, Carcinogens]\n'
    '  Candidate B. "Protection of workers from carcinogens or mutagens (3rd amendment)" [2018/0081(COD)] [active: 2018-04-05 → 2019-06-20]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "Meeting in 2023 about new amendment fits the 4th amendment (concluded 2022); the 3rd amendment was three years earlier and superseded."}\n\n'
    "Example 6 (multi-topic meeting that genuinely covers the candidate file):\n"
    '  Meeting: "Fit for 55 package — exchange on ETS, CBAM, and the Effort Sharing Regulation"\n'
    '  Candidate A. "Carbon border adjustment mechanism" [2021/0214(COD)] [active: 2021-07-14 → 2023-05-10; subjects: Trade, Climate change, Carbon emissions]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "CBAM is named as one of three substantively-discussed Fit for 55 files."}\n\n'
    "Example 7 (industry/org context disambiguates a generic keyword):\n"
    '  Meeting (org: European Steel Association): "Trade and emissions impact discussion"\n'
    '  Candidate A. "EU Emissions Trading System (revision)" [2021/0211(COD)] [active: 2021-07-14 → 2023-05-10; subjects: Industrial emissions, Climate change]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "A steel association meeting in 2021-2023 discussing emissions and trade is overwhelmingly likely to be about the ETS revision affecting their sector."}\n\n'
    "Example 8 (post-decision discussion still matches — implementation of an adopted file):\n"
    '  Meeting (date: 2024-09-20): "Implementing acts under the Digital Services Act — VLOP compliance"\n'
    '  Candidate A. "Digital Services Act" [2020/0361(COD)] [active: 2020-12-15 → 2022-10-19; subjects: Internal market, Digital services]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "Meeting discusses DSA implementation/enforcement specifically — post-adoption discussion of the regulation itself is a substantive match."}\n\n'
    "Common ambiguities to handle correctly:\n"
    "- Meeting names a 'package' that contains many files (e.g. 'Fit for 55', 'Banking Package', "
    "'Pharmaceutical Package'): match to specific files in the package only when those files are "
    "named or unambiguously implied. Do not match every package member by default.\n"
    "- Meeting mentions a related-but-distinct file: e.g. discussion of 'CBAM' should not match "
    "'EU ETS' just because they're climate-adjacent. Each file is its own match decision.\n"
    "- 'Package of revisions' or 'omnibus' meetings: if multiple files are amended together and "
    "the meeting discusses the omnibus generally, match each file in the omnibus that is "
    "individually mentioned or clearly implied.\n"
    "- Same alias maps to multiple files (e.g. several 'Plastics' regulations exist): use the "
    "candidate's date range to pick the temporally consistent one. A 2024 meeting about "
    "'plastics regulation' fits a 2023 file, not a 2010 one.\n"
    "- Meeting in language X about a file the candidates display in English: translate the "
    "meeting topic and judge on substance, not exact lexical overlap.\n"
    "- Meeting describes a STAGE of a procedure (e.g. 'trilogue on directive X', 'co-decision "
    "vote on regulation Y'): match the procedure file the stage belongs to. The stage descriptor "
    "is meeting-context, not a separate file.\n"
    "- Lobbying meetings often list several procedures the lobbyist wants to influence; match "
    "each procedure that is named or substantively discussed, not just the first one.\n\n"
    "Example 9 (correct rejection — alias appears but not as the meeting's subject):\n"
    '  Meeting: "Annual general assembly. Topics: organisational matters, board election, "\n'
    '            "year-end report, brief mention of CBAM and AI Act consultations."\n'
    '  Candidate A. "Carbon border adjustment mechanism" [2021/0214(COD)] [active: 2021-07-14 → 2023-05-10; subjects: Trade, Climate change]\n'
    '  → {"match": "no_match", "chosen": "none", "reasoning": "CBAM is one of several brief mentions in an organisational meeting; not a substantive discussion of the file."}\n\n'
    "Example 10 (file referenced by official acronym in non-English meeting):\n"
    '  Meeting (date: 2024-11-08, org: Pharma Industry Federation): "Treffen zur AMR-Verordnung — Stellungnahme der Industrie zum aktuellen Entwurf"\n'
    '  Candidate A. "Antimicrobial resistance — Action Plan and proposed regulation" [2023/0291(COD)] [active: 2023-04-26 → ongoing; subjects: Public health, Pharmaceuticals]\n'
    '  → {"match": "high", "chosen": "A", "reasoning": "German title \'AMR-Verordnung\' = AMR Regulation; meeting position-paper context fits the active 2023 file under industry consultation."}\n\n'
    "Now classify the meetings below.\n\n"
)


def _build_match_prompt_dynamic(batch: list[dict]) -> str:
    """The per-batch portion of the prompt — meeting items + output spec.

    Kept separate from `_PROMPT_STATIC_PREFIX` so the static part can be
    sent as a cached content block while this dynamic part is fresh per
    batch.
    """
    items = []
    for i, item in enumerate(batch):
        cand_lines = []
        for j, c in enumerate(item["candidates"]):
            proc = item.get("_proc_details", {}).get(c["procedure_id"], {})
            date_range = ""
            if proc.get("proposal_date"):
                date_range = f"active: {proc.get('proposal_date', '?')} → {proc.get('decision_date', 'ongoing')}"
            subjects = proc.get("subjects") or []
            subj_str = ""
            if isinstance(subjects, list) and subjects:
                subj_str = f"subjects: {', '.join(str(s) for s in subjects[:4])}"
            meta_bits = [b for b in (date_range, subj_str) if b]
            meta = f" [{'; '.join(meta_bits)}]" if meta_bits else ""
            cand_lines.append(
                f'    {chr(65+j)}. "{c["title"]}" [{c["procedure_id"]}]{meta}'
            )

        meeting_text = item["text"]
        date_str = f" (date: {item['date']})" if item.get("date") else ""
        org_str = f" (org: {item['org_name']})" if item.get("org_name") else ""
        alias_str = f' [keyword that fired: "{item["matched_alias"]}"]' if item.get("matched_alias") else ""

        items.append(
            f'{i+1}. Meeting{date_str}{org_str}: "{meeting_text}"{alias_str}\n'
            f'   Candidates:\n' + "\n".join(cand_lines)
        )

    return (
        "\n".join(items)
        + "\n\nRespond with a JSON array, one entry per meeting, in order:\n"
        '[{"match": "high"|"no_match", "chosen": "A"|"B"|"C"|"none", '
        '"reasoning": "one sentence"}, ...]\n'
        f"Return exactly {len(batch)} entries."
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
    model: str = "claude-sonnet-4-6",
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

    dynamic_prompt = _build_match_prompt_dynamic(batch)

    # Split into two content blocks so prompt caching activates on the
    # static prefix. With ~5K AI batches per full run, this is the
    # difference between paying full input rate every time vs paying ~10%
    # of it on cache hits — empirically ~10x cheaper for the AI phase.
    content_blocks = [
        {
            "type": "text",
            "text": _PROMPT_STATIC_PREFIX,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": dynamic_prompt},
    ]

    for attempt in range(5):
        try:
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": content_blocks}],
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

    # Collapse non-"high" responses to no_match (mirrors the org matcher).
    # Prompt only authorises "high" or "no_match"; if Haiku still emits
    # medium/low, treat as no_match.
    out = []
    for entry in parsed:
        if not isinstance(entry, dict):
            out.append({"match": "no_match", "chosen_index": -1, "reasoning": "invalid_entry"})
            continue
        match_val = "high" if entry.get("match") == "high" else "no_match"
        chosen_letter = str(entry.get("chosen", "none")).upper()
        chosen_index = (
            ord(chosen_letter) - 65
            if len(chosen_letter) == 1 and chosen_letter.isalpha() and chosen_letter in "ABC"
            else -1
        )
        out.append({
            "match": match_val,
            "chosen_index": chosen_index,
            "reasoning": str(entry.get("reasoning", ""))[:200],
        })
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
    n_lobbying = 0
    n_commission = 0

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
        n_lobbying += 1

    for m in commission_meetings:
        if m.get("match_status") is not None:
            continue
        # Subject only — points_raised and conclusions are discursive and
        # tend to mention legislative files in passing rather than as the
        # meeting's actual subject. Including them produced spurious alias
        # hits and inflated false positives in the gold-set evaluation
        # (commission F1=0.55 vs lobbying F1=0.67 with the long-text version).
        # Aligns with the prompt's "alias must be the meeting's actual
        # subject, not a passing reference" rule.
        to_process.append({
            "source": "commission",
            "id": m["id"],
            "text": m.get("subject") or "",
            "date": m.get("meeting_date"),
            "related_procedure": None,
            "org_name": None,
        })
        n_commission += 1

    _log(f"Meetings to process: {len(to_process)} ({n_lobbying} lobbying, {n_commission} commission)")

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
            # Direct match(es). When an alias maps to multiple procedures
            # (rare but real), rank them by temporal proximity to the meeting
            # date so the closest one is is_primary=True.
            ranked = sorted(
                results,
                key=lambda r: _date_proximity_days(meeting["date"],
                                                   matcher._by_id.get(r["procedure_id"], {})),
            )
            for rank, r in enumerate(ranked, start=1):
                method = r["match_method"]
                stats[method] = stats.get(method, 0) + 1
                link_rows.append({
                    "source": meeting["source"],
                    "meeting_id": meeting["id"],
                    "procedure_id": r["procedure_id"],
                    "match_method": method,
                    "matched_alias": r.get("matched_alias", ""),
                    "is_primary": rank == 1,
                    "match_rank": rank,
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
                            "matched_alias": item.get("matched_alias") or "",
                            "is_primary": True,
                            "match_rank": 1,
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
    """Write meeting_procedure_links rows to Supabase.

    `match_method` is the sole match-quality signal — the legacy
    `match_confidence` float column was dropped (migration
    20260504000001) because it carried no per-row info beyond what
    `match_method` already encodes. `match_details` JSONB carries the
    matched_alias for downstream debugging.
    """
    records = []
    for link in links:
        details: dict[str, Any] = {}
        if link.get("matched_alias"):
            details["matched_alias"] = link["matched_alias"]
        record: dict[str, Any] = {
            "procedure_id": link["procedure_id"],
            "match_method": link["match_method"],
            "is_primary": link.get("is_primary", True),
            "match_rank": link.get("match_rank", 1),
            "match_details": details or None,
        }
        if link["source"] == "lobbying":
            record["lobbying_meeting_id"] = link["meeting_id"]
        else:
            record["commission_meeting_id"] = link["meeting_id"]
        records.append(record)

    # Batch insert — inserts into a junction table carry no row-level locking
    # contention, so a much larger batch size is safe within the ~3s timeout.
    # 500 rows × ~200 bytes each ≈ 100KB payload, well within PostgREST limits.
    for i in range(0, len(records), 500):
        batch = records[i:i + 500]
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
