"""Unit tests for influence_pipeline.py helper functions.

Run with:
    source .venv/bin/activate
    pytest tests/test_influence_pipeline.py -v

All tests operate on pure-Python helpers; no Supabase or AI calls are made.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Module loading — stub out supabase and google.genai so we can import
# the script without those SDKs being initialised.
# ---------------------------------------------------------------------------


def _load_pipeline_module():
    """Load influence_pipeline as a module with all external SDKs stubbed."""
    # Stub supabase
    fake_sb = types.ModuleType("supabase")
    fake_sb.create_client = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["supabase"] = fake_sb

    # Stub google.genai
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_genai_types = types.ModuleType("google.genai.types")

    class _FakeGenerateContentConfig:  # noqa: D101
        def __init__(self, **kwargs):
            pass

    fake_genai_types.GenerateContentConfig = _FakeGenerateContentConfig
    fake_genai.types = fake_genai_types
    fake_genai.Client = lambda **kwargs: None  # type: ignore[attr-defined]
    fake_google.genai = fake_genai
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_genai_types

    script_path = Path(__file__).parent.parent / "scripts" / "influence_pipeline.py"
    spec = importlib.util.spec_from_file_location("influence_pipeline", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ip = _load_pipeline_module()


# ---------------------------------------------------------------------------
# safe_id
# ---------------------------------------------------------------------------


def test_safe_id_basic():
    assert ip.safe_id("2023/0212(COD)") == "2023_0212_COD"


def test_safe_id_no_special_chars():
    assert ip.safe_id("2024_0001_NLE") == "2024_0001_NLE"


def test_safe_id_leading_trailing_underscores_stripped():
    # Parentheses at start/end should not leave leading/trailing underscores
    result = ip.safe_id("(TEST)")
    assert not result.startswith("_")
    assert not result.endswith("_")


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_valid_object():
    result = ip.parse_json_response('{"key": 42}')
    assert result == {"key": 42}


def test_parse_json_valid_array():
    result = ip.parse_json_response('[{"a": 1}, {"b": 2}]')
    assert result == [{"a": 1}, {"b": 2}]


def test_parse_json_with_markdown_fence():
    fenced = "```json\n{\"key\": 99}\n```"
    result = ip.parse_json_response(fenced)
    assert result == {"key": 99}


def test_parse_json_with_plain_fence():
    fenced = "```\n{\"x\": true}\n```"
    result = ip.parse_json_response(fenced)
    assert result == {"x": True}


def test_parse_json_invalid_no_retry():
    result = ip.parse_json_response("not json at all", retry_prompt="")
    assert result is None


def test_parse_json_invalid_without_retry_prompt():
    # When retry_prompt is empty string, should not attempt retry
    result = ip.parse_json_response("{broken json", retry_prompt="")
    assert result is None


# ---------------------------------------------------------------------------
# Regex classification
# ---------------------------------------------------------------------------


def test_compile_taxonomy_patterns_valid():
    taxonomy = {
        "privacy": {"keywords": [r"personal\s+data", r"\bGDPR\b"]},
        "fees": {"keywords": [r"\bfee\b", r"interchange"]},
    }
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    assert "privacy" in compiled
    assert "fees" in compiled
    assert len(compiled["privacy"]) == 2


def test_compile_taxonomy_patterns_invalid_regex_skipped():
    taxonomy = {"bad_theme": {"keywords": [r"[invalid", r"\bvalid\b"]}}
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    # Only the valid pattern should compile; invalid one is skipped
    assert len(compiled["bad_theme"]) == 1


def test_classify_by_regex_matches():
    taxonomy = {"privacy": {"keywords": [r"\bGDPR\b", r"personal\s+data"]}}
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    themes = ip._classify_by_regex("This amendment concerns GDPR compliance.", compiled)
    assert themes == ["privacy"]


def test_classify_by_regex_no_match():
    taxonomy = {"privacy": {"keywords": [r"\bGDPR\b"]}}
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    themes = ip._classify_by_regex("This is about holding limits and fees.", compiled)
    assert themes == []


def test_classify_by_regex_multiple_themes():
    taxonomy = {
        "privacy": {"keywords": [r"\bGDPR\b"]},
        "fees": {"keywords": [r"\bfee\b"]},
    }
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    themes = ip._classify_by_regex("The GDPR fee regime is contested.", compiled)
    assert set(themes) == {"privacy", "fees"}


def test_classify_by_regex_empty_text():
    taxonomy = {"privacy": {"keywords": [r"\bGDPR\b"]}}
    compiled = ip._compile_taxonomy_patterns(taxonomy)
    assert ip._classify_by_regex("", compiled) == []


# ---------------------------------------------------------------------------
# LEI (Lobby Exposure Index)
# ---------------------------------------------------------------------------

_OWN_INTEREST = "Promotes their own interests or the collective interests of their members"
_CLIENT_INTEREST = "Advances interests of their clients"
_NON_COMMERCIAL = "Does not represent commercial interests"


def test_lei_basic():
    crossref = {"top_orgs_met": [{"org": "BigBank", "count": 5}], "total_meetings": 5}
    org_inf = {"BigBank": {"interests_represented": _OWN_INTEREST}}
    lei = ip._compute_lei(crossref, org_inf, 100)
    # 5 * 1.0 / 100 = 0.05
    assert abs(lei - 0.05) < 1e-9


def test_lei_zero_total_meetings():
    crossref = {"top_orgs_met": [{"org": "Org", "count": 5}], "total_meetings": 5}
    org_inf = {"Org": {"interests_represented": _OWN_INTEREST}}
    assert ip._compute_lei(crossref, org_inf, 0) == 0.0


def test_lei_mixed_weights():
    crossref = {
        "top_orgs_met": [
            {"org": "Commercial", "count": 4},
            {"org": "NGO", "count": 4},
        ],
        "total_meetings": 8,
    }
    org_inf = {
        "Commercial": {"interests_represented": _OWN_INTEREST},  # weight 1.0
        "NGO": {"interests_represented": _NON_COMMERCIAL},  # weight 0.3
    }
    lei = ip._compute_lei(crossref, org_inf, 100)
    # (4*1.0 + 4*0.3) / 100 = 5.2 / 100 = 0.052
    assert abs(lei - 0.052) < 1e-9


def test_lei_unknown_org():
    crossref = {"top_orgs_met": [{"org": "Mystery Org", "count": 10}], "total_meetings": 10}
    org_inf = {}  # org not in influence map
    lei = ip._compute_lei(crossref, org_inf, 100)
    # weight defaults to 0.5; 10 * 0.5 / 100 = 0.05
    assert abs(lei - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# ALAS (Amendment-Lobby Alignment Score)
# ---------------------------------------------------------------------------


def test_alas_basic():
    crossref = {
        "total_amendments": 10,
        "total_meetings": 10,
        "amendment_themes": {"privacy": 5, "fees": 5},
        "meeting_themes": {"privacy": 3, "fees": 7},
        "overlapping_themes": ["privacy", "fees"],
    }
    alas = ip._compute_alas(crossref)
    # raw_sum = (5/10)*(3/10) + (5/10)*(7/10) = 0.15 + 0.35 = 0.50
    assert abs(alas - math.sqrt(0.5)) < 1e-9


def test_alas_no_overlap():
    crossref = {
        "total_amendments": 10,
        "total_meetings": 10,
        "amendment_themes": {"privacy": 10},
        "meeting_themes": {"fees": 10},
        "overlapping_themes": [],
    }
    assert ip._compute_alas(crossref) == 0.0


def test_alas_zero_amendments():
    crossref = {
        "total_amendments": 0,
        "total_meetings": 10,
        "amendment_themes": {},
        "meeting_themes": {"fees": 10},
        "overlapping_themes": [],
    }
    assert ip._compute_alas(crossref) == 0.0


def test_alas_zero_meetings():
    crossref = {
        "total_amendments": 10,
        "total_meetings": 0,
        "amendment_themes": {"privacy": 10},
        "meeting_themes": {},
        "overlapping_themes": [],
    }
    assert ip._compute_alas(crossref) == 0.0


def test_alas_perfect_concentration():
    """Single theme, all amendments and meetings on it -> ALAS = 1.0."""
    crossref = {
        "total_amendments": 5,
        "total_meetings": 5,
        "amendment_themes": {"holding": 5},
        "meeting_themes": {"holding": 5},
        "overlapping_themes": ["holding"],
    }
    alas = ip._compute_alas(crossref)
    assert abs(alas - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# ICI (Industry Concentration Index)
# ---------------------------------------------------------------------------


def test_ici_single_org():
    crossref = {"top_orgs_met": [{"org": "A", "count": 10}], "total_meetings": 10}
    assert abs(ip._compute_ici(crossref) - 1.0) < 1e-9


def test_ici_two_equal_orgs():
    crossref = {
        "top_orgs_met": [{"org": "A", "count": 5}, {"org": "B", "count": 5}],
        "total_meetings": 10,
    }
    assert abs(ip._compute_ici(crossref) - 0.5) < 1e-9


def test_ici_zero_meetings():
    crossref = {"top_orgs_met": [], "total_meetings": 0}
    assert ip._compute_ici(crossref) == 0.0


def test_ici_decreases_with_more_orgs():
    """ICI should decrease as meetings are spread across more organisations."""
    crossref_conc = {
        "top_orgs_met": [{"org": "A", "count": 10}],
        "total_meetings": 10,
    }
    crossref_spread = {
        "top_orgs_met": [{"org": chr(65 + i), "count": 1} for i in range(10)],
        "total_meetings": 10,
    }
    assert ip._compute_ici(crossref_conc) > ip._compute_ici(crossref_spread)


# ---------------------------------------------------------------------------
# _extract_rapporteurs
# ---------------------------------------------------------------------------


def test_extract_rapporteurs_basic():
    proc = {
        "actors": [
            {"name": "Alice EXAMPLE", "role": "COM_RAPP", "group": "EPP"},
            {"name": "Bob SHADOW", "role": "COM_SHADOW", "group": "S&D"},
            {"name": "Ignore ME", "role": "OTHER", "group": "X"},
        ]
    }
    rapp = ip._extract_rapporteurs(proc)
    assert "Alice EXAMPLE" in rapp
    assert rapp["Alice EXAMPLE"]["role"] == "Rapporteur"
    assert rapp["Alice EXAMPLE"]["party"] == "EPP"
    assert "Bob SHADOW" in rapp
    assert rapp["Bob SHADOW"]["role"] == "Shadow"
    assert "Ignore ME" not in rapp


def test_extract_rapporteurs_empty_actors():
    assert ip._extract_rapporteurs({"actors": []}) == {}


def test_extract_rapporteurs_no_actors_field():
    assert ip._extract_rapporteurs({}) == {}


def test_extract_rapporteurs_json_string_actors():
    """actors field stored as JSON string should be parsed."""
    import json
    actors = json.dumps([{"name": "Carol R.", "role": "COM_RAPP", "group": "Renew"}])
    proc = {"actors": actors}
    rapp = ip._extract_rapporteurs(proc)
    assert "Carol R." in rapp


# ---------------------------------------------------------------------------
# _body_text noise filtering
# ---------------------------------------------------------------------------


def test_body_text_removes_noise():
    lines = ["PE778136.123v01-00", "Some real text", "EN", "More real text"]
    body = ip._body_text(lines)
    assert "PE778136" not in body
    assert "Some real text" in body
    assert "More real text" in body


def test_body_text_removes_language_marker():
    lines = ["Amendment content here", "Or. en"]
    body = ip._body_text(lines)
    assert "Or. en" not in body
    assert "Amendment content here" in body


def test_body_text_empty_lines_stripped():
    lines = ["  ", "content", "   "]
    body = ip._body_text(lines)
    assert body.strip() == "content"


# ---------------------------------------------------------------------------
# _meeting_text
# ---------------------------------------------------------------------------


def test_meeting_text_concatenates_fields():
    meeting = {
        "subject": "Digital euro",
        "points_raised": "Privacy concerns",
        "conclusions": "Follow up needed",
        "title": "Meeting title",
    }
    text = ip._meeting_text(meeting)
    assert "Digital euro" in text
    assert "Privacy concerns" in text
    assert "Follow up needed" in text
    assert "Meeting title" in text


def test_meeting_text_handles_none_fields():
    meeting = {"subject": None, "points_raised": "Some text", "conclusions": None}
    text = ip._meeting_text(meeting)
    assert text == "Some text"


# ---------------------------------------------------------------------------
# _interest_weight
# ---------------------------------------------------------------------------


def test_interest_weight_own():
    w = ip._interest_weight(
        "Promotes their own interests or the collective interests of their members"
    )
    assert w == 1.0


def test_interest_weight_client():
    w = ip._interest_weight("Advances interests of their clients")
    assert w == 0.8


def test_interest_weight_non_commercial():
    w = ip._interest_weight("Does not represent commercial interests")
    assert w == 0.3


def test_interest_weight_unknown():
    assert ip._interest_weight("Unknown") == 0.5


def test_interest_weight_unrecognised_defaults():
    assert ip._interest_weight("Something completely different") == 0.5
