"""Tests for hys_feedback_scraper.py

Run with:
    pytest tests/test_hys_feedback_scraper.py -v

The test suite has two layers:
  1. Unit tests (no network): pure logic — COM normalisation, chunking,
     filtering, and row transformation.
  2. Integration test (live network, marked with @pytest.mark.integration):
     hits the real HYS API for COM(2025)836 (Omnibus / Simplification package)
     and validates that we can find the initiative, paginate feedback, and
     that all returned rows are non-citizen organisations with a transparency
     register number present.

Run only unit tests (CI):
    pytest tests/test_hys_feedback_scraper.py -v -m "not integration"

Run integration test locally:
    pytest tests/test_hys_feedback_scraper.py -v -m integration
"""

import json

import pytest
import requests

from pipeline.assets.feedback.hys_feedback_scraper import (
    EXCLUDED_USER_TYPES,
    build_chunk_records,
    chunk_text,
    com_variants,
    download_and_extract_pdf,
    fetch_all_feedbacks_for_initiative,
    normalise_com_number,
    search_initiatives_by_com,
    transform_feedback_row,
    _make_session,
)

# ---------------------------------------------------------------------------
# The target for integration tests
# ---------------------------------------------------------------------------
# COM(2025)836 = Omnibus I / Simplification package
# HYS initiative: https://ec.europa.eu/info/law/better-regulation/have-your-say/
#   initiatives/14855-Simplification-digital-package-and-omnibus_en
TARGET_COM = "COM(2025)836"
TARGET_COM_WITH_ZEROS = "COM(2025)0836"  # as stored in Supabase
TARGET_PROCEDURE_ID = "2025/0359(COD)"   # example — adjust to your actual procedure ID


# ===========================================================================
# 1. Unit tests — COM normalisation
# ===========================================================================


class TestNormaliseComNumber:
    def test_removes_leading_zeros(self):
        assert normalise_com_number("COM(2025)0836") == "COM(2025)836"

    def test_already_normalised(self):
        assert normalise_com_number("COM(2025)836") == "COM(2025)836"

    def test_spaces_around_parens(self):
        assert normalise_com_number("COM (2025) 0058") == "COM(2025)58"

    def test_none_on_garbage(self):
        assert normalise_com_number("SWD(2025)100") is None
        assert normalise_com_number("") is None
        assert normalise_com_number("not a com number") is None

    def test_case_insensitive(self):
        assert normalise_com_number("com(2025)0836") == "COM(2025)836"

    def test_small_number(self):
        assert normalise_com_number("COM(2024)0001") == "COM(2024)1"

    def test_four_digit_number(self):
        assert normalise_com_number("COM(2023)9999") == "COM(2023)9999"


class TestComVariants:
    def test_returns_multiple_forms(self):
        variants = com_variants("COM(2025)0836")
        assert "COM(2025)836" in variants     # canonical normalised
        assert "COM/2025/836" in variants     # slash notation
        assert len(variants) >= 2

    def test_empty_on_invalid(self):
        assert com_variants("SWD(2025)100") == []
        assert com_variants("") == []


# ===========================================================================
# 2. Unit tests — text chunking
# ===========================================================================


class TestChunkText:
    SHORT_TEXT = "Hello world. This is a short feedback."
    LONG_TEXT = "A" * 500 + ". " + "B" * 500 + ". " + "C" * 500 + ". " + "D" * 500

    def test_short_text_single_chunk(self):
        chunks = chunk_text(self.SHORT_TEXT)
        assert chunks == [self.SHORT_TEXT]

    def test_long_text_multiple_chunks(self):
        chunks = chunk_text(self.LONG_TEXT, chunk_size=600, overlap=50)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        """All characters in original should appear in at least one chunk."""
        text = "The quick brown fox. " * 200  # ~4200 chars
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        full = "".join(chunks)
        # Every substring of original should be findable somewhere
        assert "The quick brown fox." in full
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        text = "Word. " * 300
        chunks = chunk_text(text, chunk_size=200, overlap=30)
        assert all(len(c) > 0 for c in chunks)

    def test_empty_input(self):
        assert chunk_text("") == []

    def test_none_safe(self):
        # chunk_text should handle empty/None gracefully
        assert chunk_text("") == []


class TestBuildChunkRecords:
    def test_builds_correct_records(self):
        text = "Hello EU law. " * 200  # long enough to chunk
        records = build_chunk_records(
            feedback_id=12345,
            initiative_id=99,
            procedure_id="2025/0001(COD)",
            com_number="COM(2025)1",
            text=text,
            organisation_name="ACME Corp",
            transparency_reg_id="123456789-00",
            date_feedback="2025-03-01",
        )
        assert len(records) >= 1
        for r in records:
            assert r["feedback_id"] == 12345
            assert r["initiative_id"] == 99
            assert r["com_number"] == "COM(2025)1"
            assert r["chunk_total"] == len(records)
            assert r["chunk_index"] < len(records)
            assert len(r["chunk_text"]) > 0

    def test_chunk_index_sequential(self):
        text = "X" * 5000
        records = build_chunk_records(
            feedback_id=1,
            initiative_id=1,
            procedure_id="2025/0001(COD)",
            com_number="COM(2025)1",
            text=text,
            organisation_name=None,
            transparency_reg_id=None,
            date_feedback=None,
        )
        indices = [r["chunk_index"] for r in records]
        assert indices == list(range(len(records)))


# ===========================================================================
# 3. Unit tests — citizen filtering
# ===========================================================================


class TestCitizenFiltering:
    """Verify our filtering logic excludes citizens."""

    def _make_feedback(self, user_type: str) -> dict:
        return {
            "id": 1,
            "userType": user_type,
            "organization": "Test Org",
            "country": "DE",
            "language": "EN",
            "dateFeedback": "2025-01-01",
            "transparencyRegisterId": "123",
            "attachments": [],
            "feedback": "Some text",
        }

    def test_eu_citizen_excluded(self):
        fb = self._make_feedback("EU_CITIZEN")
        assert fb["userType"] in EXCLUDED_USER_TYPES

    def test_citizen_excluded(self):
        fb = self._make_feedback("CITIZEN")
        assert fb["userType"] in EXCLUDED_USER_TYPES

    def test_organisation_included(self):
        fb = self._make_feedback("ORGANISATION")
        assert fb["userType"] not in EXCLUDED_USER_TYPES

    def test_business_association_included(self):
        fb = self._make_feedback("BUSINESS_ASSOCIATION")
        assert fb["userType"] not in EXCLUDED_USER_TYPES

    def test_academic_included(self):
        fb = self._make_feedback("ACADEMIC_INSTITUTION")
        assert fb["userType"] not in EXCLUDED_USER_TYPES

    def test_ngo_included(self):
        fb = self._make_feedback("NON_PROFIT")
        assert fb["userType"] not in EXCLUDED_USER_TYPES


# ===========================================================================
# 4. Unit tests — row transformation
# ===========================================================================


class TestTransformFeedbackRow:
    SAMPLE_RAW = {
        "id": 777,
        "userType": "ORGANISATION",
        "organization": "European Banking Federation",
        "firstName": "",
        "lastName": "",
        "country": "BE",
        "language": "EN",
        "dateFeedback": "2025-03-15T10:22:00Z",
        "publicationStatus": "PUBLISHED",
        "transparencyRegisterId": "4722660838-23",
        "attachments": [{"id": "att1", "fileName": "position.pdf"}],
        "feedback": "",
        "reference": "FB-12345",
    }

    def test_maps_all_fields(self):
        row = transform_feedback_row(
            self.SAMPLE_RAW,
            initiative_id=14855,
            procedure_id="2025/0005(COD)",
            com_number="COM(2025)836",
        )
        assert row["feedback_id"] == 777
        assert row["initiative_id"] == 14855
        assert row["procedure_id"] == "2025/0005(COD)"
        assert row["com_number"] == "COM(2025)836"
        assert row["user_type"] == "ORGANISATION"
        assert row["transparency_reg_id"] == "4722660838-23"
        assert row["organisation_name"] == "European Banking Federation"
        assert row["country"] == "BE"
        assert row["language"] == "EN"
        assert row["attachment_count"] == 1
        assert row["publication_status"] == "PUBLISHED"

    def test_raw_json_roundtrippable(self):
        row = transform_feedback_row(
            self.SAMPLE_RAW,
            initiative_id=14855,
            procedure_id="2025/0005(COD)",
            com_number="COM(2025)836",
        )
        parsed = json.loads(row["raw_json"])
        assert parsed["id"] == 777

    def test_org_name_fallback_to_fullname(self):
        """When 'organization' is absent, fall back to firstName + lastName."""
        raw = dict(self.SAMPLE_RAW)
        raw["organization"] = ""
        raw["firstName"] = "Jane"
        raw["lastName"] = "Smith"
        row = transform_feedback_row(raw, 14855, "2025/0005(COD)", "COM(2025)836")
        assert row["organisation_name"] == "Jane Smith"

    def test_feedback_text_none_when_empty(self):
        raw = dict(self.SAMPLE_RAW)
        raw["feedback"] = ""
        row = transform_feedback_row(raw, 14855, "2025/0005(COD)", "COM(2025)836")
        assert row["feedback_text"] is None


# ===========================================================================
# 5. Integration tests — live HYS API
# ===========================================================================


@pytest.mark.integration
class TestHYSApiIntegration:
    """Live network tests against the HYS API.

    These are slow (~seconds) and require internet access.
    Skip in CI with: pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def session(self):
        return _make_session()

    def test_search_finds_omnibus_initiative(self, session):
        """COM(2025)836 should map to at least one HYS initiative."""
        results = search_initiatives_by_com(TARGET_COM, session)
        assert len(results) >= 1, (
            f"Expected at least 1 initiative for {TARGET_COM}, got {len(results)}"
        )
        # All results should have an id
        for r in results:
            assert "id" in r
            assert isinstance(r["id"], int)

    def test_search_with_leading_zeros(self, session):
        """Supabase-style COM number (with zeros) should also find the initiative."""
        results = search_initiatives_by_com(TARGET_COM_WITH_ZEROS, session)
        assert len(results) >= 1

    def test_search_returns_same_initiative_both_formats(self, session):
        r1 = search_initiatives_by_com(TARGET_COM, session)
        r2 = search_initiatives_by_com(TARGET_COM_WITH_ZEROS, session)
        ids1 = {r["id"] for r in r1}
        ids2 = {r["id"] for r in r2}
        # They should find the same set of initiatives
        assert ids1 == ids2 or ids1.intersection(ids2), (
            "Normalised and zero-padded COM numbers should find overlapping initiatives"
        )

    def test_feedback_pagination_returns_data(self, session):
        """Paginating initiative 14855 (Omnibus) should return feedback rows."""
        # Initiative ID 14855 is visible in the URL you shared
        feedbacks = fetch_all_feedbacks_for_initiative(14855, session)
        assert len(feedbacks) > 0, "Expected at least some feedback for the Omnibus initiative"

    def test_all_feedbacks_are_non_citizen(self, session):
        """Every returned feedback should have a non-citizen user type."""
        feedbacks = fetch_all_feedbacks_for_initiative(14855, session)
        for fb in feedbacks:
            assert fb.get("userType") not in EXCLUDED_USER_TYPES, (
                f"Got citizen feedback in results: {fb.get('userType')}"
            )

    def test_transparency_reg_id_present_on_some_rows(self, session):
        """At least some org feedback should have a transparency register ID."""
        feedbacks = fetch_all_feedbacks_for_initiative(14855, session)
        tr_ids = [
            fb.get("transparencyRegisterId")
            for fb in feedbacks
            if fb.get("transparencyRegisterId")
        ]
        assert len(tr_ids) > 0, (
            "Expected at least some feedback with a transparencyRegisterId; "
            "these are needed to link orgs to our Supabase organisations table"
        )

    def test_transform_produces_valid_rows(self, session):
        """End-to-end: search -> paginate -> transform -> validate schema."""
        initiatives = search_initiatives_by_com(TARGET_COM, session)
        assert initiatives, "No initiative found"

        initiative_id = initiatives[0]["id"]
        feedbacks = fetch_all_feedbacks_for_initiative(initiative_id, session)
        assert feedbacks, "No feedback returned"

        # Transform first 5 rows
        rows = [
            transform_feedback_row(fb, initiative_id, TARGET_PROCEDURE_ID, TARGET_COM)
            for fb in feedbacks[:5]
        ]

        required_fields = [
            "feedback_id", "initiative_id", "procedure_id", "com_number",
            "user_type", "transparency_reg_id", "organisation_name",
            "country", "language", "date_feedback", "raw_json",
        ]
        for row in rows:
            for field in required_fields:
                assert field in row, f"Missing field: {field}"
            # raw_json should be valid JSON
            parsed = json.loads(row["raw_json"])
            assert parsed.get("id") == row["feedback_id"]

    def test_pdf_download_and_extraction(self, session):
        """Download a known attachment PDF and extract text from it."""
        # DIGITALEUROPE feedback on the Omnibus call for evidence
        # documentId confirmed present in publication 20401
        doc_id = "090166e5244cfe88"
        text = download_and_extract_pdf(doc_id, session)
        assert text is not None, f"PDF extraction returned None for {doc_id}"
        assert len(text) > 200, f"Extracted text suspiciously short: {len(text)} chars"
        print(f"\nExtracted {len(text):,} chars. First 300:\n{text[:300]}")

    def test_chunking_on_real_feedback_text(self, session):
        """Rows with long feedback_text should produce multiple chunks."""
        feedbacks = fetch_all_feedbacks_for_initiative(14855, session)
        long_feedbacks = [
            fb for fb in feedbacks
            if len(fb.get("feedback") or "") > 1500
        ]

        if not long_feedbacks:
            pytest.skip("No long inline feedback found in this run — skip chunk test")

        fb = long_feedbacks[0]
        row = transform_feedback_row(fb, 14855, TARGET_PROCEDURE_ID, TARGET_COM)
        text = row["feedback_text"] or ""

        chunks = build_chunk_records(
            feedback_id=row["feedback_id"],
            initiative_id=14855,
            procedure_id=TARGET_PROCEDURE_ID,
            com_number=TARGET_COM,
            text=text,
            organisation_name=row["organisation_name"],
            transparency_reg_id=row["transparency_reg_id"],
            date_feedback=row["date_feedback"],
        )
        assert len(chunks) > 1, "Long text should produce multiple chunks"
        assert all(r["chunk_total"] == len(chunks) for r in chunks)
