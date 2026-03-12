"""Legal document parsing and segmentation for EU legislation.

This module provides structured parsing of EU legal documents (directives, regulations)
to enable intelligent comparison and analysis.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .document_parser import sanitize_text


@dataclass
class LegalSection:
    """Represents a section of a legal document."""

    section_type: str  # 'recital', 'article', 'annex', 'preamble'
    number: Optional[str]  # e.g., '1', '2(a)', 'IV'
    title: Optional[str]
    content: str

    def __str__(self):
        if self.number and self.title:
            return f"{self.section_type.title()} {self.number}: {self.title}"
        elif self.number:
            return f"{self.section_type.title()} {self.number}"
        else:
            return self.section_type.title()


@dataclass
class ParsedLegalDocument:
    """Structured representation of a legal document."""

    preamble: str
    recitals: List[LegalSection]
    articles: List[LegalSection]
    annexes: List[LegalSection]
    raw_text: str

    def get_operative_text(self) -> str:
        """Get the operative legal text (articles only)."""
        return "\n\n".join([f"{sec}\n{sec.content}" for sec in self.articles])

    def get_full_structured_text(self) -> str:
        """Get full document with structure."""
        parts = []

        if self.preamble:
            parts.append(f"PREAMBLE\n{self.preamble}")

        if self.recitals:
            parts.append("RECITALS")
            for rec in self.recitals:
                parts.append(f"{rec}\n{rec.content}")

        if self.articles:
            parts.append("ARTICLES")
            for art in self.articles:
                parts.append(f"{art}\n{art.content}")

        if self.annexes:
            parts.append("ANNEXES")
            for ann in self.annexes:
                parts.append(f"{ann}\n{ann.content}")

        return "\n\n".join(parts)


def parse_legal_document(text: str) -> ParsedLegalDocument:
    """Parse EU legal document into structured sections.

    Args:
        text: Full document text

    Returns:
        ParsedLegalDocument with segmented content
    """

    # Split into major sections
    preamble = ""
    recitals = []
    articles = []
    annexes = []

    # Extract preamble (everything before "Whereas:" or first recital)
    preamble_match = re.search(
        r"^(.*?)(?:Whereas:|Having regard to)", text, re.DOTALL | re.IGNORECASE
    )
    if preamble_match:
        preamble = preamble_match.group(1).strip()

    # Extract recitals (numbered paragraphs in "Whereas" section)
    # Use "HAVE ADOPTED" as terminator since it's more reliable than "Article 1"
    recitals_section = re.search(
        r"Whereas:(.*?)(?:HAVE ADOPTED|HAS ADOPTED)", text, re.DOTALL | re.IGNORECASE
    )
    if recitals_section:
        recitals_text = recitals_section.group(1)
        # Match patterns like "(1)" followed by content until next "(N)" or end
        # Use non-greedy match and look for "(digit)" at word boundaries
        recital_matches = re.finditer(r"\((\d+)\)\s+(.*?)(?=\n\(\d+\)|$)", recitals_text, re.DOTALL)
        for match in recital_matches:
            number = match.group(1)
            content = match.group(2).strip()
            content = sanitize_text(content)  # Sanitize U+00A0 and other artifacts
            recitals.append(LegalSection("recital", number, None, content))

    # Extract articles (only from operative part after "HAVE ADOPTED")
    # First, find where operative part starts
    operative_start = re.search(r"HAVE ADOPTED|HAS ADOPTED", text, re.IGNORECASE)
    operative_text = text[operative_start.end() :] if operative_start else text

    # Pattern: "Article 1" or "Article 1\nTitle"
    # Look for standalone "Article N" (not "Article N(" which is a reference)
    # Only match standalone ANNEX headers (at start of line, all caps, followed by Roman numerals)
    article_pattern = r"(?:^|\n)Article\s+(\d+[a-z]?)\s*\n(.*?)(?=(?:^|\n)Article\s+\d+|(?:^|\n)ANNEX\s+[IVXLCDM]+|Done at|$)"
    article_matches = re.finditer(article_pattern, operative_text, re.DOTALL | re.IGNORECASE)

    for match in article_matches:
        number = match.group(1)
        full_content = match.group(2).strip()

        # Try to extract title (first line if it's short and not all caps)
        lines = full_content.split("\n", 1)
        if len(lines) > 1 and len(lines[0]) < 100 and not lines[0].isupper():
            title = sanitize_text(lines[0].strip())
            content = sanitize_text(lines[1].strip())
        else:
            title = None
            content = sanitize_text(full_content)

        articles.append(LegalSection("article", number, title, content))

    # Extract annexes
    annex_pattern = r"Annex\s+([IVXLCDM]+|[A-Z])\s*\n?(.*?)(?=Annex\s+|$)"
    annex_matches = re.finditer(annex_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in annex_matches:
        number = match.group(1)
        content = match.group(2).strip()

        # Try to extract title
        lines = content.split("\n", 1)
        if len(lines) > 1 and len(lines[0]) < 150:
            title = sanitize_text(lines[0].strip())
            content = sanitize_text(lines[1].strip()) if len(lines) > 1 else ""
        else:
            title = None
            content = sanitize_text(content)

        annexes.append(LegalSection("annex", number, title, content))

    return ParsedLegalDocument(
        preamble=preamble, recitals=recitals, articles=articles, annexes=annexes, raw_text=text
    )


def chunk_text(
    text: str, max_chars: int = 200000, legislative_boundary: Optional[int] = None
) -> List[str]:
    """Chunk text intelligently, prioritizing structural boundaries.

    If legislative_boundary is provided (position where annexes start),
    uses that for clean separation. Otherwise tries to detect ANNEX boundary.

    Args:
        text: Full document text
        max_chars: Maximum characters per chunk
        legislative_boundary: Position where annexes start (optional)

    Returns:
        List of text chunks with legislative content prioritized
    """
    import re

    if len(text) <= max_chars:
        return [text]

    chunks = []

    # Use provided boundary if available, otherwise try to detect
    if legislative_boundary is not None:
        # Use the provided structural boundary
        legislative_content = text[:legislative_boundary].strip()
        annexes_content = text[legislative_boundary:].strip()
    else:
        # Try to identify Annex boundary from text patterns
        # Look for standalone ANNEX (not inline references like "Annex I to Regulation")
        annex_match = re.search(r"\n(ANNEX\s+[IVXLCDM]+)\s*\n(?!to\b)", text, re.IGNORECASE)

        if annex_match:
            # Split: legislative content vs annexes
            legislative_content = text[: annex_match.start()].strip()
            annexes_content = text[annex_match.start() :].strip()
        else:
            # No clear boundary found - use structural chunking
            return _chunk_by_structure(text, max_chars)

    # Chunk legislative content (prioritized)
    if legislative_content:
        legislative_chunks = _chunk_by_structure(legislative_content, max_chars)
        chunks.extend(legislative_chunks)

    # Add annexes as separate chunk(s) - lower priority
    if annexes_content:
        if len(annexes_content) <= max_chars:
            chunks.append(annexes_content)
        else:
            # Chunk large annexes separately
            annex_chunks = _chunk_by_structure(annexes_content, max_chars)
            chunks.extend(annex_chunks)

    return chunks


def _chunk_by_structure(text: str, max_chars: int) -> List[str]:
    """Internal: Chunk text by structural boundaries.

    Respects document structure by splitting at logical boundaries:
    1. Major sections (CHAPTER, TITLE, Part) - highest priority
    2. Articles - preferred split point
    3. Paragraphs - fallback

    This ensures chunks align with document structure (e.g., Article 1-10 in chunk 1,
    Article 11-20 in chunk 2) rather than arbitrary character limits.
    """
    import re

    if len(text) <= max_chars:
        return [text]

    chunks = []

    # Try to identify major structural boundaries (ordered by priority)
    # Pattern matches: CHAPTER I, TITLE II, Article 1, SECTION 1, etc.
    structural_patterns = [
        (r"\n(CHAPTER [IVXLCDM]+)", "CHAPTER"),  # CHAPTER I, CHAPTER II
        (r"\n(TITLE [IVXLCDM]+)", "TITLE"),  # TITLE I, TITLE II
        (r"\n(Article \d+)", "ARTICLE"),  # Article 1, Article 2
        (r"\n(SECTION \d+)", "SECTION"),  # SECTION 1, SECTION 2
        (r"\n(Part \d+)", "PART"),  # Part 1, Part 2
    ]

    # Find all structural boundaries with their types
    boundaries = []
    for pattern, boundary_type in structural_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            boundaries.append((match.start(), match.group(1), boundary_type))

    # Sort boundaries by position
    boundaries.sort(key=lambda x: x[0])

    # If we have structural boundaries, use them for chunking
    if boundaries:
        current_start = 0
        current_chunk_text = []

        for pos, marker, boundary_type in boundaries:
            # Get text from current start to this boundary
            segment = text[current_start:pos].strip()

            if segment:
                # Check if adding this segment would exceed max_chars
                test_chunk = "\n\n".join(current_chunk_text + [segment])
                if len(test_chunk) > max_chars and current_chunk_text:
                    # Save current chunk and start new one
                    chunks.append("\n\n".join(current_chunk_text))
                    current_chunk_text = [segment]
                else:
                    current_chunk_text.append(segment)

            current_start = pos

        # Add remaining text
        remaining = text[current_start:].strip()
        if remaining:
            test_chunk = "\n\n".join(current_chunk_text + [remaining])
            if len(test_chunk) > max_chars and current_chunk_text:
                chunks.append("\n\n".join(current_chunk_text))
                current_chunk_text = [remaining]
            else:
                current_chunk_text.append(remaining)

        if current_chunk_text:
            chunks.append("\n\n".join(current_chunk_text))

    else:
        # Fallback to paragraph-based chunking if no structure found
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > max_chars and current_chunk:
                # Save current chunk and start new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for \n\n

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

    return chunks


def compare_sections(
    initial_sections: List[LegalSection], final_sections: List[LegalSection]
) -> Dict[str, any]:
    """Compare sections between initial and final documents.

    Args:
        initial_sections: Sections from initial proposal
        final_sections: Sections from final act

    Returns:
        Dictionary with added, removed, modified sections
    """
    initial_map = {sec.number: sec for sec in initial_sections if sec.number}
    final_map = {sec.number: sec for sec in final_sections if sec.number}

    added = [num for num in final_map if num not in initial_map]
    removed = [num for num in initial_map if num not in final_map]

    modified = []
    for num in initial_map:
        if num in final_map:
            if initial_map[num].content != final_map[num].content:
                modified.append(num)

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": [num for num in initial_map if num in final_map and num not in modified],
    }
