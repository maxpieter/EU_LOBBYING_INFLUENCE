"""Chunking strategies for legislative and SWD documents.

Creates semantically meaningful chunks respecting document structure.
"""

import re
from typing import Any, Dict, List


def chunk_legislative_document(
    html_content: str,
    structure: Dict[str, Any],
    max_chars: int = 200000,
) -> List[Dict[str, Any]]:
    """Chunk legislative document by bundling sections from parsed structure.

    NOW: Uses the structured data (with paragraphs) to build chunks.
    Groups multiple sections together until reaching max_chars (150k-200k).

    Args:
        html_content: Raw HTML content (only for explanatory memo extraction)
        structure: Parsed structure from document_parser with full article text
        max_chars: Target characters per chunk (will combine sections to reach this)

    Returns:
        List of chunks with metadata and hierarchy
    """
    from bs4 import BeautifulSoup

    chunks = []
    hierarchy = structure.get("hierarchy", {})
    all_articles = structure.get("articles", [])
    all_recitals = structure.get("recitals", [])

    # Add explanatory memorandum chunk if present (still extracted from HTML)
    if hierarchy.get("has_explanatory_memo"):
        soup = BeautifulSoup(html_content, "html.parser")
        memo_text = _extract_explanatory_memo(soup)
        if memo_text:
            chunks.append(
                {
                    "text": memo_text,
                    "type": "explanatory_memo",
                    "section": "EXPLANATORY MEMORANDUM",
                }
            )

    # Build recitals + root-level articles chunk (articles with section=None)
    root_articles = [art for art in all_articles if not art.get("section")]

    if all_recitals or root_articles:
        parts = []

        # Add recitals
        if all_recitals:
            parts.append("RECITALS")
            for recital in all_recitals:
                parts.append(f"({recital['recital_number']}) {recital['text']}")

        # Add root articles
        if root_articles:
            parts.append("\nARTICLES")
            for article in root_articles:
                article_text = f"\nArticle {article['article_number']}"
                if article.get("title"):
                    article_text += f"\n{article['title']}"
                article_text += f"\n{article.get('full_text', '')}"
                parts.append(article_text)

        combined_text = "\n\n".join(parts)
        if combined_text.strip():
            chunks.append(
                {
                    "text": combined_text,
                    "type": "legislative",
                    "article_count": len(root_articles),
                    "recital_count": len(all_recitals),
                }
            )

    # Group articles by section and build chunks
    sections_metadata = hierarchy.get("sections", [])
    chapters_metadata = hierarchy.get("chapters", [])

    current_chunk_sections = []
    current_chunk_text = []
    current_chunk_articles = 0

    for section_meta in sections_metadata:
        section_number = section_meta.get("section_number", "")
        section_title = section_meta.get("section_title", "")
        section_key = f"{section_number} {section_title}".strip()

        # Get all articles in this section
        section_articles = [art for art in all_articles if art.get("section") == section_key]

        # Build section text from flat articles
        section_text = _build_section_text_from_articles(
            section_meta, section_articles, chapters_metadata
        )

        # Add section to current chunk
        current_chunk_sections.append(section_meta)
        current_chunk_text.append(section_text)
        current_chunk_articles += len(section_articles)

        combined_text = "\n\n".join(current_chunk_text)

        # If we've reached target size or this is the last section, create chunk
        if len(combined_text) >= max_chars * 0.75 or section_meta == sections_metadata[-1]:
            chunks.append(
                {
                    "text": combined_text,
                    "type": "legislative",
                    "sections": [
                        f"{s.get('section_number', '')} {s.get('section_title', '')}".strip()
                        for s in current_chunk_sections
                    ],
                    "article_count": current_chunk_articles,
                    "hierarchy": current_chunk_sections,  # Metadata-only hierarchy
                }
            )
            # Reset for next chunk
            current_chunk_sections = []
            current_chunk_text = []
            current_chunk_articles = 0

    # Group annexes together (still extracted from HTML - annexes not fully structured)
    annexes = hierarchy.get("annexes", [])
    if annexes:
        soup = BeautifulSoup(html_content, "html.parser")
        annex_texts = []
        annex_list = []
        for annex in annexes:
            annex_text = _extract_annex_text(soup, annex)
            if annex_text:
                annex_texts.append(annex_text)
                annex_list.append(annex)

                # Create chunk when we reach target size or it's the last annex
                combined_annex_text = "\n\n".join(annex_texts)
                if len(combined_annex_text) >= max_chars * 0.75 or annex == annexes[-1]:
                    chunks.append(
                        {
                            "text": combined_annex_text,
                            "type": "annexes_group",
                            "annexes": annex_list,
                        }
                    )
                    annex_texts = []
                    annex_list = []

    # Add financial statements as a chunk if present (extracted from HTML like annexes)
    financial_statements = hierarchy.get("financial_statements", [])
    if financial_statements:
        soup = BeautifulSoup(html_content, "html.parser")

        # Find financial statement content
        fin_stmt_elem = soup.find("p", class_="Fichefinanciretitre")
        if fin_stmt_elem:
            # Extract all content after the financial statement title until signatures
            fin_content_parts = []
            current = fin_stmt_elem.find_next()

            while current:
                # Stop at signatures or next major section
                if current.name == "p":
                    classes = current.get("class", [])
                    if any(
                        cls
                        in [
                            "Fait",
                            "Institutionquisigne",
                            "Personnequisigne",
                            "Date",
                            "signature",
                            "Annexetitre",
                            "Titrearticle",
                        ]
                        for cls in classes
                    ):
                        break

                    text = current.get_text(separator=" ", strip=True)
                    if text:
                        fin_content_parts.append(text)

                elif current.name == "div" and "signature" in current.get("class", []):
                    break

                current = current.find_next()

            if fin_content_parts:
                fin_text = "\n\n".join(fin_content_parts)
                chunks.append(
                    {
                        "text": fin_text,
                        "type": "financial_statement",
                        "title": (
                            financial_statements[0]
                            if financial_statements
                            else "Legislative Financial Statement"
                        ),
                    }
                )

    # Quality check: Verify total chunk size is reasonable compared to input
    # Compare against text content (not HTML), since chunks contain extracted text
    if chunks:
        from .document_parser import extract_text_from_html

        total_chunk_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
        # Compare against extracted text, not raw HTML (HTML includes tags, cleaned markup, etc.)
        text_content = extract_text_from_html(html_content)
        text_chars = len(text_content)

        # Chunks should contain at least 40% of text (allows for signatures, TOC, headers excluded)
        if total_chunk_chars < text_chars * 0.4:
            raise ValueError(
                f"Chunking quality check failed: total chunk size ({total_chunk_chars:,} chars) "
                f"is less than 40% of source text ({text_chars:,} chars). "
                f"This suggests chunking logic is missing major content."
            )

        # Note: No logger available in this function
        # The quality check will catch major issues

    return chunks


def chunk_swd_document(
    html_content: str,
    structure: Dict[str, Any],
    max_chars: int = 200000,
) -> List[Dict[str, Any]]:
    """Chunk SWD document by table of contents structure.

    Respects Level 1 section boundaries, groups Level 2/3 subsections.

    Args:
        html_content: Raw HTML content
        structure: Parsed structure from document_parser
        max_chars: Maximum characters per chunk

    Returns:
        List of chunks with metadata
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    hierarchy = structure.get("hierarchy", [])

    if not hierarchy:
        # No TOC structure, chunk by character limit
        full_text = soup.get_text()
        return [
            {"text": full_text[i : i + max_chars], "type": "swd", "chunk_index": idx}
            for idx, i in enumerate(range(0, len(full_text), max_chars))
        ]

    chunks = []
    current_l1 = None
    current_text = []

    for i, entry in enumerate(hierarchy):
        level = entry["level"]

        if level == 1:
            # Save previous L1 section
            if current_l1 and current_text:
                chunks.append(
                    {
                        "text": "\n".join(current_text),
                        "type": "swd_section",
                        "section_title": current_l1["text"],
                        "level": 1,
                    }
                )

            # Start new L1 section
            current_l1 = entry
            current_text = [entry["text"]]
        else:
            # Add L2/L3 to current section
            if current_text:
                current_text.append(entry["text"])

    # Add last section
    if current_l1 and current_text:
        chunks.append(
            {
                "text": "\n".join(current_text),
                "type": "swd_section",
                "section_title": current_l1["text"],
                "level": 1,
            }
        )

    return chunks


def _build_section_text_from_articles(
    section_meta: Dict[str, Any],
    section_articles: List[Dict[str, Any]],
    chapters_metadata: List[Dict[str, Any]],
) -> str:
    """Build section text from flat articles list.

    Args:
        section_meta: Section metadata (section_number, section_title)
        section_articles: All articles in this section (flat list)
        chapters_metadata: Chapter metadata for organizing articles

    Returns:
        Combined text for the section
    """
    parts = []

    # Section header
    section_number = section_meta.get("section_number", "")
    section_title = section_meta.get("section_title", "")
    if section_number or section_title:
        parts.append(f"SECTION {section_number}: {section_title}".strip())

    # Get chapters for this section
    section_chapters = [ch for ch in chapters_metadata if ch.get("section") == section_number]

    if section_chapters:
        # Group articles by chapter
        for chapter in section_chapters:
            chapter_number = chapter.get("chapter_number", "")
            chapter_title = chapter.get("chapter_title", "")
            chapter_key = f"{chapter_number} {chapter_title}".strip()

            if chapter_number or chapter_title:
                parts.append(f"\nCHAPTER {chapter_number}: {chapter_title}".strip())

            # Get articles for this chapter
            chapter_articles = [
                art for art in section_articles if art.get("chapter") == chapter_key
            ]
            for article in chapter_articles:
                parts.append(_format_article_text(article))

        # Also include articles directly under section (no chapter)
        direct_articles = [art for art in section_articles if not art.get("chapter")]
        for article in direct_articles:
            parts.append(_format_article_text(article))
    else:
        # No chapters - all articles directly under section
        for article in section_articles:
            parts.append(_format_article_text(article))

    return "\n\n".join(parts)


def _format_article_text(article: Dict[str, Any]) -> str:
    """Format article with number, title, and full text."""
    article_number = article.get("article_number", "")
    article_title = article.get("title", "")
    full_text = article.get("full_text", "")

    # Build header
    if article_number and article_title:
        header = f"Article {article_number}: {article_title}"
    elif article_number:
        header = f"Article {article_number}"
    elif article_title:
        header = f"Article: {article_title}"
    else:
        header = "Article"

    # Use full_text which already includes formatted paragraphs
    return f"{header}\n{full_text}" if full_text else header


def _extract_section_text(soup, section: Dict[str, Any]) -> str:
    """Extract text content for a section.

    Args:
        soup: BeautifulSoup object
        section: Section dict with 'section_number' and 'section_title' keys
    """
    # Get section number and title
    section_number = section.get("section_number", "").strip()
    section_title = section.get("section_title", "").strip()

    # Try to find section using "TITLE {number}" (most reliable)
    if section_number:
        section_marker = f"TITLE {section_number}"
    else:
        # Fallback: construct from title
        section_marker = f"TITLE {section_title}"

    text = soup.get_text()
    start = text.find(section_marker)
    if start == -1:
        # Try alternative: just look for the number alone
        if section_number:
            start = text.find(f"TITLE {section_number}")

    if start == -1:
        return ""

    # Find next title or annex
    next_section = None
    for marker in ["TITLE ", "ANNEX "]:
        pos = text.find(marker, start + len(section_marker))
        if pos != -1 and (next_section is None or pos < next_section):
            next_section = pos

    if next_section:
        return text[start:next_section]
    return text[start:]


def _extract_chapter_text(soup, chapter: Dict[str, Any]) -> str:
    """Extract text content for a chapter.

    Args:
        soup: BeautifulSoup object
        chapter: Chapter dict with 'chapter_number' and 'chapter_title' keys
    """
    # Get chapter number and title
    chapter_number = chapter.get("chapter_number", "").strip()
    chapter_title = chapter.get("chapter_title", "").strip()

    # Use chapter number for reliable matching
    if chapter_number:
        chapter_marker = f"CHAPTER {chapter_number}"
    else:
        chapter_marker = f"CHAPTER {chapter_title}"

    text = soup.get_text()
    start = text.find(chapter_marker)
    if start == -1:
        return ""

    # Find next chapter/title or use reasonable chunk size
    next_marker = None
    for marker in ["CHAPTER ", "TITLE "]:
        pos = text.find(marker, start + len(chapter_marker))
        if pos != -1 and (next_marker is None or pos < next_marker):
            next_marker = pos

    if next_marker:
        return text[start:next_marker]
    return text[start : start + 50000]  # Reasonable default


def _extract_annex_text(soup, annex: str) -> str:
    """Extract text content for an annex.

    Args:
        soup: BeautifulSoup object
        annex: Annex string like "III Methodology for calculation"
    """
    # Extract just the number part for more reliable matching
    number_match = re.match(r"^([IVXLCDM]+|\d+)\s+", annex)
    if number_match:
        annex_number = number_match.group(1)
        annex_marker = f"ANNEX {annex_number}"
    else:
        annex_marker = f"ANNEX {annex}"

    text = soup.get_text()
    start = text.find(annex_marker)
    if start == -1:
        return ""

    # Find next annex or use end of document
    next_annex = text.find("ANNEX ", start + len(annex_marker))

    if next_annex != -1:
        return text[start:next_annex]
    return text[start:]


def _extract_explanatory_memo(soup) -> str:
    """Extract explanatory memorandum text.

    Extracts all text between the 'Exposdesmotifstitre' marker and
    the 'Formuledadoption' marker (or first TITLE if no adoption formula).

    Args:
        soup: BeautifulSoup object

    Returns:
        Text content of explanatory memorandum
    """
    # Find the start marker
    start_elem = None
    for p in soup.find_all("p"):
        if "Exposdesmotifstitre" in p.get("class", []):
            start_elem = p
            break

    if not start_elem:
        return ""

    # Find the end marker (adoption formula)
    end_elem = None
    for p in soup.find_all("p"):
        if "Formuledadoption" in p.get("class", []):
            end_elem = p
            break

    # Get all <p> elements
    all_p_elements = soup.find_all("p")

    # Find indices
    try:
        start_idx = all_p_elements.index(start_elem)
    except ValueError:
        return ""

    # Determine end index
    if end_elem:
        try:
            end_idx = all_p_elements.index(end_elem)
        except ValueError:
            # If adoption formula not found in list, use first TITLE
            end_idx = None
            for i, p in enumerate(all_p_elements[start_idx + 1 :], start=start_idx + 1):
                classes = p.get("class", [])
                if any(cls.startswith("Section") or cls.startswith("Title") for cls in classes):
                    end_idx = i
                    break
            if end_idx is None:
                end_idx = len(all_p_elements)
    else:
        # Find first TITLE/Section as end marker
        end_idx = None
        for i, p in enumerate(all_p_elements[start_idx + 1 :], start=start_idx + 1):
            classes = p.get("class", [])
            if any(cls.startswith("Section") or cls.startswith("Title") for cls in classes):
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(all_p_elements)

    # Extract text from elements in range
    text_parts = []
    for p in all_p_elements[start_idx:end_idx]:
        text = p.get_text(strip=True)
        if text:
            text_parts.append(text)

    return "\n".join(text_parts)
