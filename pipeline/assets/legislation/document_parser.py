"""Lean HTML parser for EUR-Lex legislative documents.

Parse EUR-Lex HTML by iterating <p> elements and matching CSS classes.
No complex tree traversal - just direct iteration.
"""

import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup


def sanitize_text(text: str) -> str:
    """Normalize text by removing artifacts from EUR-Lex HTML.

    Handles:
    - U+00A0 (non-breaking space) → regular space
    - Tracked change marker characters (⇨, ⇦, ⌦, ⌫, etc.)
    - Zero-width spaces and joiners
    - Various typographic spaces

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text with artifacts replaced
    """
    # Tracked change markers (keep in sync with parse_legislative_structure)
    marker_chars = [
        "\u21e8",  # ⇨ rightwards white arrow
        "\u21e6",  # ⇦ leftwards white arrow
        "\u2326",  # ⌦ erase to the right
        "\u232b",  # ⌫ erase to the left
        "\u1f87b",  # 🡻 downwards black arrow
        "\u21e9",  # ⇩ downwards white arrow
        "\u21eb",  # ⇧ upwards white arrow
    ]
    for marker in marker_chars:
        text = text.replace(marker, "")

    # Non-breaking space to regular space
    text = text.replace("\u00a0", " ")

    # Zero-width characters (invisible but can cause issues)
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200d", "")  # Zero-width joiner
    text = text.replace("\ufeff", "")  # Byte order mark

    # Various typographic spaces to regular space
    text = text.replace("\u2002", " ")  # En space
    text = text.replace("\u2003", " ")  # Em space
    text = text.replace("\u2009", " ")  # Thin space

    return text


def extract_text_from_html(html_content: str) -> str:
    """Extract text content from HTML for comparison purposes.

    Removes tracked changes markup and extracts text, mimicking what
    parse_legislative_structure does before chunking.

    Args:
        html_content: Raw EUR-Lex HTML

    Returns:
        Extracted text content (no HTML tags, cleaned of CR markup)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove tracked changes markup (same as parse_legislative_structure)
    removal_classes = ["CRDeleted", "CRRefonteDeleted", "CRMinorChangeDeleted", "CRMarker"]
    for cls in removal_classes:
        for elem in soup.find_all(class_=cls):
            elem.decompose()

    # Extract text content
    text = soup.get_text(separator=" ", strip=True)

    # Clean up Unicode markers and normalize whitespace (same as parse_legislative_structure)
    marker_chars = ["\u21e8", "\u21e6", "\u2326", "\u232b", "\u1f87b", "\u21e9", "\u21eb"]
    for marker in marker_chars:
        text = text.replace(marker, "")

    # Normalize non-breaking spaces to regular spaces
    text = text.replace("\u00a0", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_legislative_structure(html_content: str) -> Dict[str, Any]:
    """Parse legislative document structure from EUR-Lex HTML.

    Iterates directly through <p> elements and extracts structure based on CSS classes.

    Args:
        html_content: Raw EUR-Lex HTML

    Returns:
        Dict with 'elements' (flat list), 'hierarchy' (nested structure), and 'recitals'/'articles' (structured)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove tracked changes markup (deleted content and visual markers)
    # EUR-Lex uses multiple classes for tracked changes in recast/amended documents:
    # - CRDeleted: Standard deleted text (old directives, replaced text)
    # - CRRefonteDeleted: Recast deleted text (consolidated versions)
    # - CRMinorChangeDeleted: Minor deleted text (typos, formatting changes)
    # - CRMarker: Visual change markers (⌦, ⌫, ⇨, ⇦, 🡻, etc.) - not actual content
    # Keep: CRMinorChangeAdded (insertions are kept), CRReference (citations are kept)
    removal_classes = ["CRDeleted", "CRRefonteDeleted", "CRMinorChangeDeleted", "CRMarker"]
    for cls in removal_classes:
        for elem in soup.find_all(class_=cls):
            elem.decompose()

    # Clean up remaining Unicode marker characters that aren't wrapped in spans
    # These visual markers for tracked changes should not be in the final text
    marker_chars = [
        "\u21e8",  # ⇨ rightwards white arrow
        "\u21e6",  # ⇦ leftwards white arrow
        "\u2326",  # ⌦ erase to the right
        "\u232b",  # ⌫ erase to the left
        "\u1f87b",  # 🡻 downwards black arrow
        "\u21e9",  # ⇩ downwards white arrow
        "\u21eb",  # ⇧ upwards white arrow
    ]

    # Replace marker characters in all text nodes
    for text_node in soup.find_all(text=True):
        cleaned_text = text_node
        for marker in marker_chars:
            cleaned_text = cleaned_text.replace(marker, "")
        # Also normalize multiple spaces and non-breaking spaces
        cleaned_text = cleaned_text.replace("\u00a0", " ")  # nbsp to regular space
        if cleaned_text != text_node:
            text_node.replace_with(cleaned_text)

    elements = []
    recitals = []

    # Track boundary markers for explanatory memorandum
    has_explanatory_memo = False
    # explanatory_memo_elem = None  # Not used
    # adoption_formula_elem = None  # Not used

    # Check for Official Journal format (oj-* CSS classes)
    # Official Journal recitals are in <div class="eli-subdivision" id="rct_N"> with table structure
    oj_recital_divs = soup.find_all("div", class_="eli-subdivision", id=re.compile(r"^rct_\d+"))
    if oj_recital_divs:
        # Official Journal format detected - extract recitals from table structure
        for div in oj_recital_divs:
            table = div.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) == 2:
                        # First cell: recital number
                        num_text = cells[0].get_text(strip=True)
                        match = re.match(r"\((\d+)\)", num_text)
                        if match:
                            recital_number = int(match.group(1))
                            # Second cell: recital text
                            recital_text = sanitize_text(
                                cells[1].get_text(separator=" ", strip=True)
                            )
                            if recital_text:
                                recitals.append(
                                    {
                                        "recital_number": recital_number,
                                        "text": recital_text,
                                        "order_index": recital_number,
                                    }
                                )

    # Direct iteration through all <p> elements
    all_paragraphs = soup.find_all("p")
    i = 0
    while i < len(all_paragraphs):
        p = all_paragraphs[i]
        classes = p.get("class", [])
        if not classes:
            i += 1
            continue

        entry = None

        # Explanatory Memorandum marker
        if "Exposdesmotifstitre" in classes:
            has_explanatory_memo = True
            # explanatory_memo_elem = p  # Not used currently
            i += 1
            continue

        # Adoption formula marker (end of preamble)
        if "Formuledadoption" in classes:
            # adoption_formula_elem = p  # Not used currently
            i += 1
            continue

        # Recital (ManualConsidrant, Considerant, or Considrant class in EUR-Lex HTML)
        # Note: HTML has class="li ManualConsidrant" (two classes), not class="li_ManualConsidrant"
        if any(
            cls in ["ManualConsidrant", "Considerant", "Considrant"]
            or cls.startswith("Considerant")
            for cls in classes
        ):
            recital_entry = _extract_recital(p, len(recitals) + 1)
            if recital_entry:
                recitals.append(recital_entry)

        # Article: Titrearticle or Titlearticle (EUR-Lex typo)
        # OR Official Journal format: oj-ti-art
        if "Titrearticle" in classes or "Titlearticle" in classes or "oj-ti-art" in classes:
            entry = _extract_article(p)

        # Section/Title/Chapter: Check content to distinguish
        # EUR-Lex uses SectionTitle for both sections and chapters
        elif any(cls.startswith(("Section", "Title", "Chapter")) for cls in classes):
            # Check text content to determine if it's a chapter or section
            text = p.get_text(separator=" ", strip=True)
            if "CHAPTER" in text.upper():
                entry = _extract_chapter(p)

                # Handle split chapter (number in one <p>, title in next <p>)
                if entry and not entry.get("chapter_title"):
                    # Look ahead for title in next SectionTitle element
                    if i + 1 < len(all_paragraphs):
                        next_p = all_paragraphs[i + 1]
                        next_classes = next_p.get("class", [])
                        if any(cls.startswith(("Section", "Title")) for cls in next_classes):
                            next_text = next_p.get_text(separator=" ", strip=True)
                            # If next element doesn't contain CHAPTER, use it as title
                            if "CHAPTER" not in next_text.upper():
                                entry["chapter_title"] = next_text
                                i += 1  # Skip next element since we consumed it
            else:
                entry = _extract_section(p)

        # Annex: Any class starting with "Annex"
        elif any(cls.startswith("Annex") for cls in classes) or (
            "oj-doc-ti" in classes
            and re.match(r"ANNEX", p.get_text(strip=True), re.IGNORECASE)
            and p.find_parent("div", class_="eli-container")
            and p.find_parent("div", class_="eli-container").get("id", "").startswith("anx_")
        ):
            entry = _extract_annex(p)

        # Financial statement (Legislative financial statement)
        elif "Fichefinanciretitre" in classes:
            entry = {
                "type": "financial_statement",
                "title": p.get_text(separator=" ", strip=True),
            }

        if entry:
            elements.append(entry)

        i += 1

    # Build flat articles list and metadata-only hierarchy in single pass
    hierarchy_result = _build_hierarchy_and_articles(elements)
    hierarchy = hierarchy_result["hierarchy"]
    structured_articles = hierarchy_result["articles"]

    # Add explanatory memorandum metadata (just flag, chunking will find elements)
    hierarchy["has_explanatory_memo"] = has_explanatory_memo

    return {
        "hierarchy": hierarchy,
        "total_articles": hierarchy["total_articles"],
        "total_sections": hierarchy["total_sections"],
        "total_chapters": hierarchy["total_chapters"],
        "total_annexes": hierarchy["total_annexes"],
        "has_explanatory_memo": has_explanatory_memo,
        "recitals": recitals,
        "articles": structured_articles,
    }


def _extract_recital(elem, order_index: int) -> Optional[Dict[str, Any]]:
    """Extract recital from <p class="ManualConsidrant">, <p class="Considrant">, or <p class="Considerant*"> element.

    EUR-Lex structure:
    <p class="li ManualConsidrant">  (Note: two separate classes)
      <span class="num"><span>(1)</span></span>
      <span>Text content...</span>
    </p>

    Or:
    <p class="li Considrant">  (Note: two separate classes)
      <span><span class="num">(1)</span></span>
      <span>Text content...</span>
    </p>

    Returns:
        Dict with recital_number, text, and order_index
    """
    # Look for span with class="num" containing the recital number
    num_span = elem.find("span", class_="num")
    if num_span:
        # Extract number from nested structure
        num_text = num_span.get_text(strip=True)
        match = re.match(r"\((\d+)\)", num_text)
        if match:
            recital_number = int(match.group(1))

            # Get text from all spans except the num span
            text_parts = []
            for span in elem.find_all("span"):
                if "num" not in span.get("class", []):
                    text_parts.append(span.get_text())

            recital_text = "".join(text_parts).strip()

            # Clean up: remove any remaining (XX) prefix pattern
            # Sometimes the number appears in text even after filtering spans
            recital_text = re.sub(r"^\(\d+\)\s*", "", recital_text)
            recital_text = sanitize_text(recital_text)

            if recital_text:
                return {
                    "recital_number": recital_number,
                    "text": recital_text,
                    "order_index": order_index,
                }

    # Fallback: try text-based extraction for old format
    text = " ".join(elem.get_text().split())
    text = sanitize_text(text)
    if not text:
        return None

    match = re.match(r"^\((\d+)\)\s*(.*)", text, re.DOTALL)
    if match:
        recital_number = int(match.group(1))
        recital_text = match.group(2).strip()

        return {
            "recital_number": recital_number,
            "text": recital_text,
            "order_index": order_index,
        }

    return None


def _extract_article(elem) -> Dict[str, Any]:
    """Extract article from <p class="Titrearticle"> and following content with paragraphs.

    Handles both standard EUR-Lex format and Official Journal format (oj-ti-art).
    Removes <span> and <br> tags to get clean text: "Article 21 Preparation and submission"
    Also extracts article content from following paragraphs, parsing numbered paragraphs.
    """
    # Remove span and br tags to get clean title text
    # Replace <br> with space to avoid "Article 1Subject" issue
    title_elem = elem
    for br in title_elem.find_all("br"):
        br.replace_with(" ")
    for span in title_elem.find_all("span"):
        span.unwrap()

    # Get text and normalize whitespace (don't use separator to avoid "follow ing:")
    title_text = " ".join(title_elem.get_text().split())
    title_text = sanitize_text(title_text)

    # Skip quoted/referenced articles (appear in amendment text)
    # These have leading quotes and are not part of main document structure
    if title_text and title_text[0] in "'\"\u2018\u2019\u201c\u201d":
        return None  # Skip this article

    # Remove any remaining leading/trailing quotes
    title_text = title_text.strip("'\"\u2018\u2019\u201c\u201d")

    # Check if this is Official Journal format - article title in eli-subdivision div
    classes = elem.get("class", [])
    if "oj-ti-art" in classes:
        # Official Journal format: look for parent eli-subdivision div with id="art_N"
        parent_div = elem.find_parent("div", class_="eli-subdivision")
        if parent_div and parent_div.get("id", "").startswith("art_"):
            # Extract all oj-normal paragraphs within this div
            # Include both direct children and nested subdivisions
            content_parts = []

            # First, get direct child paragraphs (for articles like Article 4 without nested divs)
            for p in parent_div.find_all("p", class_="oj-normal", recursive=False):
                content_text = " ".join(p.get_text().split())
                content_text = sanitize_text(content_text)
                if content_text:
                    content_parts.append(content_text)

            # Then, get paragraphs from nested subdivisions (for articles with numbered paragraphs)
            # Look for any nested divs (not just eli-subdivision) that contain oj-normal paragraphs
            for subdiv in parent_div.find_all("div", recursive=True):
                for p in subdiv.find_all("p", class_="oj-normal"):
                    content_text = " ".join(p.get_text().split())
                    content_text = sanitize_text(content_text)
                    if content_text:
                        content_parts.append(content_text)
        else:
            content_parts = []
    else:
        # Standard EUR-Lex format: extract content from following elements (across div boundaries)
        content_parts = []
        current = elem.find_next()  # Use find_next() to traverse entire tree, not just siblings

        # Collect content until we hit another structural element
        while current:
            # Check if this is a structural element (article/section/chapter/annex)
            if current.name == "p":
                classes = current.get("class", [])

                # Stop at next article (both spelling variants + Official Journal format)
                if "Titrearticle" in classes or "Titlearticle" in classes or "oj-ti-art" in classes:
                    break

                # Stop at section/title/chapter (any class starting with these)
                if any(cls.startswith(("Section", "Title", "Chapter", "Annex")) for cls in classes):
                    break

                # Stop at document closure sections (signatures, financial statements, etc.)
                # These mark the end of the legislative text
                if any(
                    cls
                    in [
                        "Fait",
                        "Institutionquisigne",
                        "Personnequisigne",
                        "Date",
                        "signature",
                    ]
                    for cls in classes
                ):
                    break

                # Collect content from paragraph (but skip if it's just whitespace)
                content_text = " ".join(current.get_text().split())
                content_text = sanitize_text(content_text)
                if content_text:
                    content_parts.append(content_text)

            # Also stop if we hit a div with class 'signature' (signature block)
            elif current.name == "div" and "signature" in current.get("class", []):
                break

            current = current.find_next()

    # Parse numbered paragraphs (e.g., "1. Text", "2. Text")
    # Use \n between lines to preserve sub-elements (a), (b), etc.
    paragraphs = []

    for content in content_parts:
        # Try to match numbered paragraph: "1. Text" or "1.Text"
        para_match = re.match(r"^(\d+)\.\s*(.*)", content, re.DOTALL)
        if para_match:
            para_num = int(para_match.group(1))
            para_text = para_match.group(2).strip()
            paragraphs.append({"paragraph_number": para_num, "text": para_text})
        else:
            # If no number, add to last paragraph or as unnumbered
            if paragraphs:
                # Append to last paragraph with single newline
                paragraphs[-1]["text"] += "\n" + content
            else:
                # First paragraph without number - assume it's paragraph 1
                paragraphs.append({"paragraph_number": 1, "text": content})

    # Extract "Article 123 Title text" or "Article 27a Title text"
    # Use case-insensitive for "Article" keyword only, strict lowercase for suffix
    match = re.match(r"(?i:Article)\s*(\d+[a-z]?)\s*(.*)", title_text)
    if match:
        article_number_str = match.group(1)
        # Try to parse as int, but keep as string if it has letter suffix
        try:
            article_number = int(article_number_str)
        except ValueError:
            article_number = article_number_str
        article_title = match.group(2).strip()
    else:
        article_number = None
        article_title = title_text

    return {
        "type": "article",
        "article_number": article_number,
        "article_title": article_title,
        "paragraphs": paragraphs,
    }


def _get_clean_text(elem) -> str:
    """Get clean text from element, treating <br> as space separator.

    Does NOT modify the original element.
    Handles split words across spans (e.g., <span>F</span><span>inal</span> -> "Final").
    """
    from bs4 import Comment, NavigableString

    result = []
    for child in elem.descendants:
        if child.name == "br":
            # <br> becomes a space separator
            result.append(" ")
        elif isinstance(child, NavigableString) and not isinstance(child, Comment):
            result.append(str(child))

    # Join all parts and normalize whitespace
    text = " ".join("".join(result).split())
    return sanitize_text(text)


def _extract_section(elem) -> Dict[str, Any]:
    """Extract section/title from <p class="SectionTitle">.

    Handles patterns like:
    - <span>TITLE </span><span>VII</span><span><br/>GOVERNANCE OF THE PLAN</span>
    - <span>TITLE III</span><span>NAME</span>
    """
    text = _get_clean_text(elem)

    # Extract "TITLE VII GOVERNANCE OF THE PLAN"
    match = re.match(
        r"TITLE\s*(\d+|M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\s*(.*)",
        text,
        re.IGNORECASE,
    )
    if match:
        number = match.group(1)
        name = match.group(2).strip()
        return {
            "type": "section",
            "section_number": number,
            "section_title": name,
        }

    return None


def _extract_chapter(elem) -> Dict[str, Any]:
    """Extract chapter from <p class="ChapterTitle*"> or <p class="SectionTitle"> with CHAPTER text.

    Handles patterns like:
    - <span>CHAPTER 1</span><span><br/>Plan authorities</span>
    - <span>CHAPTER II</span><span><br/>Rules on payments</span>
    """
    text = _get_clean_text(elem)

    # Extract "CHAPTER I Name" or "CHAPTER 3 Name"
    match = re.match(
        r"CHAPTER\s*(\d+|M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\s*(.*)",
        text,
        re.IGNORECASE,
    )
    if match:
        number = match.group(1)
        name = match.group(2).strip()
        return {
            "type": "chapter",
            "chapter_number": number,
            "chapter_title": name,
        }

    return None


def _extract_annex(elem) -> Dict[str, Any]:
    """Extract annex from <p class="Annexetitre">.

    Removes <span> and <br> tags to get clean text: "ANNEX III Methodology for calculation"
    """
    # Remove span and br tags
    for tag in elem.find_all(["span", "br"]):
        tag.unwrap() if tag.name == "span" else tag.replace_with(" ")

    text = " ".join(elem.get_text().split())

    # Extract "ANNEX III Name"
    match = re.match(
        r"ANNEX\s*(\d+|M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\s*(.*)",
        text,
        re.IGNORECASE,
    )
    if match:
        number = match.group(1)
        name = match.group(2).strip()
        return {
            "type": "annex",
            "annex": f"{number} {name}" if name else number,
        }

    return None


def _build_hierarchy_and_articles(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build metadata-only hierarchy and flat articles list in single pass.

    Returns:
        Dict with:
        - hierarchy: Metadata-only structure (sections/chapters without nested articles)
        - articles: Flat list with section/chapter context as metadata
    """
    hierarchy = {
        "total_articles": 0,
        "total_sections": 0,
        "total_chapters": 0,
        "total_annexes": 0,
        "sections": [],
        "chapters": [],
        "annexes": [],
        "financial_statements": [],
    }

    articles = []
    current_section = None
    current_chapter = None

    for elem in elements:
        elem_type = elem["type"]

        if elem_type == "section":
            # Add section metadata only (no nested articles)
            current_section = {
                "section_number": elem.get("section_number", ""),
                "section_title": elem.get("section_title", ""),
            }
            hierarchy["sections"].append(current_section)
            hierarchy["total_sections"] += 1
            current_chapter = None

        elif elem_type == "chapter":
            # Add chapter metadata with section reference
            current_chapter = {
                "chapter_number": elem.get("chapter_number", ""),
                "chapter_title": elem.get("chapter_title", ""),
                "section": current_section.get("section_number") if current_section else None,
            }
            hierarchy["chapters"].append(current_chapter)
            hierarchy["total_chapters"] += 1

        elif elem_type == "article":
            # Build full_text from paragraphs
            paragraphs = elem.get("paragraphs", [])
            if paragraphs:
                full_text = "\n\n".join(
                    (
                        f"{p['paragraph_number']}. {p['text']}"
                        if "paragraph_number" in p
                        else p["text"]
                    )
                    for p in paragraphs
                )
            else:
                full_text = ""

            # Add to flat articles list with section/chapter metadata
            section_number = current_section.get("section_number") if current_section else None
            section_title = current_section.get("section_title") if current_section else None
            chapter_number = current_chapter.get("chapter_number") if current_chapter else None
            chapter_title = current_chapter.get("chapter_title") if current_chapter else None

            article_obj = {
                "article_number": elem.get("article_number"),
                "title": elem.get("article_title", ""),
                "full_text": full_text,
                "section": f"{section_number} {section_title}".strip() if section_number else None,
                "chapter": f"{chapter_number} {chapter_title}".strip() if chapter_number else None,
            }
            articles.append(article_obj)
            hierarchy["total_articles"] += 1

        elif elem_type == "annex":
            hierarchy["annexes"].append(elem["annex"])
            hierarchy["total_annexes"] += 1

        elif elem_type == "financial_statement":
            hierarchy["financial_statements"].append(
                elem.get("title", "Legislative Financial Statement")
            )

    return {"hierarchy": hierarchy, "articles": articles}


def parse_swd_structure(html_content: str) -> Dict[str, Any]:
    """Parse SWD (Staff Working Document) table of contents structure.

    SWD documents use TOC + heading anchors for navigation.

    Args:
        html_content: Raw EUR-Lex HTML

    Returns:
        Dict with 'hierarchy' (TOC structure) and heading positions
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Check if this is an SWD document
    if not soup.find("p", class_="TOCHeading"):
        return {"hierarchy": [], "is_swd": False}

    # Extract TOC entries with their levels
    toc_entries = []
    for p in soup.find_all("p"):
        classes = p.get("class", [])
        if not classes:
            continue

        # Look for TOC1, TOC2, TOC3, li TOC1, li TOC2, etc.
        level = None
        for cls in classes:
            if "TOC1" in cls:
                level = 1
            elif "TOC2" in cls:
                level = 2
            elif "TOC3" in cls:
                level = 3

        if level:
            # Extract heading text and anchor link
            link = p.find("a", class_="Hyperlink")
            if link:
                text = " ".join(link.get_text().split())
                href = link.get("href", "")
                anchor = href.lstrip("#") if href.startswith("#") else None

                toc_entries.append(
                    {
                        "level": level,
                        "text": text,
                        "anchor": anchor,
                    }
                )

    # Find corresponding heading positions in document
    for entry in toc_entries:
        if entry["anchor"]:
            heading = soup.find(id=entry["anchor"])
            if heading:
                entry["heading_position"] = len(str(heading.find_all_previous()))

    return {
        "hierarchy": toc_entries,
        "is_swd": True,
        "total_sections": sum(1 for e in toc_entries if e["level"] == 1),
    }
