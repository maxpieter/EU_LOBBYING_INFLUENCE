"""Parse EP amendment documents (A10, A9, PE reports in HTML; INTA-AM, plenary AM in DOCX).

These documents show proposed amendments in a side-by-side table format:
- Left column: Original text proposed by Commission
- Right column: Amended text proposed by Parliament

Supported formats:
1. HTML (A10/A9 reports): Opinion amendments embedded in committee reports
2. DOCX (INTA-AM-*, A-10-*-AM-*): Committee and plenary amendment lists

A10 reports may contain multiple committee opinions, each with their own amendments:
- Main report by responsible committee (e.g., INTA)
- Opinion(s) from other committees (e.g., AGRI, ENVI, etc.)

Structure example:
    OPINION OF THE COMMITTEE ON AGRICULTURE AND RURAL DEVELOPMENT
    for the Committee on International Trade

    Rapporteur for opinion: Veronika Vrecionova

    AMENDMENTS
    The Committee on Agriculture submits the following...

    Amendment 1
    Proposal for a regulation
    Recital 1
    <table>
      <tr><td>Original text...</td><td>Amended text...</td></tr>
    </table>
"""

import re
from typing import Any, Dict

from bs4 import BeautifulSoup

from .document_parser import sanitize_text


def extract_committee_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract committee opinion metadata from the document.

    Args:
        soup: BeautifulSoup object of the document

    Returns:
        Dict with:
        - responsible_committee: Main committee (e.g., "INTA")
        - main_rapporteur: Name of main rapporteur
        - opinions: List of committee opinions with metadata
    """
    metadata = {"responsible_committee": None, "main_rapporteur": None, "opinions": []}

    # Find main rapporteur and responsible committee
    # They often appear together like "Committee on International Trade\nRapporteur: Inese Vaidere"
    for p in soup.find_all("p"):
        text = p.get_text(separator="\n", strip=True)

        # Check if this paragraph contains both committee and rapporteur
        if "Rapporteur:" in text and "Committee on" in text and "opinion" not in text.lower():
            lines = text.split("\n")
            for line in lines:
                # Extract committee
                if line.startswith("Committee on"):
                    committee_match = re.search(r"Committee on (.+)", line)
                    if committee_match:
                        committee_name = committee_match.group(1).strip()
                        # Extract abbreviation if present
                        abbr_match = re.search(r"\(([A-Z]+)\)", committee_name)
                        if abbr_match:
                            metadata["responsible_committee"] = abbr_match.group(1)
                        else:
                            metadata["responsible_committee"] = committee_name

                # Extract main rapporteur
                if line.startswith("Rapporteur:"):
                    rapporteur_match = re.search(r"Rapporteur:\s*(.+)", line)
                    if rapporteur_match:
                        metadata["main_rapporteur"] = rapporteur_match.group(1).strip()

    # Find committee opinions (look for "OPINION OF THE COMMITTEE ON...")
    for heading in soup.find_all(["p", "h1", "h2", "h3"]):
        text = heading.get_text(strip=True)

        if text.startswith("OPINION OF THE COMMITTEE ON"):
            opinion = {}

            # Extract committee name and date
            committee_match = re.search(r"OPINION OF THE COMMITTEE ON (.+)", text, re.IGNORECASE)
            if committee_match:
                committee_full = committee_match.group(1).strip()

                # Extract date in parentheses if present (e.g., "(7.5.2025)")
                date_match = re.search(r"\((\d{1,2}\.\d{1,2}\.\d{4})\)", committee_full)
                if date_match:
                    opinion["date"] = date_match.group(1)
                    committee_name = re.sub(r"\s*\([^)]+\)\s*$", "", committee_full)
                else:
                    committee_name = committee_full

                opinion["committee"] = committee_name
                opinion["committee_code"] = None  # Could be extracted from procedure tables

            # Look ahead for "for the Committee on..."
            current = heading.find_next("p")
            for _ in range(5):  # Check next 5 paragraphs
                if not current:
                    break
                current_text = current.get_text(strip=True)

                # For committee
                if current_text.startswith("for the Committee on"):
                    for_committee_match = re.search(
                        r"for the Committee on (.+)", current_text, re.IGNORECASE
                    )
                    if for_committee_match:
                        opinion["for_committee"] = for_committee_match.group(1).strip()

                # Opinion rapporteur
                if "Rapporteur for" in current_text or "Rapporteur:" in current_text:
                    rapporteur_match = re.search(r"Rapporteur[^:]*:\s*(.+)", current_text)
                    if rapporteur_match:
                        opinion["rapporteur"] = rapporteur_match.group(1).strip()

                # Date adopted
                if "Date adopted" in current_text:
                    date_match = re.search(r"(\d{1,2}\.\d{1,2}\.\d{4})", current_text)
                    if date_match:
                        opinion["date_adopted"] = date_match.group(1)

                current = current.find_next_sibling("p")

            metadata["opinions"].append(opinion)

    return metadata


def parse_docx_amendments(html_content: str) -> Dict[str, Any]:
    """Parse amendments from DOCX converted to HTML (via mammoth).

    Mammoth converts DOCX to standard HTML, losing custom tags.
    Strategy: Track amendment metadata from paragraphs before each table.

    Pattern in DOCX:
    1. "Amendment X" (heading)
    2. Submitters (names, comma-separated)
    3. Document type (e.g., "Proposal for a regulation")
    4. Target (e.g., "Recital 2" or "Article 1 - paragraph 2")
    5. Table with original/amended text

    Args:
        html_content: HTML content (DOCX converted via mammoth)

    Returns:
        Dict with:
        - metadata: Committee info (if available)
        - amendments_by_committee: List of committee blocks
        - all_amendments: Flat list of all amendments
        - total_amendments: Total count
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract metadata from first paragraphs
    metadata = {}
    paras = soup.find_all("p")

    # Look for committee, rapporteur in first 20 paragraphs
    for para in paras[:20]:
        text = para.get_text(strip=True)

        if "Committee" in text and not metadata.get("committee"):
            # Try to extract committee name
            if "Committee on" in text:
                committee_match = re.search(r"Committee on ([A-Za-z\s]+)", text)
                if committee_match:
                    metadata["committee"] = committee_match.group(1).strip()

        if "Rapporteur" in text and ":" in text and not metadata.get("rapporteur"):
            metadata["rapporteur"] = text.split(":", 1)[1].strip()

        if "AMENDMENTS" in text.upper() and "-" in text:
            # Extract amendment range (e.g., "1 - 59")
            range_match = re.search(r"(\d+)\s*-\s*(\d+)", text)
            if range_match:
                metadata["amendment_range"] = f"{range_match.group(1)}-{range_match.group(2)}"

    # Extract amendments: track metadata from paragraphs, then match to tables
    # Mammoth preserves XML-like tags: <Amend>, <NumAm>, <Members>, <DocAmend>, <Article>
    amendment_metadata = []
    current_metadata = None

    for para in paras:
        text = para.get_text(strip=True)

        # Check for amendment number with XML tags
        amendment_match = re.search(r"<Amend>.*?<NumAm>(\d+)</NumAm>", text)
        if amendment_match:
            # Save previous metadata
            if current_metadata and current_metadata.get("amendment_number"):
                amendment_metadata.append(current_metadata)

            # Start new metadata block
            current_metadata = {
                "amendment_number": int(amendment_match.group(1)),
                "submitted_by": [],
                "represents": None,
                "type": None,
                "target_article": None,
                "justification": None,
            }
            continue

        if current_metadata:
            # Extract members
            members_match = re.search(r"<Members>(.*?)</Members>", text)
            if members_match:
                members_text = members_match.group(1).strip()
                # Split by comma and clean
                submitters = [name.strip() for name in members_text.split(",") if name.strip()]
                current_metadata["submitted_by"] = submitters
                continue

            # Extract "on behalf of" (represents)
            if "on behalf of" in text.lower():
                represents_match = re.search(r"on behalf of (.+)", text, re.IGNORECASE)
                if represents_match:
                    current_metadata["represents"] = represents_match.group(1).strip()
                continue

            # Extract document type
            type_match = re.search(r"<DocAmend>(.*?)</DocAmend>", text)
            if type_match:
                current_metadata["type"] = type_match.group(1).strip()
                continue

            # Extract target article/recital
            article_match = re.search(r"<Article>(.*?)</Article>", text)
            if article_match:
                target = article_match.group(1).strip()
                # Replace special characters (box character used as separator)
                target = target.replace("\u25a1", "\u2013").replace("  ", " ")
                current_metadata["target_article"] = target
                continue

            # Extract justification title
            if "Justification" in text or "<TitreJust>" in text:
                # Mark that we're in justification section
                current_metadata["_in_justification_section"] = True
                continue

            # If we're in justification section, capture the content
            if current_metadata.get("_in_justification_section"):
                # Accumulate justification text (could be multiple paragraphs)
                # Stop at: </Amend> tag, Or. language marker, or next amendment
                if "</Amend>" in text or text.startswith("Or.") or "<Amend>" in text:
                    current_metadata["_in_justification_section"] = False
                elif len(text) > 20 and "<" not in text:
                    # This is justification content
                    if current_metadata["justification"]:
                        current_metadata["justification"] += " " + text
                    else:
                        current_metadata["justification"] = text

    # Save last metadata
    if current_metadata and current_metadata.get("amendment_number"):
        amendment_metadata.append(current_metadata)

    # Extract amendments from tables and match with metadata
    all_amendments = []
    tables = soup.find_all("table")
    metadata_idx = 0

    for table in tables:
        rows = table.find_all("tr")

        # Need at least 2 rows (header + content)
        if len(rows) < 2:
            continue

        # Check last two rows for amendment pattern (header + content)
        # Look for "Text proposed" or "Amendment" in any row
        has_amendment_header = False
        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                header_text = " ".join(cell.get_text(strip=True) for cell in cells)
                if any(
                    keyword in header_text
                    for keyword in ["Text proposed", "Commission", "Amendment"]
                ):
                    has_amendment_header = True
                    break

        if not has_amendment_header:
            continue

        # Find the content row (last row with 2 cells and content)
        for row in reversed(rows):
            cells = row.find_all("td")
            if len(cells) == 2:
                cell_text = " ".join(cell.get_text(strip=True) for cell in cells)
                # Skip header rows
                if "Text proposed" in cell_text or (
                    "Amendment" in cell_text and len(cell_text) < 50
                ):
                    continue

                original_text = sanitize_text(cells[0].get_text(strip=True))
                amended_text = sanitize_text(cells[1].get_text(strip=True))

                # Add if at least one cell has content
                if original_text or amended_text:
                    # Match with metadata if available
                    amendment = {
                        "amendment_number": len(all_amendments) + 1,
                        "submitted_by": [],
                        "represents": None,
                        "type": None,
                        "target_article": None,
                        "original": original_text,
                        "amended": amended_text,
                        "justification": None,
                        "committee": metadata.get("committee", "Unknown"),
                        "rapporteur": metadata.get("rapporteur"),
                    }

                    # Try to match with metadata
                    if metadata_idx < len(amendment_metadata):
                        meta = amendment_metadata[metadata_idx]
                        amendment["amendment_number"] = meta.get(
                            "amendment_number", len(all_amendments) + 1
                        )
                        amendment["submitted_by"] = meta.get("submitted_by", [])
                        amendment["represents"] = meta.get("represents")
                        amendment["type"] = meta.get("type")
                        amendment["target_article"] = meta.get("target_article")
                        amendment["justification"] = meta.get("justification")
                        metadata_idx += 1

                    all_amendments.append(amendment)
                    break  # Only take one content row per table

    # Group by committee (single committee for DOCX)
    # committee_name = metadata.get("committee", "Unknown")  # Not used currently

    return {
        "metadata": metadata,
        "all_amendments": all_amendments,
        "total_amendments": len(all_amendments),
    }


def parse_amendment_document(html_content: str) -> Dict[str, Any]:
    """Parse amendments from DOCX-converted HTML content (mammoth output).

    All amendment documents (A10 reports, INTA-AM lists, plenary amendments)
    are now parsed as DOCX files, which mammoth converts to clean HTML with
    2-column tables and XML-like tags.

    Returns unified structure:
    - metadata: Committee and rapporteur information
    - all_amendments: List of all amendments with full metadata
    - total_amendments: Count

    Args:
        html_content: HTML content converted from DOCX by mammoth

    Returns:
        Dict with:
        - metadata: Committee/rapporteur info
        - all_amendments: List of amendments with all fields
        - total_amendments: Count
    """
    if not html_content or not html_content.strip():
        return {
            "metadata": {},
            "all_amendments": [],
            "total_amendments": 0,
        }

    # All documents are now DOCX-converted (mammoth output)
    return parse_docx_amendments(html_content)
