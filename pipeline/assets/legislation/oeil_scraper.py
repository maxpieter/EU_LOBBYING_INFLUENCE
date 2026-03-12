"""OEIL HTML scraper for EU legislation procedure details.

This module scrapes procedure information directly from OEIL HTML pages,
providing a robust alternative to the V2 API which may have missing/delayed updates.

Key data extracted:
- Procedure title and reference
- Procedure type (COD, CNS, APP)
- Committee responsible and rapporteur
- Events timeline (votes, decisions, publications)
- Subjects and policy areas
- Legal basis
- Current status and stage
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup

# Constants
BASE_URL = "https://oeil.europarl.europa.eu/oeil/en/procedure-file"
DEFAULT_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.75  # Seconds between requests

# Activity type mapping: OEIL text -> V2 API controlled vocabulary
# Based on V2 API structure (def/ep-activities/*)
ACTIVITY_TYPE_MAPPING = {
    # Legislative proposal
    "legislative proposal published": "PROPOSAL_PUBLICATION",
    "legislative proposal": "PROPOSAL_PUBLICATION",
    "commission proposal": "PROPOSAL_PUBLICATION",
    # Committee activities
    "committee referral announced": "REFERRAL",
    "referral to committee": "REFERRAL",
    "committee report tabled": "COMMITTEE_REPORT",
    "vote in committee": "COMMITTEE_VOTE",
    # Plenary activities
    "debate in parliament": "PLENARY_DEBATE",
    "results of vote in parliament": "PLENARY_VOTE",
    "vote in parliament": "PLENARY_VOTE",
    "decision by parliament": "PLENARY_ADOPT_POSITION",
    # Urgency procedures
    "urgent procedure requested": "REQUEST_VOTE_URGENCY",
    "urgent procedure approved": "APPROVE_VOTE_URGENCY",
    # Council activities
    "act adopted by council": "COUNCIL_ADOPTION",
    "council agreement": "COUNCIL_AGREEMENT",
    # Final stages
    "final act signed": "SIGNATURE",
    "final act published in official journal": "PUBLICATION_OFFICIAL_JOURNAL",
    "end of procedure in parliament": "PROCEDURE_COMPLETED",
}

# Event type classification: Activity vs Decision
DECISION_KEYWORDS = [
    "decision",
    "adopt",
    "adopted",
    "vote",
    "approved",
    "rejected",
    "agreement",
]


def _classify_event_type(activity_text: str) -> str:
    """Classify event as Activity or Decision based on text.

    Args:
        activity_text: Event activity type text

    Returns:
        "Decision" or "Activity"
    """
    text_lower = activity_text.lower()

    # Check if any decision keyword is present
    if any(keyword in text_lower for keyword in DECISION_KEYWORDS):
        return "Decision"

    return "Activity"


def _map_activity_type(oeil_text: str) -> str:
    """Map OEIL event text to V2 API activity type code.

    Args:
        oeil_text: Event type text from OEIL

    Returns:
        V2 API activity type code or original text if no mapping found
    """
    text_lower = oeil_text.lower().strip()

    # Try exact match first
    if text_lower in ACTIVITY_TYPE_MAPPING:
        return ACTIVITY_TYPE_MAPPING[text_lower]

    # Try partial match (find if OEIL text contains any mapping key)
    for key, value in ACTIVITY_TYPE_MAPPING.items():
        if key in text_lower:
            return value

    # Return original text if no mapping found
    return oeil_text


def _log(msg: str, logger: Optional[Any] = None, level: str = "info"):
    """Helper to log messages."""
    if logger:
        getattr(logger, level, logger.info)(msg)
    else:
        print(f"[{level.upper()}] {msg}")


def _sanitize_text(text: str) -> str:
    """Clean and normalize text from HTML."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters that cause encoding issues
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text.strip()


def _construct_document_url(doc_id: str) -> Optional[str]:
    """Construct document URL from document ID if possible.

    Args:
        doc_id: Document ID (e.g., "COM(2025)0106", "T10-0100/2025")

    Returns:
        Direct URL to document or None if pattern not recognized

    Examples:
        COM(2025)0106 -> PDF on europarl.europa.eu
        T10-0100/2025 -> HTML on europarl.europa.eu/doceo
        A-9-2025-0173 -> HTML on europarl.europa.eu/doceo
    """
    if not doc_id:
        return None

    # COM documents (Commission proposals)
    # Pattern: COM(YYYY)NNNN
    com_match = re.match(r"COM\((\d{4})\)(\d+)", doc_id)
    if com_match:
        year = com_match.group(1)
        number = com_match.group(2).lstrip("0")  # Remove leading zeros
        # Use CELEX format (canonical per EUR-Lex robots.txt):
        # 5YYYYPCNNNN where 5=proposal, PC=Commission proposal
        celex = f"5{year}PC{number.zfill(4)}"
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

    # T documents (Texts adopted by Parliament)
    # Pattern: T10-NNNN/YYYY or T9-NNNN/YYYY
    t_match = re.match(r"T(\d+)-(\d+)/(\d{4})", doc_id)
    if t_match:
        legislature = t_match.group(1)
        number = t_match.group(2)
        year = t_match.group(3)
        # EP Doceo HTML format
        return f"https://www.europarl.europa.eu/doceo/document/TA-{legislature}-{year}-{number}_EN.html"

    # A documents (Reports)
    # Pattern: A-9-YYYY-NNNN or A9-YYYY/NNNN
    a_match = re.match(r"A-?(\d+)-(\d{4})[/-](\d+)", doc_id)
    if a_match:
        legislature = a_match.group(1)
        year = a_match.group(2)
        number = a_match.group(3)
        # EP Doceo HTML format
        return (
            f"https://www.europarl.europa.eu/doceo/document/A-{legislature}-{year}-{number}_EN.html"
        )

    # Could not construct URL from ID pattern
    return None


def get_with_retry(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    logger: Optional[Any] = None,
) -> requests.Response:
    """HTTP GET with retry logic for resilient scraping."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                _log(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s...",
                    logger,
                    "warning",
                )
                time.sleep(wait_time)
            else:
                _log(f"Request failed after {max_retries} attempts: {e}", logger, "error")
                raise


def _extract_procedure_title(soup: BeautifulSoup) -> str:
    """Extract procedure title from page."""
    # Title is in <h2 class="es_title-h2">
    h2 = soup.find("h2", class_="es_title-h2")
    if h2:
        return _sanitize_text(h2.get_text())

    # Fallback: extract from <title> tag
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text()
        # Format: "Procedure File: 2025/0058(COD) | Legislative Observatory..."
        # Extract just the title part after the reference
        if "|" in title_text:
            title_text = title_text.split("|")[0].strip()
        if ":" in title_text:
            title_text = title_text.split(":", 1)[1].strip()
        # Remove procedure reference if present
        title_text = re.sub(r"\d{4}/\d{4}\([A-Z]+\)", "", title_text).strip()
        return _sanitize_text(title_text)

    # Last fallback to h1
    h1 = soup.find("h1")
    if h1:
        return _sanitize_text(h1.get_text())

    return ""


def _extract_procedure_type(soup: BeautifulSoup, oeil_reference: str = "") -> str:
    """Extract procedure type (COD, CNS, APP, etc.).

    Args:
        soup: BeautifulSoup object
        oeil_reference: OEIL reference (e.g., "2025/0580(CNS)") - used as fallback
    """
    # Look for procedure type badge or text
    proc_type_elem = soup.find("td", string=lambda t: t and "legislative procedure" in t.lower())
    if proc_type_elem:
        text = proc_type_elem.get_text()
        # Extract COD, CNS, APP from text like "COD - Ordinary legislative procedure"
        match = re.search(r"([A-Z]{3,})\s*-", text)
        if match:
            return match.group(1)

    # Fallback: check badge
    badge = soup.find("span", class_="es_badge-procedure")
    if badge:
        text = badge.get_text()
        match = re.search(r"\(([A-Z]{3,})\)", text)
        if match:
            return match.group(1)

    # Final fallback: extract from OEIL reference itself (e.g., "2025/0580(CNS)")
    if oeil_reference:
        match = re.search(r"\(([A-Z]{3,})\)", oeil_reference)
        if match:
            return match.group(1)

    return ""


def _extract_parliament_actors(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all European Parliament actors (committees and MEPs).

    Returns list of actors with structure:
    - Committee responsible with rapporteur (or joint committee responsible)
    - Shadow rapporteurs (if any)
    - Committees for opinion (if any)
    """
    actors = []

    # Find "European Parliament" section
    parliament_section = soup.find("span", class_="t-x", string="European Parliament")
    if not parliament_section:
        return actors

    # Navigate to parent container with tables
    parent = parliament_section.find_parent("div")
    if not parent:
        return actors

    # Find all tables in Parliament section
    tables = parent.find_all("table", class_="table")

    for table in tables:
        # Check table header to determine type
        thead = table.find("thead")
        if not thead:
            continue

        header_text = _sanitize_text(thead.get_text())

        # Joint committee responsible table (takes precedence)
        if "Joint committee responsible" in header_text and "Rapporteur" in header_text:
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
            for row in rows:
                # Check if this is a shadow rapporteur row
                row_text = _sanitize_text(row.get_text())
                is_shadow_row = "Shadow rapporteur" in row_text

                # Extract committee
                committee_cell = row.find("th")
                committee_code = None
                committee_name = None
                if committee_cell:
                    committee_badge = committee_cell.find("span", class_="es_badge-committee")
                    committee_code = (
                        _sanitize_text(committee_badge.get_text()) if committee_badge else None
                    )
                    committee_link = committee_cell.find("a")
                    committee_name = (
                        _sanitize_text(committee_link.get_text()) if committee_link else None
                    )

                    if committee_code:
                        # Add as joint committee responsible
                        actors.append(
                            {
                                "actor_type": "committee",
                                "role": "joint_committee_responsible",
                                "committee_code": committee_code,
                                "committee_name": committee_name,
                            }
                        )

                # Extract MEPs from the row
                cells = row.find_all("td")
                if len(cells) >= 1:
                    mep_cell = cells[0]
                    # Find ALL MEP links in the cell (there can be multiple)
                    mep_links = mep_cell.find_all("a")
                    for mep_link in mep_links:
                        mep_name = _sanitize_text(mep_link.get_text())
                        href = mep_link.get("href", "")
                        mep_id_match = re.search(r"/meps/en/(\d+)", href)
                        mep_id = int(mep_id_match.group(1)) if mep_id_match else None

                        # Determine role based on row type
                        role = "shadow_rapporteur" if is_shadow_row else "rapporteur"

                        actors.append(
                            {
                                "actor_type": "mep",
                                "role": role,
                                "mep_id": mep_id,
                                "mep_name": mep_name,
                                "committee_code": (
                                    committee_code if not is_shadow_row else None
                                ),  # Shadow rapporteurs not tied to specific committee
                                "committee_name": committee_name if not is_shadow_row else None,
                            }
                        )

        # Committee responsible table (single committee)
        elif "Committee responsible" in header_text and "Rapporteur" in header_text:
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
            for row in rows:
                # Extract committee
                committee_cell = row.find("th")
                committee_code = None
                committee_name = None
                if committee_cell:
                    committee_badge = committee_cell.find("span", class_="es_badge-committee")
                    committee_code = (
                        _sanitize_text(committee_badge.get_text()) if committee_badge else None
                    )
                    committee_link = committee_cell.find("a")
                    committee_name = (
                        _sanitize_text(committee_link.get_text()) if committee_link else None
                    )

                    if committee_code:
                        # Add committee actor
                        actors.append(
                            {
                                "actor_type": "committee",
                                "role": "committee_responsible",
                                "committee_code": committee_code,
                                "committee_name": committee_name,
                            }
                        )

                # Extract all rapporteurs from rapporteur cell(s)
                cells = row.find_all("td")
                if len(cells) >= 1:
                    rapporteur_cell = cells[0]
                    # Find ALL rapporteur links in the cell (there can be multiple)
                    rapporteur_links = rapporteur_cell.find_all("a")
                    for rapporteur_link in rapporteur_links:
                        rapporteur_name = _sanitize_text(rapporteur_link.get_text())
                        href = rapporteur_link.get("href", "")
                        mep_id_match = re.search(r"/meps/en/(\d+)", href)
                        mep_id = int(mep_id_match.group(1)) if mep_id_match else None

                        actors.append(
                            {
                                "actor_type": "mep",
                                "role": "rapporteur",
                                "mep_id": mep_id,
                                "mep_name": rapporteur_name,
                                "committee_code": committee_code,  # Associate rapporteur with committee
                                "committee_name": committee_name,
                            }
                        )

        # Shadow rapporteurs table
        elif "Shadow rapporteur" in header_text:
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 1:
                    # First cell: Shadow rapporteur name
                    shadow_link = cells[0].find("a")
                    if shadow_link:
                        shadow_name = _sanitize_text(shadow_link.get_text())
                        href = shadow_link.get("href", "")
                        mep_id_match = re.search(r"/meps/en/(\d+)", href)
                        mep_id = int(mep_id_match.group(1)) if mep_id_match else None

                        actors.append(
                            {
                                "actor_type": "mep",
                                "role": "shadow_rapporteur",
                                "mep_id": mep_id,
                                "mep_name": shadow_name,
                            }
                        )

        # Committee for opinion table
        elif "Committee for opinion" in header_text:
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
            for row in rows:
                # Extract committee
                committee_cell = row.find("th")
                committee_code = None
                committee_name = None
                if committee_cell:
                    committee_badge = committee_cell.find("span", class_="es_badge-committee")
                    committee_code = (
                        _sanitize_text(committee_badge.get_text()) if committee_badge else None
                    )
                    committee_link = committee_cell.find("a")
                    committee_name = (
                        _sanitize_text(committee_link.get_text()) if committee_link else None
                    )

                    if committee_code:
                        actors.append(
                            {
                                "actor_type": "committee",
                                "role": "committee_for_opinion",
                                "committee_code": committee_code,
                                "committee_name": committee_name,
                            }
                        )

                # Extract opinion rapporteurs from <td> cells (if present)
                cells = row.find_all("td")
                if len(cells) >= 1:
                    rapporteur_cell = cells[0]
                    # Find ALL rapporteur links in the cell
                    rapporteur_links = rapporteur_cell.find_all("a")
                    for rapporteur_link in rapporteur_links:
                        rapporteur_name = _sanitize_text(rapporteur_link.get_text())
                        href = rapporteur_link.get("href", "")
                        mep_id_match = re.search(r"/meps/en/(\d+)", href)
                        mep_id = int(mep_id_match.group(1)) if mep_id_match else None

                        actors.append(
                            {
                                "actor_type": "mep",
                                "role": "opinion_rapporteur",
                                "mep_id": mep_id,
                                "mep_name": rapporteur_name,
                                "committee_code": committee_code,  # Associate rapporteur with opinion committee
                                "committee_name": committee_name,
                            }
                        )

    return actors


def _extract_council_actors(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract Council of the European Union actors.

    Returns list of actors with Council configurations and meeting dates.
    """
    actors = []

    # Find "Council of the European Union" accordion item
    # Look through all accordion buttons to find the one with Council text
    council_button = None
    for button in soup.find_all("button", class_="es_accordion-item-title"):
        if "Council of the European Union" in button.get_text():
            council_button = button
            break

    if not council_button:
        return actors

    # Get the accordion content div
    target_id = council_button.get("data-target", "").lstrip("#")
    if target_id:
        content_div = soup.find("div", id=target_id)
        if not content_div:
            return actors
    else:
        # Fallback: find next sibling div
        content_div = council_button.find_next_sibling("div", class_="es_accordion-item-content")
        if not content_div:
            return actors

    # Find Council configuration table within this accordion
    table = content_div.find("table", class_="table")
    if not table:
        return actors

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) >= 3:
            # First cell: Configuration name
            config_cell = cells[0]
            config_link = config_cell.find("a")
            configuration = _sanitize_text(config_link.get_text()) if config_link else None

            # Second cell: Meetings number
            meetings_cell = cells[1]
            meetings_link = meetings_cell.find("a")
            meetings_number = _sanitize_text(meetings_link.get_text()) if meetings_link else None

            # Third cell: Date
            date_cell = cells[2]
            date_text = _sanitize_text(date_cell.get_text())

            if configuration:
                actors.append(
                    {
                        "actor_type": "council",
                        "role": "council_meeting",
                        "configuration": configuration,
                        "meeting_number": meetings_number,
                        "meeting_date": date_text,
                    }
                )

    return actors


def _extract_commission_actors(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract European Commission actors (DG and Commissioner).

    Returns list of actors with Commission Policy Area and Commissioner.
    """
    actors = []

    # Find "European Commission" accordion in Key Players section
    commission_button = None
    for button in soup.find_all("button", class_="es_accordion-item-title"):
        button_text = button.get_text()
        # Find the first European Commission button (in Key Players, not Documents)
        if "European Commission" in button_text and "es_accordion-item3" in button.get(
            "data-target", ""
        ):
            commission_button = button
            break

    if not commission_button:
        return actors

    # Get the accordion content div
    target_id = commission_button.get("data-target", "").lstrip("#")
    if target_id:
        content_div = soup.find("div", id=target_id)
        if not content_div:
            return actors
    else:
        return actors

    # Find Commission table within this accordion
    table = content_div.find("table", class_="table")
    if not table:
        return actors

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            # First cell: Commission Policy Area
            dg_cell = cells[0]
            dg_link = dg_cell.find("a")
            dg_name = (
                _sanitize_text(dg_link.get_text())
                if dg_link
                else _sanitize_text(dg_cell.get_text())
            )

            # Second cell: Commissioner
            commissioner_cell = cells[1]
            commissioner_name = _sanitize_text(commissioner_cell.get_text())

            if dg_name:
                actors.append(
                    {
                        "actor_type": "commission",
                        "role": "commission_dg",
                        "institution_name": f"DG {dg_name}",
                        "commissioner_name": commissioner_name if commissioner_name else None,
                    }
                )

    return actors


def _extract_consultative_bodies(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all consultative and advisory bodies.

    Generically captures any consultative body accordion (EESC, CoR, or others).
    Returns list of actors for consultative bodies that have accordions in Key Players section.
    """
    actors = []

    # Known consultative body names (can be extended)
    consultative_keywords = [
        "European Economic and Social Committee",
        "European Committee of the Regions",
        "Committee",  # Generic fallback for any committee not already captured
    ]

    # Already captured institutions (skip these)
    skip_institutions = [
        "European Parliament",
        "Council of the European Union",
        "European Commission",
    ]

    # Find all accordion buttons in Key Players section
    for button in soup.find_all("button", class_="es_accordion-item-title"):
        button_text = _sanitize_text(button.get_text())

        # Skip already-captured institutions
        if any(skip_name in button_text for skip_name in skip_institutions):
            continue

        # Check if this looks like a consultative/advisory body
        # Either matches known consultative bodies OR contains "Committee"/"Commission" keywords
        is_consultative = False
        institution_name = button_text

        if any(keyword in button_text for keyword in consultative_keywords):
            is_consultative = True

        if is_consultative:
            actors.append(
                {
                    "actor_type": "consultative_body",
                    "role": "opinion",
                    "institution_name": institution_name,
                }
            )

    return actors


def _extract_subjects(soup: BeautifulSoup) -> List[str]:
    """Extract subject classifications and policy areas.

    Subjects are in Basic Information section, format:
    <p class="font-weight-bold mb-1">Subject</p>
    <p>3.10.04.02 Animal protection<br>3.70.01 ...</p>

    Subject codes can have 1-4 parts: "4 Economic..." or "3.70.01 Protection..." or "3.10.04.02 Animal..."
    """
    subjects = []

    # Find "Subject" label in Basic information
    subject_label = soup.find(
        "p", class_="font-weight-bold mb-1", string=lambda t: t and "Subject" in t
    )
    if subject_label:
        # Next sibling paragraph contains subjects separated by <br>
        subject_p = subject_label.find_next_sibling("p")
        if subject_p:
            # Get text and split by lines
            text = subject_p.get_text()
            # Split by newlines and extract subject codes
            # Pattern: "4 Economic..." or "3.10.04.02 Animal protection" or "3.70.01 Protection..."
            # Subject codes: 1-4 dotted numbers followed by space and description
            for line in text.split("\n"):
                line = line.strip()
                # Match: digit(s), optional dots and more digits, space, then word character
                if re.match(r"\d+(\.\d+){0,3}\s+\w", line):
                    subjects.append(_sanitize_text(line))

    return subjects  # Return all subjects (no limit)


def _extract_legal_basis(soup: BeautifulSoup) -> List[str]:
    """Extract legal basis (Treaty articles) as a list.

    Legal basis items are in separate <span class="d-block"> tags within a <td>.
    Examples:
    - "Rules of Procedure EP 59"
    - "Treaty on the Functioning of the EU TFEU 149"
    - "Treaty on the Functioning of the EU TFEU 175-p3"
    """
    legal_basis_items = []

    # Find "Legal basis" header cell
    legal_header = soup.find(string=lambda t: t and "Legal basis" in t)
    if legal_header:
        parent = legal_header.find_parent()
        if parent:
            # Find next sibling <td> containing the legal basis items
            next_elem = parent.find_next_sibling() or parent.find_next()
            if next_elem:
                # Extract all <span class="d-block"> elements
                spans = next_elem.find_all("span", class_="d-block")
                if spans:
                    # Each span contains one legal basis item
                    for span in spans:
                        text = _sanitize_text(span.get_text())
                        if text:
                            legal_basis_items.append(text)
                else:
                    # Fallback: if no spans found, try to parse text by newlines
                    text = next_elem.get_text()
                    for line in text.split("\n"):
                        line = _sanitize_text(line)
                        if line:
                            legal_basis_items.append(line)

    return legal_basis_items


def _extract_amending_acts(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract amending regulations/directives references.

    Format in HTML:
    <p>
        Amending Regulation 2023/955
        <a href="/oeil/en/procedure-file?reference=2021/0206(COD)">2021/0206(COD)</a>
        <br>
        Amending Regulation 2024/2509
        <a href="/oeil/en/procedure-file?reference=2022/0162(COD)">2022/0162(COD)</a>
    </p>

    Returns:
        List of dicts with keys: 'type' (Regulation/Directive), 'number', 'procedure_reference'
    """
    amending_acts = []
    seen = set()  # Deduplicate by (type, number)

    # Look for text starting with "Amending Regulation" or "Amending Directive"
    for amending_text in soup.find_all(string=lambda t: t and t.strip().startswith("Amending ")):
        text = _sanitize_text(amending_text)

        # Extract type (Regulation or Directive) and number
        # Pattern: "Amending Regulation 2023/955" or "Amending Directive 2011/36"
        match = re.match(r"Amending (Regulation|Directive)\s+(\S+)", text)
        if match:
            act_type = match.group(1)
            act_number = match.group(2)

            # Deduplicate
            key = (act_type, act_number)
            if key in seen:
                continue
            seen.add(key)

            # Find the associated procedure reference link (if present)
            parent = amending_text.find_parent()
            procedure_link = None
            if parent:
                link = parent.find_next("a", href=lambda h: h and "procedure-file?reference=" in h)
                if link:
                    # Extract procedure reference from link text
                    procedure_link = _sanitize_text(link.get_text())

            amending_acts.append(
                {
                    "type": act_type,
                    "number": act_number,
                    "procedure_reference": procedure_link,
                }
            )

    return amending_acts


def _extract_events(
    soup: BeautifulSoup, fetch_summaries: bool = False, logger: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Extract timeline events from procedure page with documents and summaries.

    IMPORTANT: Only extracts from the "Key events" section (id="section3") to avoid
    mixing actual legislative events with transparency meetings, national parliament
    contributions, or other non-event data.

    Event rows have 4 cells:
    - Cell 0: Date (DD/MM/YYYY)
    - Cell 1: Event type (text)
    - Cell 2: Documents (links to COM docs, TA docs, etc.)
    - Cell 3: Summary (link to document-summary page)

    Args:
        soup: BeautifulSoup object
        fetch_summaries: If True, fetch full summary text for events with summary links
        logger: Optional logger

    Returns:
        List of event dictionaries with summary_id and optionally summary_text
    """
    events = []

    # Find the "Key events" section specifically
    key_events_section = soup.find("div", id="section3")
    if not key_events_section:
        # Fallback: look for h2 with "Key events" text
        h2 = soup.find("h2", class_="es_title-h2", string=lambda t: t and "Key events" in t)
        if h2:
            key_events_section = h2.find_parent("div", class_="erpl-product-content")

    if not key_events_section:
        return events

    # Find the table within the Key events section only
    table = key_events_section.find("table", class_="table")
    if not table:
        return events

    # Extract events from this table only
    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 2:
            # First cell: date
            date_text = cells[0].get_text(strip=True)

            # Try to parse date (format: DD/MM/YYYY)
            try:
                event_date = datetime.strptime(date_text, "%d/%m/%Y").date()
            except (ValueError, AttributeError):
                continue

            # Second cell: event type
            event_type = _sanitize_text(cells[1].get_text())

            # Skip empty event types
            if not event_type:
                continue

            # Third cell: documents (if present)
            documents = []
            if len(cells) >= 3:
                doc_cell = cells[2]
                for link in doc_cell.find_all("a"):
                    href = link.get("href", "")
                    link_text = _sanitize_text(link.get_text())

                    # Extract document ID from link text or href
                    # Common patterns: COM(2025)0106, T10-0100/2025, A-9-2025-0173
                    if link_text and not href.endswith("/"):
                        # Skip EUR-Lex flag icons
                        if "flag-icon" in link.get("class", []):
                            continue

                        # Prepare document dict
                        doc_dict = {
                            "id": link_text,
                            "relationship": "based_on",  # Default - will refine based on event type
                        }

                        # Add document URL if available from HTML
                        if href:
                            # Make relative URLs absolute
                            if href.startswith("/"):
                                doc_dict["url"] = f"https://oeil.europarl.europa.eu{href}"
                            elif href.startswith("http"):
                                doc_dict["url"] = href
                        else:
                            # Try to construct URL from document ID
                            constructed_url = _construct_document_url(link_text)
                            if constructed_url:
                                doc_dict["url"] = constructed_url

                        documents.append(doc_dict)

            # Fourth cell: summary link (if present)
            summary_id = None
            summary_text = None
            if len(cells) >= 4:
                summary_cell = cells[3]
                summary_link = summary_cell.find("a", href=lambda h: h and "document-summary" in h)
                if summary_link:
                    summary_href = summary_link.get("href", "")
                    # Extract ID from /oeil/en/document-summary?id=1806245
                    if "id=" in summary_href:
                        summary_id = summary_href.split("id=")[-1]

                        # Fetch full summary text if requested
                        if fetch_summaries:
                            summary_text = _fetch_summary_text(summary_id, logger)
                            # Rate limit between summary fetches
                            if summary_text:
                                time.sleep(RATE_LIMIT_DELAY)

            # Map activity type to V2 API controlled vocabulary
            mapped_activity_type = _map_activity_type(event_type)

            # Classify as Activity or Decision
            classified_event_type = _classify_event_type(event_type)

            # Generate event_id (date + activity_type for uniqueness)
            event_id = f"{event_date.isoformat()}_{mapped_activity_type}"

            event_dict = {
                "event_id": event_id,  # Add required event_id
                "event_date": event_date.isoformat(),
                "event_type": classified_event_type,  # "Activity" or "Decision"
                "activity_type": mapped_activity_type,  # V2 API code or original text
                "activity_type_original": event_type,  # Keep original for reference
                "documents": documents,
                "summary_id": summary_id,
            }

            # Add summary_text if fetched
            if summary_text:
                event_dict["summary_text"] = summary_text

            events.append(event_dict)

    return events


def _extract_current_stage(soup: BeautifulSoup) -> str:
    """Extract current legislative stage/status."""
    # Look for status badge or stage indicator
    # Common stages: "Parliament", "Council", "Concluded"

    # Check accordion sections for active stage
    accordions = soup.find_all("button", class_="es_accordion-item-title")
    for accordion in accordions:
        # Check if expanded (aria-expanded="true")
        if accordion.get("aria-expanded") == "true":
            return _sanitize_text(accordion.get_text())

    # Fallback: check for procedure status text
    status_elem = soup.find(
        string=lambda t: t
        and any(word in t.lower() for word in ["concluded", "ongoing", "terminated"])
    )
    if status_elem:
        return _sanitize_text(status_elem)

    return ""


def _extract_status(soup: BeautifulSoup) -> str:
    """Extract procedure status (in_progress, completed, etc.).

    Status can be found in two locations:
    1. Basic information: <p class="font-weight-bold mb-1">Status</p> followed by status text
    2. Technical information: "Stage reached in procedure" row in table

    Returns raw OEIL text which will be normalized in silver_finalization.py
    """
    # Method 1: Look for "Status" label in Basic information
    status_label = soup.find(
        "p", class_="font-weight-bold mb-1", string=lambda t: t and "Status" in t
    )
    if status_label:
        # Next sibling paragraph contains the status
        status_p = status_label.find_next_sibling("p")
        if status_p:
            status_text = _sanitize_text(status_p.get_text())
            if status_text:
                return status_text  # Return raw text for silver layer to normalize

    # Method 2: Look in Technical information table
    # Find row with "Stage reached in procedure"
    for th in soup.find_all("th", scope="row"):
        if "stage reached" in th.get_text().lower():
            td = th.find_next_sibling("td")
            if td:
                status_text = _sanitize_text(td.get_text())
                if status_text:
                    return status_text  # Return raw text for silver layer to normalize

    return ""


def _extract_commission_document(soup: BeautifulSoup) -> Optional[str]:
    """Extract Commission document reference (e.g., COM(2025)0106)."""
    # Look for COM(...) pattern in links or text
    # Pattern: COM(YYYY)NNNN
    com_pattern = re.compile(r"COM\((\d{4})\)(\d{4})")

    # Search in all links first (most reliable)
    for link in soup.find_all("a"):
        href = link.get("href", "")
        text = link.get_text()

        # Check both href and link text
        for search_text in [href, text]:
            match = com_pattern.search(search_text)
            if match:
                year, number = match.groups()
                return f"COM({year}){number}"

    # Fallback: search in all text
    page_text = soup.get_text()
    match = com_pattern.search(page_text)
    if match:
        year, number = match.groups()
        return f"COM({year}){number}"

    return None


def _extract_celex_number(soup: BeautifulSoup) -> Optional[str]:
    """Extract CELEX number for final act (e.g., 32025L1237)."""
    # CELEX pattern: 3YYYYTNNNN where T is type (L=directive, R=regulation, D=decision)
    celex_pattern = re.compile(r"(3\d{4}[A-Z]\d{4})")

    # Search in EUR-Lex links
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if "eur-lex.europa.eu" in href and "CELEX" in href:
            match = celex_pattern.search(href)
            if match:
                return match.group(1)

    # Fallback: search in all text
    page_text = soup.get_text()
    match = celex_pattern.search(page_text)
    if match:
        return match.group(1)

    return None


def _extract_background_documents(soup: BeautifulSoup) -> List[Dict[str, Optional[str]]]:
    """Extract background documents (SWD, SEC, etc.) from documents table.

    Looks for a table with columns: Document type, Reference, Date, Summary
    Common document types: SWD (Staff Working Document), SEC (Commission staff document)

    Returns:
        List of dicts with keys: 'type', 'reference', 'date', 'url'
    """
    background_docs = []

    # Look for tables containing document information
    # Common header texts: "Document type", "Reference", "Date"
    for table in soup.find_all("table"):
        # Check if this table has document-related headers
        headers = table.find_all("th") or table.find_all("td", class_=["font-weight-bold"])
        header_texts = [_sanitize_text(h.get_text()) for h in headers]

        # Look for tables with "Document type" and "Reference" headers
        if any("document" in h.lower() and "type" in h.lower() for h in header_texts):
            # Found a documents table, extract rows
            for row in table.find_all("tr")[1:]:  # Skip header row
                # Document type often in <th> tag, reference in <td> tags
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    # First cell: document type (may be in th or td)
                    doc_type_cell = cells[0]
                    doc_type = _sanitize_text(doc_type_cell.get_text())

                    # Second cell: reference
                    ref_cell = cells[1]
                    reference = _sanitize_text(ref_cell.get_text())

                    # Extract date if present (usually 3rd column)
                    date = None
                    if len(cells) >= 3:
                        date_text = _sanitize_text(cells[2].get_text())
                        # Parse date if it looks like DD/MM/YYYY
                        if re.match(r"\d{2}/\d{2}/\d{4}", date_text):
                            try:
                                date_obj = datetime.strptime(date_text, "%d/%m/%Y")
                                date = date_obj.isoformat()[:10]  # YYYY-MM-DD
                            except ValueError:
                                pass

                    # Try to find EUR-Lex URL in reference cell
                    url = None
                    link = ref_cell.find("a", href=lambda h: h and "eur-lex.europa.eu" in h)
                    if link:
                        url = link.get("href")
                        if url and not url.startswith("http"):
                            url = (
                                f"https:{url}"
                                if url.startswith("//")
                                else f"https://eur-lex.europa.eu{url}"
                            )

                    # Only include SWD, SEC, and other background documents (not COM or proposals)
                    if reference and any(
                        prefix in reference for prefix in ["SWD(", "SEC(", "JOIN("]
                    ):
                        background_docs.append(
                            {
                                "type": doc_type,
                                "reference": reference,
                                "date": date,
                                "url": url,
                            }
                        )

    return background_docs


def _extract_national_parliament_documents(
    soup: BeautifulSoup,
) -> List[Dict[str, Optional[str]]]:
    """Extract national parliament contributions and reasoned opinions.

    Returns:
        List of documents with structure:
        - type: "Reasoned opinion" or "Contribution"
        - parliament_code: e.g., "DE_BUNDESRAT", "ES_CONGRESS"
        - reference: Document reference if available
        - date: Date of submission if available
    """
    national_docs = []

    # Find "National parliaments" accordion section
    nat_parl_section = soup.find("span", class_="t-x", string="National parliaments")
    if not nat_parl_section:
        return national_docs

    # Navigate to parent container
    parent = nat_parl_section.find_parent("div")
    if not parent:
        return national_docs

    # Find all tables in National parliaments section
    tables = parent.find_all("table", class_="table")

    for table in tables:
        # Check if this is a documents table
        thead = table.find("thead")
        if not thead:
            continue

        header_texts = [_sanitize_text(th.get_text()) for th in thead.find_all("th")]

        # Look for tables with document type and parliament columns
        if any("document" in h.lower() and "type" in h.lower() for h in header_texts):
            # Extract rows
            tbody = table.find("tbody")
            if not tbody:
                continue

            for row in tbody.find_all("tr"):
                # Document type often in <th> tag, other data in <td> tags
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    # First cell: document type
                    doc_type = _sanitize_text(cells[0].get_text())

                    # Skip if not reasoned opinion or contribution
                    if doc_type not in ["Reasoned opinion", "Contribution"]:
                        continue

                    # Second cell: parliament code (e.g., DE_BUNDESRAT)
                    parliament_cell = cells[1]
                    parliament_badge = parliament_cell.find("span", class_="es_badge-mep")
                    parliament_code = (
                        _sanitize_text(parliament_badge.get_text())
                        if parliament_badge
                        else _sanitize_text(parliament_cell.get_text())
                    )

                    # Third cell: reference (if available)
                    reference = None
                    if len(cells) >= 3:
                        reference = _sanitize_text(cells[2].get_text())

                    # Fourth cell: date (if available)
                    date = None
                    if len(cells) >= 4:
                        date_text = _sanitize_text(cells[3].get_text())
                        if date_text:
                            # Parse date in format DD/MM/YYYY
                            try:
                                parsed_date = datetime.strptime(date_text, "%d/%m/%Y")
                                date = parsed_date.strftime("%Y-%m-%d")
                            except ValueError:
                                date = date_text

                    national_docs.append(
                        {
                            "type": doc_type,
                            "parliament_code": parliament_code,
                            "reference": reference,
                            "date": date,
                        }
                    )

    return national_docs


def _fetch_summary_text(summary_id: str, logger: Optional[Any] = None) -> Optional[str]:
    """Fetch summary text from OEIL document-summary page.

    Args:
        summary_id: Summary ID (e.g., "1806245")
        logger: Optional logger

    Returns:
        Summary text or None if fetch fails
    """
    url = f"https://oeil.europarl.europa.eu/oeil/en/document-summary?id={summary_id}"

    try:
        _log(f"Fetching summary {summary_id}", logger, "debug")
        response = get_with_retry(url, logger=logger, timeout=DEFAULT_TIMEOUT)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the summary content div
        # Summary text is in <div class="es_product-content">
        content_div = soup.find("div", class_="es_product-content")
        if not content_div:
            return None

        # Extract all paragraph text
        paragraphs = []
        for p in content_div.find_all("p", class_="MsoNormal"):
            text = _sanitize_text(p.get_text())
            if text:
                paragraphs.append(text)

        if paragraphs:
            summary_text = "\n\n".join(paragraphs)
            _log(f"Fetched summary {summary_id}: {len(summary_text)} chars", logger, "debug")
            return summary_text

        return None

    except Exception as e:
        _log(f"Error fetching summary {summary_id}: {e}", logger, "warning")
        return None


def _extract_key_dates(events: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Extract key dates from events list.

    Returns:
        Dict with proposal_date and decision_date
    """
    proposal_date = None
    decision_date = None

    # Keywords for identifying proposal events
    proposal_keywords = ["legislative proposal published", "proposal", "commission proposal"]

    # Keywords for identifying decision events (in priority order)
    decision_keywords = [
        "final act published",  # Highest priority - actual publication
        "final act signed",  # Second - signing
        "act adopted",  # Third - adoption
        "decision by parliament",  # Fourth - parliament decision
    ]

    # Find proposal date (first matching event)
    for event in events:
        event_type = event.get("event_type", "").lower()
        if any(keyword in event_type for keyword in proposal_keywords):
            proposal_date = event.get("event_date")
            break

    # Find decision date (latest matching event, prioritize by keyword order)
    for keyword in decision_keywords:
        for event in events:
            event_type = event.get("event_type", "").lower()
            if keyword in event_type:
                decision_date = event.get("event_date")
                break
        if decision_date:
            break

    return {
        "proposal_date": proposal_date,
        "decision_date": decision_date,
    }


def scrape_oeil_procedure(
    oeil_reference: str,
    logger: Optional[Any] = None,
    use_cache: bool = False,
    fetch_summaries: bool = True,
) -> Dict[str, Any]:
    """Scrape full procedure details from OEIL HTML page.

    Args:
        oeil_reference: OEIL reference format (e.g., "2025/0058(COD)")
        logger: Optional Dagster logger
        use_cache: Whether to cache results (for future enhancement)
        fetch_summaries: If True, fetch full summary text for events with summary links (default: True)

    Returns:
        Dictionary with procedure details:
        - id: OEIL reference
        - process_id: Process ID (derived from reference, e.g., "2025-0058")
        - reference: Same as id (for consistency)
        - title: Procedure title
        - procedure_type: COD, CNS, APP, etc.
        - status: Procedure status (completed, in_progress, etc.)
        - subjects: List of subject classifications
        - policy_area: Primary policy area (first subject)
        - legal_basis: Treaty articles
        - events: List of timeline events with documents, summary_id, and optionally summary_text
        - actors: List of all actors (committees, MEPs, Council)
        - current_stage: Current legislative stage
        - proposal_date: Date of legislative proposal
        - decision_date: Date of final decision/publication
        - commission_document: Commission proposal reference (e.g., COM(2025)0106)
        - celex_number: CELEX number for final act (e.g., 32025L1237)
        - oeil_url: Source URL
        - eurlex_proposal_url: EUR-Lex URL for Commission proposal
        - eurlex_final_act_url: EUR-Lex URL for final adopted act
    """
    url = f"{BASE_URL}?reference={oeil_reference}"

    _log(f"Scraping OEIL procedure: {oeil_reference}", logger)

    try:
        # Fetch HTML with retry logic
        response = get_with_retry(url, logger=logger)
        soup = BeautifulSoup(response.content, "html.parser")

        # Derive process_id from reference (e.g., "2025/0360(COD)" -> "2025-0360")
        process_id = None
        if "/" in oeil_reference and "(" in oeil_reference:
            ref_parts = oeil_reference.split("(")[0]  # "2025/0360"
            process_id = ref_parts.replace("/", "-")  # "2025-0360"

        # Extract all data
        procedure_data = {
            "id": oeil_reference,
            "reference": oeil_reference,  # Add reference field for consistency
            "process_id": process_id,  # Add process_id (required by Pydantic model)
            "title": _extract_procedure_title(soup),
            "procedure_type": _extract_procedure_type(soup, oeil_reference),
            "oeil_url": url,
        }

        # Subjects and policy areas
        procedure_data["subjects"] = _extract_subjects(soup)
        procedure_data["policy_area"] = (
            procedure_data["subjects"][0] if procedure_data["subjects"] else None
        )

        # Legal basis
        procedure_data["legal_basis"] = _extract_legal_basis(soup)

        # Amending acts (regulations/directives being amended)
        procedure_data["amending_acts"] = _extract_amending_acts(soup)

        # Background documents (SWD, SEC, impact assessments)
        procedure_data["background_documents"] = _extract_background_documents(soup)

        # National parliament contributions (stored in dedicated field)
        procedure_data["national_parliament_documents"] = _extract_national_parliament_documents(
            soup
        )

        # Events timeline (with documents and summaries)
        procedure_data["events"] = _extract_events(
            soup, fetch_summaries=fetch_summaries, logger=logger
        )

        # Add national parliament documents as events ONLY if they have dates
        # (Matches V2 API philosophy: all events must have dates)
        for nat_doc in procedure_data["national_parliament_documents"]:
            if nat_doc["date"]:  # Only create event if date exists
                event_type = (
                    "REASONED_OPINION"
                    if nat_doc["type"] == "Reasoned opinion"
                    else "NATIONAL_PARLIAMENT_CONTRIBUTION"
                )
                event_id = f"{nat_doc['date']}_{event_type}_{nat_doc['parliament_code']}"
                event = {
                    "event_id": event_id,  # Add required event_id
                    "event_date": nat_doc["date"],  # Use event_date field
                    "event_type": "Activity",  # These are activities
                    "activity_type": event_type,
                    "activity_label": f"{nat_doc['type']} - {nat_doc['parliament_code']}",
                    "body": "National Parliament",
                    "parliament_code": nat_doc["parliament_code"],
                    "documents": [],  # National parliament events don't have documents array
                    "summary_id": None,
                }
                if nat_doc["reference"]:
                    event["reference"] = nat_doc["reference"]

                procedure_data["events"].append(event)

        # Actors (Parliament, Council, Commission, consultative bodies)
        parliament_actors = _extract_parliament_actors(soup)
        council_actors = _extract_council_actors(soup)
        commission_actors = _extract_commission_actors(soup)
        consultative_actors = _extract_consultative_bodies(soup)
        procedure_data["actors"] = (
            parliament_actors + council_actors + commission_actors + consultative_actors
        )

        # Status and stage
        procedure_data["status"] = _extract_status(soup)
        procedure_data["current_stage"] = _extract_current_stage(soup)

        # Extract key dates from events
        key_dates = _extract_key_dates(procedure_data["events"])
        procedure_data.update(key_dates)

        # Extract Commission document and CELEX references
        procedure_data["commission_document"] = _extract_commission_document(soup)
        procedure_data["celex_number"] = _extract_celex_number(soup)

        # Generate EUR-Lex URLs (HTML format for better text extraction)
        if procedure_data["commission_document"]:
            # Extract year and number from COM(YYYY)NNNN
            com_match = re.search(r"COM\((\d{4})\)(\d+)", procedure_data["commission_document"])
            if com_match:
                year = com_match.group(1)
                number = com_match.group(2).lstrip("0")  # Remove leading zeros
                # CELEX format for COM proposals: 5YYYYPCNNNN
                celex = f"5{year}PC{number.zfill(4)}"
                procedure_data["eurlex_proposal_url"] = (
                    f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
                )

        if procedure_data["celex_number"]:
            # Use HTML format for cleaner text extraction
            procedure_data["eurlex_final_act_url"] = (
                f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{procedure_data['celex_number']}"
            )

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        return procedure_data

    except Exception as e:
        _log(f"Error scraping {oeil_reference}: {e}", logger, "error")
        # Return minimal data on error
        return {
            "id": oeil_reference,
            "oeil_url": url,
            "error": str(e),
        }


def scrape_multiple_procedures(
    oeil_references: List[str],
    logger: Optional[Any] = None,
    fetch_summaries: bool = True,
) -> List[Dict[str, Any]]:
    """Scrape multiple procedures with progress logging.

    Args:
        oeil_references: List of OEIL references to scrape
        logger: Optional Dagster logger
        fetch_summaries: If True, fetch full summary text for events (default: True)

    Returns:
        List of procedure dictionaries
    """
    procedures = []
    total = len(oeil_references)

    for i, ref in enumerate(oeil_references, 1):
        if i % 10 == 0:
            _log(f"Progress: {i}/{total} procedures scraped", logger)

        proc_data = scrape_oeil_procedure(ref, logger=logger, fetch_summaries=fetch_summaries)
        procedures.append(proc_data)

    _log(f"Completed scraping {total} procedures", logger)
    return procedures
