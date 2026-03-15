"""Bronze layer: Scrape EU Commission meetings from EC Transparency Initiative.

Flow:
1. Fetch commissioner list → extract slugs + portfolios
2. For each commissioner page → find host UUID for transparency initiative
3. Paginate through all meetings per commissioner
4. For each meeting with minutes PDF → download and parse structured fields
"""

import hashlib
import io
import re
import time
from datetime import datetime
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

# Commissioners with their transparency initiative host UUIDs.
# Discovered by scraping commission.europa.eu pages.
# Format: {slug: {name, portfolio, host_id}}
# This is populated dynamically by discover_commissioner_host_ids().

COMMISSIONERS_URL = "https://commission.europa.eu/about/organisation/college-commissioners_en"
COMMISSIONER_BASE = "https://commission.europa.eu/about/organisation/college-commissioners"
PRESIDENT_URL = "https://commission.europa.eu/about/organisation/president_en"
MEETINGS_BASE = "https://ec.europa.eu/transparencyinitiative/meetings/meeting.do"
MINUTES_BASE = "https://ec.europa.eu/transparencyinitiative/meetings/exportmeetings.do"

# EP9 (2019-2024) Commission — different domain
EP9_COLLEGE_URL = "https://commissioners.ec.europa.eu/college-commissioners-2019-2024_en"
EP9_BASE = "https://commissioners.ec.europa.eu"

USER_AGENT = "EU-Lobby-Pipeline/1.0 (EU Parliament Transparency Research)"
REQUEST_DELAY = 1.0  # seconds between requests


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    time.sleep(REQUEST_DELAY)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def generate_meeting_id(commissioner_name: str, date_str: str, subject: str, orgs_raw: str) -> str:
    """Deterministic meeting ID from key fields."""
    raw = f"{commissioner_name}|{date_str}|{subject}|{orgs_raw}".lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def discover_commissioners(session: requests.Session, logger=None) -> list[dict]:
    """Scrape the college of commissioners page to get names, slugs, and portfolios."""
    resp = _get(session, COMMISSIONERS_URL)
    soup = BeautifulSoup(resp.text, "lxml")

    commissioners = []

    # Find all commissioner links — they point to /college-commissioners/{slug}_en
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/college-commissioners/" not in href or href.endswith("college-commissioners_en"):
            continue

        # Extract slug from URL
        match = re.search(r"/college-commissioners/([^/_]+)_en", href)
        if not match:
            continue

        slug = match.group(1)
        name = link.get_text(strip=True)

        # Skip empty or navigation-only links
        if not name or len(name) < 3:
            continue

        # Try to find portfolio text nearby
        portfolio = None
        parent = link.find_parent(["div", "li", "article"])
        if parent:
            # Look for role/portfolio text
            role_el = parent.find(class_=re.compile(r"role|portfolio|subtitle|field--name-field"))
            if role_el:
                portfolio = role_el.get_text(strip=True)

        commissioners.append({
            "name": name,
            "slug": slug,
            "portfolio": portfolio,
        })

    # Deduplicate by slug
    seen = set()
    unique = []
    for c in commissioners:
        if c["slug"] not in seen:
            seen.add(c["slug"])
            unique.append(c)

    if logger:
        logger.info(f"Discovered {len(unique)} commissioners")
    return unique


def get_host_uuids(session: requests.Session, slug: str, logger=None) -> list[dict]:
    """Visit a commissioner's page and find ALL transparency initiative meetings links.

    Returns list of {host_id, meeting_type} where meeting_type is
    'commissioner' or 'cabinet'.
    """
    url = f"{COMMISSIONER_BASE}/{slug}_en"
    try:
        resp = _get(session, url)
    except requests.RequestException as e:
        if logger:
            logger.warning(f"Failed to fetch commissioner page {slug}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")

    hosts = []
    seen_ids = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "transparencyinitiative/meetings/meeting.do" not in href or "host=" not in href:
            continue

        match = re.search(r"host=([a-f0-9-]+)", href)
        if not match:
            continue

        host_id = match.group(1)
        if host_id in seen_ids:
            continue
        seen_ids.add(host_id)

        # Determine type from link text
        link_text = link.get_text(strip=True).lower()
        if "cabinet" in link_text:
            meeting_type = "cabinet"
        else:
            meeting_type = "commissioner"

        hosts.append({"host_id": host_id, "meeting_type": meeting_type})

    if logger:
        types = [h["meeting_type"] for h in hosts]
        logger.debug(f"Found {len(hosts)} host UUIDs for {slug}: {types}")

    return hosts


def get_host_uuids_from_url(session: requests.Session, url: str, logger=None) -> list[dict]:
    """Fetch a page directly and find all transparency initiative meeting links."""
    try:
        resp = _get(session, url)
    except requests.RequestException as e:
        if logger:
            logger.warning(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    hosts = []
    seen_ids = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "transparencyinitiative/meetings/meeting.do" not in href or "host=" not in href:
            continue
        match = re.search(r"host=([a-f0-9-]+)", href)
        if not match:
            continue
        host_id = match.group(1)
        if host_id in seen_ids:
            continue
        seen_ids.add(host_id)

        link_text = link.get_text(strip=True).lower()
        meeting_type = "cabinet" if "cabinet" in link_text else "commissioner"
        hosts.append({"host_id": host_id, "meeting_type": meeting_type})

    if logger:
        logger.debug(f"Found {len(hosts)} host UUIDs from {url}")
    return hosts


def discover_ep9_commissioners(session: requests.Session, logger=None) -> list[dict]:
    """Discover 2019-2024 commissioners from commissioners.ec.europa.eu."""
    if logger:
        logger.info("Discovering EP9 (2019-2024) commissioners...")
    resp = _get(session, EP9_COLLEGE_URL)
    soup = BeautifulSoup(resp.text, "lxml")

    commissioners = []
    seen_slugs = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.endswith("_en") or href.endswith("2019-2024_en"):
            continue
        if "commissioners.ec.europa.eu" not in href and not href.startswith("/"):
            continue

        match = re.search(r"/([a-z-]+)_en$", href)
        if not match:
            continue

        slug = match.group(1)
        if slug in seen_slugs or slug in ("college-commissioners-2019-2024",):
            continue

        name = link.get_text(strip=True)
        if not name or len(name) < 3:
            continue

        seen_slugs.add(slug)
        full_url = f"{EP9_BASE}/{slug}_en" if href.startswith("/") else href
        commissioners.append({"slug": slug, "name": name, "url": full_url})

    if logger:
        logger.info(f"  Found {len(commissioners)} EP9 commissioners")
    return commissioners


def _extract_org_names(cell) -> list[str]:
    """Extract clean organization names from an HTML <td> cell.

    Each org is separated by <br> tags. Org names may be followed by
    abbreviations in parentheses like "(Volvo Cars)" which we strip.
    """
    # Split cell contents by <br> tags
    # Each segment between <br>s is one organization entry
    parts = []
    current_texts = []

    for child in cell.children:
        if getattr(child, "name", None) == "br":
            # <br> boundary — flush current text as one org
            text = " ".join(current_texts).strip()
            if text:
                parts.append(text)
            current_texts = []
        else:
            t = child.get_text(strip=True) if hasattr(child, "get_text") else str(child).strip()
            if t:
                current_texts.append(t)

    # Flush remaining
    text = " ".join(current_texts).strip()
    if text:
        parts.append(text)

    # Clean each org: strip trailing abbreviation "(ABBREV)"
    result = []
    for part in parts:
        # Remove trailing abbreviation like "(Volvo Cars)" or "(ST)"
        clean = re.sub(r"\s*\([^)]{1,40}\)\s*$", "", part).strip()
        # Skip if only an abbreviation remains or empty
        if clean and len(clean) > 1 and not re.match(r"^\([^)]+\)$", clean):
            result.append(clean)

    return result


def scrape_meetings_page(
    session: requests.Session, host_id: str, page: int = 1
) -> tuple[list[dict], int]:
    """Scrape a single page of meetings. Returns (meetings, total_count)."""
    url = f"{MEETINGS_BASE}?host={host_id}&page={page}"
    resp = _get(session, url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Parse total count from "1-10 of 81" text
    total_count = 0
    paging_text = soup.find(string=re.compile(r"\d+\s*-\s*\d+\s+of\s+\d+"))
    if paging_text:
        match = re.search(r"of\s+(\d+)", paging_text)
        if match:
            total_count = int(match.group(1))

    meetings = []
    table = soup.find("table")
    if not table:
        return meetings, total_count

    # Detect table format: commissioner pages have 4 data columns,
    # cabinet pages have 5 (extra "Commission representative(s)" first column)
    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    has_rep_column = any("commission representative" in h for h in headers)

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if has_rep_column and len(cells) >= 5:
            cabinet_member = cells[0].get_text(separator=", ", strip=True)
            date_text = cells[1].get_text(strip=True)
            location = cells[2].get_text(strip=True)
            orgs_cell = cells[3]
            subject = cells[4].get_text(strip=True)
        elif len(cells) >= 4:
            cabinet_member = None
            date_text = cells[0].get_text(strip=True)
            location = cells[1].get_text(strip=True)
            orgs_cell = cells[2]
            subject = cells[3].get_text(strip=True)
        else:
            continue

        # Find minutes PDF link in this row or nearby expandable section
        minutes_url = None
        minutes_link = row.find("a", href=re.compile(r"exportmeetings\.do"))
        if not minutes_link:
            # Check next sibling row (expandable detail)
            next_row = row.find_next_sibling("tr")
            if next_row:
                minutes_link = next_row.find("a", href=re.compile(r"exportmeetings\.do"))
        if minutes_link:
            minutes_url = minutes_link["href"]
            if not minutes_url.startswith("http"):
                minutes_url = f"https://ec.europa.eu{minutes_url}"

        # Parse date
        parsed_date = None
        for fmt in ("%d/%m/%Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(date_text, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        # Extract org names as clean list from HTML structure
        org_names = _extract_org_names(orgs_cell)
        orgs_raw = orgs_cell.get_text(separator="\n", strip=True)

        meeting = {
            "date_raw": date_text,
            "meeting_date": parsed_date,
            "location": location,
            "organizations": org_names,
            "organizations_raw": orgs_raw,
            "subject": subject,
            "cabinet_member": cabinet_member,
            "minutes_url": minutes_url,
            "source_url": url,
        }
        meetings.append(meeting)

    return meetings, total_count


def scrape_all_meetings(
    session: requests.Session, host_id: str, logger=None
) -> list[dict]:
    """Paginate through all meetings for a commissioner."""
    all_meetings = []
    page = 1

    meetings, total = scrape_meetings_page(session, host_id, page=1)
    all_meetings.extend(meetings)

    if logger and total > 0:
        logger.info(f"  Host {host_id}: {total} meetings total, page 1 got {len(meetings)}")

    # Each page has 10 meetings
    if total > 10:
        total_pages = (total + 9) // 10
        for page in range(2, total_pages + 1):
            try:
                meetings, _ = scrape_meetings_page(session, host_id, page=page)
                all_meetings.extend(meetings)
                if logger:
                    logger.debug(f"  Page {page}/{total_pages}: {len(meetings)} meetings")
            except Exception as e:
                if logger:
                    logger.warning(f"  Failed page {page}: {e}")

    return all_meetings


def parse_minutes_pdf(pdf_content: bytes, logger=None) -> Optional[dict]:
    """Parse a meeting minutes PDF and extract structured fields.

    Expected fields: Date, Location, Commission Representatives,
    Interest Representatives (with REG number), Subject,
    Main points raised, Conclusions, Ares number.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_content))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts)
        if not full_text.strip():
            return None

    except Exception as e:
        if logger:
            logger.warning(f"Failed to extract PDF text: {e}")
        return None

    result = {"full_text": full_text}

    # Extract structured fields using regex patterns
    # These patterns match the standard Commission meeting minutes format
    patterns = {
        # "Subject matter:" or just "Subject " (no colon, no "matter")
        "subject": (
            r"Subject(?:\s+matter)?\s*:?\s*(.+?)"
            r"(?=Main\s+points|Points\s+raised|$)"
        ),
        # "Main points raised and positions expressed:" — colon optional
        # Stop at "Conclusion" only when it's a field label (after newline),
        # not when "conclusion" appears mid-sentence.
        # Don't stop at "European Commission" — it appears in normal text too.
        "points_raised": (
            r"(?:Main\s+points\s+raised\s+(?:and\s+positions?\s+expressed\s*)?|"
            r"Points\s+raised)\s*:?\s*(.+?)"
            r"(?=\n\s*Conclusions?\s*:?\s*(?:\n|[A-Z])|Ares\s*(?:number|\()|$)"
        ),
        # "Conclusions:" — colon optional, stop at Ares or "EUROPEAN COMMISSION"
        # page header (require newline before it to avoid matching mid-text)
        "conclusions": (
            r"\n\s*Conclusions?\s*:?\s*(.*?)"
            r"(?=\nEUROPEAN\s+COMMISSION|Ares\s*(?:number|\()|$)"
        ),
        "ares_number": r"(Ares\(\d{4}\)\d+(?:\s*/\s*Ares\(\d{4}\)\d+)*)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Normalize whitespace
            value = re.sub(r"\s+", " ", value).strip()
            # Strip leaked field label prefixes (PDF layout artifacts)
            value = re.sub(
                r"^(?:Subject\s+)?matter\s*:\s*|^(?:Main\s+)?points?\s+raised"
                r"(?:\s+and\s+positions?\s+expressed)?\s*:\s*|^Conclusions?\s*:\s*",
                "", value, flags=re.IGNORECASE,
            ).strip()
            # Strip leading punctuation/symbols (stray colons, dashes, bullets)
            value = re.sub(r"^[\s:;\-–—•*,]+", "", value).strip()
            # Remove trailing page numbers (e.g., " 2" at end from PDF pagination)
            value = re.sub(r"\s+\d{1,2}\s*$", "", value).strip()
            # Normalize bullet points: • → newline-separated clean text
            if "\u2022" in value:
                bullets = [b.strip() for b in value.split("\u2022") if b.strip()]
                value = "\n".join(f"- {b}" for b in bullets)
            # Also handle dash bullets that got collapsed
            elif re.match(r"^[-–]\s", value):
                lines = re.split(r"\s+[-–]\s+", value)
                value = "\n".join(f"- {l.strip()}" for l in lines if l.strip())
            # Strip surrounding quotes before N/A check
            value = re.sub(r"^['\"\s]+|['\"\s]+$", "", value).strip()
            na_values = {"n.a.", "n/a", "na", "-", "\u2013", "none"}
            if value and value.lower() not in na_values:
                result[field] = value
            elif field == "conclusions":
                result[field] = None  # explicit N/A

    # Extract transparency register IDs (format: digits-digits)
    tr_ids = re.findall(r"\b(\d{10,}-\d{2})\b", full_text)
    if tr_ids:
        result["transparency_register_ids"] = list(set(tr_ids))

    # Extract commission representatives
    repr_match = re.search(
        r"(?:Names and functions\s*of the Commission\s*Representatives?|"
        r"Commission\s*Representatives?)\s*(.+?)(?=Names? of the interest|Interest representative|$)",
        full_text,
        re.DOTALL | re.IGNORECASE,
    )
    if repr_match:
        repr_text = repr_match.group(1).strip()
        # Split by newlines or common separators
        reps = []
        for line in re.split(r"\n|;", repr_text):
            line = line.strip()
            if not line or line.upper() == "N.A.":
                continue
            # Try to extract name and function
            func_match = re.search(
                r"(.+?),\s*(Commissioner|Cabinet\s*Member|Head\s*of\s*Cabinet|"
                r"Director.General|Member\s*of\s*Cabinet|Chef\s*de\s*Cabinet)",
                line,
                re.IGNORECASE,
            )
            if func_match:
                reps.append({"name": func_match.group(1).strip(), "function": func_match.group(2).strip()})
            else:
                # Check reverse pattern: "Commissioner Name"
                func_match2 = re.search(
                    r"(Commissioner|Cabinet\s*Member|Head\s*of\s*Cabinet)\s+(.+)",
                    line,
                    re.IGNORECASE,
                )
                if func_match2:
                    reps.append({"name": func_match2.group(2).strip(), "function": func_match2.group(1).strip()})
                elif len(line) > 2:
                    reps.append({"name": line, "function": None})
        if reps:
            result["commission_representatives"] = reps

    return result


def download_minutes(
    session: requests.Session, minutes_url: str, logger=None
) -> Optional[dict]:
    """Download and parse a meeting minutes PDF. Returns None on failure (expected)."""
    try:
        resp = session.get(minutes_url, timeout=60)
        resp.raise_for_status()

        if "application/pdf" not in resp.headers.get("Content-Type", ""):
            if logger:
                logger.debug(f"Minutes not a PDF: {minutes_url}")
            return None

        return parse_minutes_pdf(resp.content, logger=logger)

    except Exception as e:
        if logger:
            logger.debug(f"Minutes download failed (expected for some): {e}")
        return None


def _load_pdf_cache(logger=None) -> dict[str, dict]:
    """Load previously parsed PDF data from existing bronze output.

    Returns dict keyed by meeting ID with minutes fields.
    """
    import json
    import os

    cache_path = os.path.join("data", "eu_commission_meetings_bronze", "data.json")
    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cache = {}
        pdf_fields = [
            "minutes_subject", "points_raised", "conclusions",
            "ares_number", "transparency_register_ids", "commission_representatives",
        ]
        # Don't cache any previous PDF parses — the old regex was broken
        # and even "successful" extractions may be truncated or partial.
        # All PDFs will be re-downloaded and re-parsed with the fixed regex.
        if logger:
            logger.info("PDF cache: skipping (full re-parse needed due to regex fixes)")
        return {}
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load PDF cache: {e}")
        return {}


def scrape_commission_meetings(context, actors: list[dict]) -> list[dict]:
    """Full bronze scrape: use actors from DB → find host UUIDs → scrape meetings → parse minutes.

    Args:
        context: Dagster context with logger
        actors: List of actor records from the actors table (must have profile_url, fullName, portfolio)
    """
    logger = context.log
    session = _make_session()

    # Step 1: Use actors from DB instead of re-scraping commissioner list
    logger.info(f"Step 1: Using {len(actors)} actors from database...")
    if not actors:
        raise ValueError("No actors found in database — run actors pipeline first")

    # Step 2: Get host UUIDs from each actor's profile_url (both commissioner + cabinet)
    logger.info("Step 2: Finding meeting page UUIDs (commissioner + cabinet)...")
    scrape_targets = []
    for actor in actors:
        profile_url = actor.get("profile_url")
        name = actor.get("fullName", "")
        if not profile_url:
            logger.info(f"  Skipping {name} (no profile_url)")
            continue

        # Extract slug from profile_url — handle both commissioner and president URLs
        slug = None
        match = re.search(r"/college-commissioners/([^/_]+)_en", profile_url)
        if match:
            slug = match.group(1)
            hosts = get_host_uuids(session, slug, logger)
        elif "/president" in profile_url:
            # President's page doesn't expose the meetings links in the same way
            hosts = [
                {"host_id": "a2c7c963-a9ad-4c47-aa73-4bb46b06dd5d", "meeting_type": "commissioner"},
                {"host_id": "9fd4662a-8580-4cee-bb3f-3c2fba5c12c6", "meeting_type": "cabinet"},
            ]
        else:
            logger.debug(f"  Non-standard URL for {name}: {profile_url}")
            continue
        if hosts:
            for h in hosts:
                scrape_targets.append({
                    "actor_id": actor.get("actor_id"),
                    "name": name,
                    "portfolio": actor.get("portfolio"),
                    "slug": slug,
                    **h,
                })
        else:
            logger.info(f"  Skipping {name} (no meetings page found)")

    commissioner_count = sum(1 for t in scrape_targets if t["meeting_type"] == "commissioner")
    cabinet_count = sum(1 for t in scrape_targets if t["meeting_type"] == "cabinet")
    logger.info(
        f"Found {len(scrape_targets)} meeting pages: "
        f"{commissioner_count} commissioner, {cabinet_count} cabinet"
    )

    return _scrape_meetings_from_targets(session, scrape_targets, logger)


def scrape_ep9_commission_meetings(context) -> list[dict]:
    """Scrape EP9 (2019-2024) commission meetings via commissioner discovery.

    No actors table needed — discovers commissioners from the EP9 college page.
    """
    logger = context.log
    session = _make_session()

    # Step 1: Discover EP9 commissioners
    commissioners = discover_ep9_commissioners(session, logger)

    # Step 2: Get host UUIDs for each
    logger.info("Finding EP9 meeting page UUIDs...")
    scrape_targets = []
    for comm in commissioners:
        hosts = get_host_uuids_from_url(session, comm["url"], logger)
        if hosts:
            for h in hosts:
                scrape_targets.append({
                    "actor_id": None,
                    "name": comm["name"],
                    "portfolio": None,
                    "slug": comm["slug"],
                    **h,
                })
            types = [h["meeting_type"] for h in hosts]
            logger.info(f"  {comm['name']}: {', '.join(types)}")
        else:
            logger.info(f"  {comm['name']}: no meeting pages found")

    commissioner_count = sum(1 for t in scrape_targets if t["meeting_type"] == "commissioner")
    cabinet_count = sum(1 for t in scrape_targets if t["meeting_type"] == "cabinet")
    logger.info(
        f"EP9 targets: {len(scrape_targets)} "
        f"({commissioner_count} commissioner, {cabinet_count} cabinet)"
    )

    return _scrape_meetings_from_targets(session, scrape_targets, logger)


def _scrape_meetings_from_targets(
    session: requests.Session, scrape_targets: list[dict], logger
) -> list[dict]:
    """Shared logic: scrape meetings from targets and download minutes PDFs."""
    # Step 3: Scrape meetings for each target
    logger.info("Scraping meetings...")
    all_meetings = []
    for target in scrape_targets:
        label = f"{target['name']} ({target['meeting_type']})"
        logger.info(f"  Scraping {label}...")
        try:
            meetings = scrape_all_meetings(session, target["host_id"], logger)

            for meeting in meetings:
                meeting["actor_id"] = target.get("actor_id")
                meeting["commissioner_name"] = target["name"]
                meeting["commissioner_portfolio"] = target.get("portfolio")
                meeting["host_id"] = target["host_id"]
                meeting["meeting_type"] = target["meeting_type"]

                # Generate deterministic ID — include host_id to distinguish
                # commissioner vs cabinet meetings for same date/subject/org
                meeting["id"] = generate_meeting_id(
                    target["host_id"],
                    meeting.get("meeting_date", meeting.get("date_raw", "")),
                    meeting.get("subject", ""),
                    meeting.get("organizations_raw", ""),
                )

            all_meetings.extend(meetings)
            logger.info(f"    → {len(meetings)} meetings")
        except Exception as e:
            logger.error(f"  Failed to scrape {label}: {e}")

    logger.info(f"Total meetings scraped: {len(all_meetings)}")

    # Step 4: Download and parse meeting minutes PDFs (with caching + concurrency)
    pdf_cache = _load_pdf_cache(logger)

    # Separate cached vs needs-download
    to_download = []
    minutes_cached = 0
    minutes_skipped = 0

    for meeting in all_meetings:
        if not meeting.get("minutes_url"):
            minutes_skipped += 1
            continue

        cache_key = f"{meeting.get('host_id')}|{meeting.get('meeting_date', '')}|{meeting.get('subject', '')}"
        cached = pdf_cache.get(cache_key)
        if cached:
            for field, value in cached.items():
                if value is not None:
                    meeting[field] = value
            minutes_cached += 1
        else:
            to_download.append(meeting)

    logger.info(
        f"PDF minutes — {minutes_cached} cached, "
        f"{len(to_download)} to download, {minutes_skipped} no URL"
    )

    # Download PDFs concurrently (5 workers to be respectful)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    minutes_success = 0
    minutes_failed = 0

    def _download_one(meeting):
        """Download and parse one PDF. Returns (meeting, parsed_data)."""
        # Use requests.get directly — requests.Session is not thread-safe
        try:
            resp = requests.get(
                meeting["minutes_url"],
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            if "application/pdf" not in resp.headers.get("Content-Type", ""):
                return meeting, None
            return meeting, parse_minutes_pdf(resp.content)
        except Exception:
            return meeting, None

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_download_one, m): m for m in to_download}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if done_count % 200 == 0:
                logger.info(
                    f"  PDF progress: {done_count}/{len(to_download)} "
                    f"(success={minutes_success}, failed={minutes_failed})"
                )
            try:
                meeting, parsed = future.result()
                if parsed:
                    minutes_success += 1
                    meeting["minutes_subject"] = parsed.get("subject")
                    meeting["points_raised"] = parsed.get("points_raised")
                    meeting["conclusions"] = parsed.get("conclusions")
                    meeting["ares_number"] = parsed.get("ares_number")
                    meeting["transparency_register_ids"] = parsed.get("transparency_register_ids", [])
                    meeting["commission_representatives"] = parsed.get("commission_representatives", [])
                else:
                    minutes_failed += 1
            except Exception:
                minutes_failed += 1

    logger.info(
        f"Minutes: {minutes_success} new, {minutes_cached} cached, "
        f"{minutes_failed} failed, {minutes_skipped} no URL"
    )

    session.close()
    return all_meetings
