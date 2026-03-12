"""Bronze stage: Extract raw lobbying data from transparency register and meetings API.

This module provides extraction functions for:
1. Meetings: Fetches from EU Parliament search-meetings endpoint via progressive web-scraping.
2. Organizations: Loads from Transparency Register XLS file.
"""

import json
import queue
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tenacity import retry, stop_after_attempt, wait_exponential


# ––– BROWSER POOL CLASS –––#
class BrowserPool:
    """Manages a pool of Chrome browser instances for efficient scraping."""

    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.drivers = queue.Queue()
        self.lock = threading.Lock()
        self._initialize_pool()

    def _create_driver(self):
        """Create a new Chrome driver instance."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-images")
        options.add_argument("--disable-dev-tools")
        options.add_argument("--disable-web-security")

        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            return driver
        except Exception as e:
            print(f"Failed to create Chrome driver: {e}")
            raise

    def _initialize_pool(self):
        """Initialize the browser pool."""
        for _ in range(self.pool_size):
            try:
                driver = self._create_driver()
                self.drivers.put(driver)
            except Exception as e:
                print(f"Failed to create browser instance: {e}")

    def get_driver(self):
        """Get a driver from the pool, create new one if pool is empty."""
        try:
            driver = self.drivers.get_nowait()
            # Clear any existing page state
            try:
                driver.delete_all_cookies()
                driver.execute_script("window.localStorage.clear();")
                driver.execute_script("window.sessionStorage.clear();")
            except Exception:
                pass
            return driver
        except queue.Empty:
            # Pool is empty, create a new driver
            return self._create_driver()

    def return_driver(self, driver):
        """Return a driver to the pool."""
        if driver and self.drivers.qsize() < self.pool_size:
            try:
                # Clear any remaining page state
                driver.get("about:blank")
                driver.delete_all_cookies()
                driver.execute_script("window.localStorage.clear();")
                driver.execute_script("window.sessionStorage.clear();")
                self.drivers.put(driver)
            except Exception:
                # Driver is broken, quit it
                try:
                    if driver:
                        driver.quit()
                except Exception:
                    pass
        else:
            # Pool is full or driver is None, quit the driver
            try:
                if driver:
                    driver.quit()
            except Exception:
                pass

    def close_all(self):
        """Close all drivers in the pool."""
        while not self.drivers.empty():
            try:
                driver = self.drivers.get_nowait()
                driver.quit()
            except Exception:
                pass


# Global browser pool instance
_browser_pool = None


def get_browser_pool():
    """Get the global browser pool instance."""
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool(pool_size=3)
    return _browser_pool


def close_browser_pool():
    """Close the global browser pool and free resources."""
    global _browser_pool
    if _browser_pool:
        try:
            _browser_pool.close_all()
        except Exception:
            pass
        finally:
            _browser_pool = None


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Sanitize text to ensure UTF-8 compatibility.

    Removes or replaces characters that cause encoding issues.
    Handles Windows-1252 characters commonly found in EU Parliament data.
    """
    if not text:
        return text

    # Encode as UTF-8 with error replacement, then decode back
    # This replaces problematic characters with '?'
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def extract_transparency_register_id(url: str) -> Optional[str]:
    """Extract transparency register ID from EU transparency register URLs.

    Args:
        url: URL like 'https://transparency-register.europa.eu/search-details_en?id=68368571120-55&prefLang=...'

    Returns:
        The transparency register ID (e.g., '68368571120-55') or None if not found
    """
    if not url or "transparency-register.europa.eu" not in url:
        return None

    # Extract ID parameter from URL
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else None


def parse_meeting_html_elements(
    meeting_divs: List[Any], logger: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Parse meeting data from HTML div.es_document elements.

    Args:
        meeting_divs: List of BeautifulSoup div.es_document elements (updated from erpl_document)
        logger: Optional logger

    Returns:
        List of meeting dictionaries
    """
    meetings = []

    for div in meeting_divs:
        try:
            # Extract title
            title_elem = div.find("h3", class_="es_document-title")
            title = None
            if title_elem:
                title_span = title_elem.find("span", class_="t-item")
                title = sanitize_text(title_span.text.strip()) if title_span else None

            # Extract member info
            member_elem = div.find("span", class_="es_document-subtitle-member")
            member_name = sanitize_text(member_elem.text.strip()) if member_elem else None

            # Note: Member ID will be matched by name in scrape_meetings_progressive()
            # The HTML no longer contains member links
            member_id = None

            # Extract date
            date_elem = div.find("time")
            meeting_date = date_elem.get("datetime") if date_elem else None

            # Extract location
            location_elem = div.find("span", class_="es_document-subtitle-location")
            location = sanitize_text(location_elem.text.strip()) if location_elem else None

            # Extract capacity
            capacity_elem = div.find("span", class_="es_document-subtitle-capacity")
            member_capacity = sanitize_text(capacity_elem.text.strip()) if capacity_elem else None

            # Extract procedure reference
            procedure_elem = div.find("span", class_="es_document-subtitle-reference")
            procedure_reference = (
                sanitize_text(procedure_elem.text.strip()) if procedure_elem else None
            )

            # Extract committee code
            committee_elem = div.find("span", class_="es_badge-committee")
            committee_code = sanitize_text(committee_elem.text.strip()) if committee_elem else None

            # Extract attendees and transparency register IDs
            author_elems = div.find_all("span", class_="es_document-subtitle-author")
            attendees = []
            lobbyist_ids = []

            for author in author_elems:
                # Check for transparency register link
                transparency_link = author.find(
                    "a",
                    href=lambda href: href and "transparency-register.europa.eu" in href,
                )

                if transparency_link:
                    # Extract organization name from link text
                    org_name = sanitize_text(transparency_link.text.strip())
                    # Extract transparency register ID
                    transparency_id = extract_transparency_register_id(
                        transparency_link.get("href", "")
                    )
                    if org_name:
                        attendees.append(org_name)
                    if transparency_id:
                        lobbyist_ids.append(transparency_id)
                else:
                    # No transparency register link, just get the text
                    org_name = sanitize_text(author.get_text(separator="|", strip=True))
                    if org_name:
                        attendees.append(org_name)

            # Join attendees and lobbyist IDs
            attendees_str = "|".join(attendees) if attendees else None
            lobbyist_id = lobbyist_ids[0] if lobbyist_ids else None

            # Only include meetings with actual organization attendance
            if attendees_str:
                meeting = {
                    "title": title,
                    "member_id": member_id,
                    "member_name": member_name,
                    "meeting_date": meeting_date,
                    "member_capacity": member_capacity,
                    "procedure_reference": procedure_reference,
                    "attendees": attendees_str,
                    "lobbyist_id": lobbyist_id,
                    "committee_code": committee_code,
                    "location": location,
                }
                meetings.append(meeting)

        except Exception as e:
            if logger:
                logger.warning(f"Error parsing meeting element: {e}")
            continue

    return meetings


def get_active_meps_from_facets(
    driver, from_date: str, to_date: str, logger=None
) -> List[Tuple[str, int, str]]:
    """
    Fetches the list of MEPs who have meetings in the given date range using the facets endpoint.
    Returns a list of (mep_id, expected_count, mep_name) tuples.
    """

    # Format dates for URL (dd/MM/yyyy)
    def format_date_param(d_str):
        try:
            if "-" in d_str:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                return dt.strftime("%d/%m/%Y")
            return d_str
        except ValueError:
            return d_str

    f_from = quote(format_date_param(from_date), safe="")
    f_to = quote(format_date_param(to_date), safe="")

    url = (
        f"https://www.europarl.europa.eu/meps/en/search-meetings/facets?"
        f"textualSearch=&fromDate={f_from}&toDate={f_to}"
    )

    if logger:
        logger.info(f"Fetching MEP facets from: {url}")

    driver.get(url)

    try:
        content = driver.find_element(By.TAG_NAME, "body").text
        try:
            pre = driver.find_element(By.TAG_NAME, "pre").text
            content = pre
        except NoSuchElementException:
            pass

        data = json.loads(content)
        meps = []

        # Find the 'memberIds' facet
        fields = data.get("fields", [])
        for field in fields:
            if field.get("name") == "memberIds":
                values = field.get("availableValues", [])
                for val in values:
                    mep_id = val.get("value")
                    label = val.get("label", "")

                    # Label is "NAME Name (Count)"
                    count_match = re.search(r"\((\d+)\)$", label)
                    count = int(count_match.group(1)) if count_match else 0

                    # Extract name part (remove count)
                    name = re.sub(r"\s*\(\d+\)$", "", label).strip()

                    if mep_id:
                        meps.append((mep_id, count, name))
                break

        return meps

    except Exception as e:
        if logger:
            logger.error(f"Error parsing facets: {e}")
        return []


def scrape_mep_chunk(
    browser_pool,
    mep_ids,
    expected_count,
    from_date,
    to_date,
    mep_lookup=None,
    logger=None,
):
    """
    Scrapes meetings for a bundle of MEPs using a driver from the pool.
    mep_lookup: dict of { normalized_name : mep_id } to resolve missing IDs.
    """
    driver = None
    meetings = []

    def format_date_param(d_str):
        try:
            if "-" in d_str:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                return dt.strftime("%d/%m/%Y")
            return d_str
        except ValueError:
            return d_str

    f_from = format_date_param(from_date)
    f_to = format_date_param(to_date)

    params = [("memberIds", mid) for mid in mep_ids]
    params.append(("textualSearch", ""))
    params.append(("fromDate", f_from))
    params.append(("toDate", f_to))

    query_string = urlencode(params, quote_via=quote)
    url = f"https://www.europarl.europa.eu/meps/en/search-meetings?{query_string}"

    try:
        driver = browser_pool.get_driver()
        driver.get(url)
        time.sleep(1.0)

        # Handle "Load more"
        click_count = 0
        max_clicks = 60  # Increased for larger chunks
        consecutive_no_growth = 0
        last_count = 0

        while click_count < max_clicks:
            try:
                current_docs = driver.find_elements(By.CLASS_NAME, "es_document")
                current_count = len(current_docs)

                if current_count == last_count and click_count > 0:
                    consecutive_no_growth += 1
                else:
                    consecutive_no_growth = 0

                # More patience for larger lists
                if consecutive_no_growth > 3:
                    break

                last_count = current_count

                if current_count >= expected_count:
                    break

                load_more = driver.find_element(
                    By.CSS_SELECTOR, "button.europarl-expandable-async-loadmore"
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
                time.sleep(0.2)  # Wait for scroll
                driver.execute_script("arguments[0].click();", load_more)
                click_count += 1
                time.sleep(2.0)  # Increased wait for async JS to complete (was 0.6s)

            except NoSuchElementException:
                break
            except Exception:
                break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        meeting_divs = soup.find_all("div", class_="es_document")

        meetings = parse_meeting_html_elements(meeting_divs, logger)

        # Post-process to fix missing IDs
        if mep_lookup:
            for m in meetings:
                if not m.get("member_id") and m.get("member_name"):
                    # Normalize name from HTML
                    raw_name = m["member_name"]
                    # Remove "Mrs " etc? Usually just Name
                    normalized = " ".join(sorted(raw_name.lower().split()))

                    matched_id = mep_lookup.get(normalized)
                    if matched_id:
                        m["member_id"] = matched_id

    except Exception as e:
        if logger:
            logger.error(f"Error scraping chunk of {len(mep_ids)} MEPs: {e}")
    finally:
        if driver:
            browser_pool.return_driver(driver)

    return meetings


def scrape_meetings_progressive(
    from_date: str,
    to_date: str,
    batch_size: int = 100,
    logger: Optional[Any] = None,
) -> Generator[List[Dict[str, Any]] | int, None, None]:
    """
    Scrapes meetings using page-based pagination (10 results per page).

    The EU Parliament Load More button is broken, but page-based URLs work.
    Example: ...&page=1, ...&page=2, etc.

    Note: Member IDs are no longer in the HTML, so we match names to IDs using facets data.
    """
    if logger:
        logger.info("Starting page-based meeting scraping")

    browser_pool = get_browser_pool()
    driver = browser_pool.get_driver()

    try:
        # Format dates
        def format_date_param(d_str):
            try:
                if "-" in d_str:
                    dt = datetime.strptime(d_str, "%Y-%m-%d")
                    return dt.strftime("%d/%m/%Y")
                return d_str
            except ValueError:
                return d_str

        f_from = format_date_param(from_date)
        f_to = format_date_param(to_date)

        # Get MEP lookup from facets (for name -> ID matching)
        mep_lookup = {}
        try:
            facets_driver = browser_pool.get_driver()
            meps_with_counts = get_active_meps_from_facets(
                facets_driver, from_date, to_date, logger
            )
            browser_pool.return_driver(facets_driver)

            # Build name lookup: normalize names for matching
            # Facet names are "LASTNAME Firstname", HTML names are "LASTNAME Firstname"
            for mep_id, _, name in meps_with_counts:
                # Normalize: lowercase and sort words
                normalized = " ".join(sorted(name.lower().split()))
                mep_lookup[normalized] = mep_id

            if logger:
                logger.info(f"Built MEP lookup with {len(mep_lookup)} entries")
        except Exception as e:
            if logger:
                logger.warning(f"Could not build MEP lookup: {e}")

        # Build base URL
        base_url = (
            f"https://www.europarl.europa.eu/meps/en/search-meetings?"
            f"textualSearch=&fromDate={quote(f_from)}&toDate={quote(f_to)}"
        )

        # Get total count from first page
        if logger:
            logger.info(f"Fetching first page: {base_url}")

        driver.get(base_url)
        time.sleep(2.0)

        # Get total expected count from page counter
        try:
            counter_elem = driver.find_element(By.ID, "meetingSearchResultCounterText")
            counter_text = counter_elem.text  # "Showing 10 of 450 results"
            match = re.search(r"of\s+(\d+)", counter_text)
            total_expected = int(match.group(1)) if match else 0
            if logger:
                logger.info(f"Total meetings: {total_expected}")
        except Exception:
            total_expected = 0

        yield total_expected  # Yield expected count first

        # Calculate pages (10 results per page)
        results_per_page = 10
        num_pages = (total_expected + results_per_page - 1) // results_per_page

        if logger:
            logger.info(f"Fetching {num_pages} pages")

        all_meetings = []

        # Parse first page (already loaded)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        meeting_divs = soup.find_all("div", class_="es_document")
        first_page_meetings = parse_meeting_html_elements(meeting_divs, logger)

        # Post-process: match member names to IDs for first page
        for meeting in first_page_meetings:
            if not meeting.get("member_id") and meeting.get("member_name"):
                raw_name = meeting["member_name"]
                normalized = " ".join(sorted(raw_name.lower().split()))
                matched_id = mep_lookup.get(normalized)
                if matched_id:
                    meeting["member_id"] = matched_id

        all_meetings.extend(first_page_meetings)

        if logger:
            logger.info(f"Parsed first page: {len(first_page_meetings)} meetings")

        # Fetch remaining pages.
        # The EU Parliament site uses 0-based page indexing: no ?page param = page 0
        # (already loaded above). Subsequent pages are &page=1, &page=2, …
        for page_num in range(1, num_pages):
            page_url = f"{base_url}&page={page_num}"

            if logger and page_num % 10 == 0:
                logger.info(f"Page {page_num}/{num_pages - 1}...")

            driver.get(page_url)
            time.sleep(1.0)

            # Parse meetings
            soup = BeautifulSoup(driver.page_source, "html.parser")
            meeting_divs = soup.find_all("div", class_="es_document")

            # Stop if page is empty (website pagination bug)
            if not meeting_divs:
                if logger:
                    logger.info(f"Page {page_num} is empty, stopping pagination")
                break

            page_meetings = parse_meeting_html_elements(meeting_divs, logger)

            # Post-process: match member names to IDs
            for meeting in page_meetings:
                if not meeting.get("member_id") and meeting.get("member_name"):
                    # Normalize name from HTML
                    raw_name = meeting["member_name"]
                    normalized = " ".join(sorted(raw_name.lower().split()))

                    matched_id = mep_lookup.get(normalized)
                    if matched_id:
                        meeting["member_id"] = matched_id

            all_meetings.extend(page_meetings)

            # Yield in batches
            while len(all_meetings) >= batch_size:
                batch = all_meetings[:batch_size]
                all_meetings = all_meetings[batch_size:]
                if logger:
                    logger.info(f"Yielding batch of {len(batch)} meetings")
                yield batch

        # Yield remaining meetings and log final match-rate stats
        if all_meetings:
            matched = sum(1 for m in all_meetings if m.get("member_id"))
            total = len(all_meetings)
            if logger:
                logger.info(f"Yielding final batch of {total} meetings")
                logger.info(
                    f"Member ID match rate: {matched}/{total} "
                    f"({100 * matched // total if total else 0}%)"
                )
            yield all_meetings

    except Exception as e:
        if logger:
            logger.error(f"Error scraping meetings: {e}")
        raise
    finally:
        browser_pool.return_driver(driver)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_meetings_scraped(
    from_date: str,
    to_date: str,
    logger: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetch lobbying meetings from EU Parliament search-meetings endpoint via web-scraping.

    Uses progressive Selenium-based scraping with batching to handle large datasets (1500+ meetings).
    Automatically handles date ranges larger than 31 days by splitting requests.

    Endpoint: https://www.europarl.europa.eu/meps/en/search-meetings

    Args:
        from_date: Start date in DD/MM/YYYY format (e.g., "01/12/2025")
        to_date: End date in DD/MM/YYYY format (e.g., "14/12/2025")
        logger: Optional logger

    Returns:
        Tuple of (list of meeting dictionaries, total expected meetings count)
    """
    # Parse dates to check range
    try:
        start_dt = datetime.strptime(from_date, "%d/%m/%Y")
        end_dt = datetime.strptime(to_date, "%d/%m/%Y")
    except ValueError as e:
        if logger:
            logger.error(f"Invalid date format: {e}")
        raise

    # If range > 31 days, split into monthly chunks
    if (end_dt - start_dt).days > 31:
        all_meetings = []
        total_expected_sum = 0
        current_start = start_dt

        while current_start <= end_dt:
            # Chunk size of ~1 month (30 days)
            current_end = min(current_start + timedelta(days=30), end_dt)

            chunk_from = current_start.strftime("%d/%m/%Y")
            chunk_to = current_end.strftime("%d/%m/%Y")

            if logger:
                logger.info(f"Processing chunk: {chunk_from} to {chunk_to}")

            # Recursive call for the chunk
            chunk_meetings, chunk_expected = fetch_meetings_scraped(chunk_from, chunk_to, logger)
            all_meetings.extend(chunk_meetings)
            total_expected_sum += chunk_expected

            current_start = current_end + timedelta(days=1)

        return all_meetings, total_expected_sum

    # Base case: scrape single chunk using progressive scraping
    if logger:
        logger.info(f"Scraping meetings from {from_date} to {to_date}")

    all_meetings = []

    try:
        # Use progressive scraping generator
        gen = scrape_meetings_progressive(from_date, to_date, batch_size=100, logger=logger)

        # First yielded value is always total_expected (int)
        total_expected = next(gen)
        if not isinstance(total_expected, int):
            # Fallback if protocol mismatch, though shouldn't happen
            total_expected = 0

        for batch in gen:
            all_meetings.extend(batch)

            if logger:
                logger.info(
                    f"Collected batch of {len(batch)} meetings (total so far: {len(all_meetings)})"
                )

        if logger:
            logger.info(f"Scraping complete. Total meetings: {len(all_meetings)}")

        return all_meetings, total_expected

    except Exception as e:
        if logger:
            logger.error(f"Error scraping meetings: {e}")
        raise


def _sanitize_xml(xml_content: bytes) -> str:
    """Sanitize XML content by removing invalid character references.

    The EU Transparency Register XML sometimes contains invalid XML character
    references (e.g., &#1; &#8; etc.) that need to be removed before parsing.

    Args:
        xml_content: Raw XML bytes from API

    Returns:
        Sanitized XML string safe for parsing
    """
    # Decode to string
    xml_str = xml_content.decode("utf-8", errors="ignore")

    # Remove invalid XML character references
    # Valid XML chars: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    # Pattern matches &#NUMBER; where NUMBER is invalid
    def is_valid_xml_char(match):
        """Check if character reference is valid XML."""
        char_num = int(match.group(1))
        # Valid ranges
        if char_num in (0x9, 0xA, 0xD):
            return True
        if 0x20 <= char_num <= 0xD7FF:
            return True
        if 0xE000 <= char_num <= 0xFFFD:
            return True
        if 0x10000 <= char_num <= 0x10FFFF:
            return True
        return False

    def replace_char_ref(match):
        """Replace invalid char refs with empty string."""
        if is_valid_xml_char(match):
            return match.group(0)  # Keep valid ones
        return ""  # Remove invalid ones

    # Replace both decimal (&#N;) and hex (&#xN;) character references
    xml_str = re.sub(r"&#(\d+);", replace_char_ref, xml_str)
    xml_str = re.sub(
        r"&#x([0-9a-fA-F]+);",
        lambda m: replace_char_ref(re.match(r"&#(\d+);", f"&#{int(m.group(1), 16)};")),
        xml_str,
    )

    return xml_str


def load_transparency_register(
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Load organizations from EU Transparency Register XML API.

    Fetches the latest data from the official EU Transparency Register API.

    API URL: https://transparency-register.europa.eu/odplastorganisationxml_en

    Args:
        logger: Optional logger

    Returns:
        List of organization dictionaries

    Raises:
        Exception: If API request fails
    """
    url = "https://transparency-register.europa.eu/odplastorganisationxml_en"

    if logger:
        logger.info(f"Fetching transparency register from {url}")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    # Sanitize XML to remove invalid character references
    if logger:
        logger.info("Sanitizing XML content")
    xml_str = _sanitize_xml(response.content)

    # Parse XML
    if logger:
        logger.info("Parsing XML")
    root = ET.fromstring(xml_str)

    organizations = []

    # Find resultList (has xmlns="", so no namespace)
    # The root has namespace, but resultList removes it with xmlns=""
    result_list = root.find("resultList")
    if result_list is None:
        # Try with namespace
        ns = {"odp": "http://intragate.ec.europa.eu/transparencyregister/odp"}
        result_list = root.find("odp:resultList", ns)

    if result_list is None:
        if logger:
            logger.error("Could not find resultList element")
        return []

    # Find all interestRepresentative elements (no namespace due to xmlns="")
    ir_elements = result_list.findall("interestRepresentative")

    if logger:
        logger.info(f"Found {len(ir_elements)} interest representatives")

    for ir_elem in ir_elements:
        org_data = {}

        # Helper to get text from element
        def get_text(elem, tag_name):
            """Get text from child element."""
            child = elem.find(tag_name)
            if child is not None and child.text:
                return sanitize_text(child.text.strip())
            return None

        # Extract fields (no namespace needed - xmlns="" removes it)
        org_data["identificationCode"] = get_text(ir_elem, "identificationCode")
        org_data["registrationDate"] = get_text(ir_elem, "registrationDate")
        org_data["lastUpdateDate"] = get_text(ir_elem, "lastUpdateDate")

        # Name (nested in name/originalName)
        name_elem = ir_elem.find("name")
        if name_elem is not None:
            org_data["name"] = get_text(name_elem, "originalName")

        org_data["acronym"] = get_text(ir_elem, "acronym")
        org_data["entityForm"] = get_text(ir_elem, "entityForm")
        org_data["webSiteURL"] = get_text(ir_elem, "webSiteURL")
        org_data["registrationCategory"] = get_text(ir_elem, "registrationCategory")
        org_data["goals"] = get_text(ir_elem, "goals")
        org_data["interestRepresented"] = get_text(ir_elem, "interestRepresented")

        # Head office
        head_office = ir_elem.find("headOffice")
        if head_office is not None:
            org_data["headOffice"] = {
                "address": get_text(head_office, "address"),
                "postCode": get_text(head_office, "postCode"),
                "city": get_text(head_office, "city"),
                "country": get_text(head_office, "country"),
            }

        # EU Office
        eu_office = ir_elem.find("EUOffice")
        if eu_office is not None:
            org_data["EUOffice"] = {
                "address": get_text(eu_office, "address"),
                "postCode": get_text(eu_office, "postCode"),
                "city": get_text(eu_office, "city"),
                "country": get_text(eu_office, "country"),
            }

        # Levels of interest
        levels_elem = ir_elem.find("levelsOfInterest")
        if levels_elem is not None:
            levels = []
            for level_wrapper in levels_elem.findall("levelOfInterest"):
                level_text = get_text(level_wrapper, "levelOfInterest")
                if level_text:
                    levels.append(level_text)
            if levels:
                org_data["levelsOfInterest"] = levels

        # Interests (policy areas)
        interests_elem = ir_elem.find("interests")
        if interests_elem is not None:
            interests = []
            for interest in interests_elem.findall("interest"):
                interest_name = get_text(interest, "name")
                if interest_name:
                    interests.append(interest_name)
            if interests:
                org_data["interests"] = interests

        # Members info
        members_elem = ir_elem.find("members")
        if members_elem is not None:
            org_data["members"] = {
                "members100Percent": get_text(members_elem, "members100Percent"),
                "members50Percent": get_text(members_elem, "members50Percent"),
                "totalMembers": get_text(members_elem, "members"),
                "membersFTE": get_text(members_elem, "membersFTE"),
            }

        # EP accredited number
        org_data["EPAccreditedNumber"] = get_text(ir_elem, "EPAccreditedNumber")

        # Communication activities
        org_data["communicationActivities"] = get_text(ir_elem, "communicationActivities")

        # EU Legislative Proposals
        org_data["EULegislativeProposals"] = get_text(ir_elem, "EULegislativeProposals")

        if org_data.get("identificationCode") or org_data.get("name"):
            organizations.append(org_data)

    if logger:
        logger.info(f"Parsed {len(organizations)} organizations from XML API")

    return organizations
