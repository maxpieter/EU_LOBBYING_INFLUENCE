"""This pipeline is used to scrape MEP data from the Europarl website."""

import concurrent.futures
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import nltk
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from pipeline.models.members import Member

# Suppress Chrome/absl warnings
os.environ["WDM_LOG_LEVEL"] = "0"
os.environ["WDM_PRINT_FIRST_LINE"] = "False"
os.environ["ABSL_LOGGING_MIN_LEVEL"] = "2"  # Suppress absl warnings

# Performance Configuration
PERFORMANCE_CONFIG = {
    "browser_pool_size": 3,  # Number of browser instances to keep in pool
    "max_workers": 10,  # Maximum concurrent threads for MEP processing
    "batch_size": 50,  # Number of MEPs to process in each batch
    "speech_batch_size": 10,  # Number of MEPs to process for speech summarization
    "page_load_timeout": 30,  # Seconds to wait for page load
    "implicit_wait": 10,  # Seconds to wait for elements
    "retry_attempts": 3,  # Number of retry attempts for failed requests
    "delay_between_batches": 1,  # Seconds to wait between batches
    "memory_cleanup_interval": 50,  # Force GC every N MEPs
}

# Download NLTK stopwords if not already downloaded
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass  # Continue even if nltk download fails


# ––– BROWSER POOL CLASS –––#
class BrowserPool:
    """Manages a pool of Chrome browser instances for efficient scraping."""

    def __init__(self, pool_size: int = None):
        self.pool_size = pool_size or PERFORMANCE_CONFIG["browser_pool_size"]
        self.drivers = queue.Queue()
        self.lock = threading.Lock()
        self._initialize_pool()

    def _create_driver(self):
        """Create a new Chrome driver instance."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-logging")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-images")
        # Add these arguments to prevent DevTools messages
        options.add_argument("--disable-dev-tools")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-translate")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--mute-audio")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-hang-monitor")
        options.add_argument("--disable-prompt-on-repost")
        options.add_argument("--disable-client-side-phishing-detection")
        options.add_argument("--disable-component-extensions-with-background-pages")
        options.add_argument("--disable-background-downloads")
        options.add_argument("--disable-add-to-shelf")
        options.add_argument("--disable-client-side-phishing-detection")
        options.add_argument("--disable-datasaver-prompt")
        options.add_argument("--disable-desktop-notifications")
        options.add_argument("--disable-domain-reliability")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-prompt-on-repost")
        options.add_argument("--disable-sync-preferences")
        options.add_argument("--disable-web-resources")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-features=BlinkGenPropertyTrees")
        options.add_argument("--disable-features=EnableDrDc")
        options.add_argument("--disable-features=UseChromeOSDirectVideoDecoder")
        options.add_argument("--disable-features=VaapiVideoDecoder")
        options.add_argument("--disable-features=VaapiVideoEncoder")
        options.add_argument("--disable-features=WebRtcHideLocalIpsWithMdns")
        options.add_argument("--disable-features=WebRtcUseMinMaxVEADimensions")
        options.add_argument("--disable-features=WebRtcUseMinMaxVEADimensions")
        options.add_argument("--disable-features=WebRtcUseMinMaxVEADimensions")

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )
            driver.set_page_load_timeout(PERFORMANCE_CONFIG["page_load_timeout"])
            driver.implicitly_wait(PERFORMANCE_CONFIG["implicit_wait"])
            return driver
        except Exception as e:
            print(f"Failed to create Chrome driver: {e}")
            raise

    def _initialize_pool(self):
        """Initialize the browser pool."""
        for _ in range(self.pool_size):
            try:
                driver = self._create_driver()
                # Set page load timeout to prevent hanging
                driver.set_page_load_timeout(30)
                driver.implicitly_wait(10)
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


def performance_monitor(func):
    """Decorator to monitor function performance."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


# ––– SPEECH EXCERPT CLASS –––#
class SpeechExcerpt:
    """Container for speech text with citation information."""

    def __init__(self, text: str, ref: str, url: str, date: str = "Unknown"):
        self.text = text
        self.ref = ref  # e.g., "[1]", "[2]", "[3]"
        self.url = url  # Europarl video URL
        self.date = date  # Speech date


# ––– CONSTANTS –––#
MEP_LIST_URL = "https://www.europarl.europa.eu/meps/en/full-list/xml"
BASE_URL = "https://www.europarl.europa.eu"

# ––– SCRAPING LOGIC –––#


def _clean_text(text: str) -> str:
    """Clean text by removing invalid UTF-8 characters.

    Args:
        text: Input text that may contain invalid UTF-8 bytes

    Returns:
        Cleaned text safe for UTF-8 encoding
    """
    if not text:
        return text
    # Encode to UTF-8 with error handling, then decode back
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _safe_console_text(text: str) -> str:
    """Make text safe for console output by replacing problematic Unicode characters.

    Args:
        text: Text that may contain Unicode characters not supported by console

    Returns:
        ASCII-safe text for console output
    """
    if not text:
        return text
    # Encode to ASCII with xmlcharrefreplace to show &#...; for special chars
    # or use 'replace' to replace with '?'
    return text.encode("ascii", errors="ignore").decode("ascii")


def _fetch_mep_list_xml(logger):
    logger.info("Fetching list of MEPs...")
    response = requests.get(MEP_LIST_URL)
    response.raise_for_status()
    return response.content


def _parse_meps_from_xml(xml_content: bytes) -> List[Member]:
    soup = BeautifulSoup(xml_content, "xml")
    meps = []
    for mep_tag in soup.find_all("mep"):
        mep_id = str(mep_tag.find("id").text)

        # Helper to safely extract text from XML elements
        def safe_text(field_name: str) -> Optional[str]:
            element = mep_tag.find(field_name)
            return element.text if element else None

        meps.append(
            Member(
                mepid=mep_id,
                full_name=safe_text("fullName") or "Unknown",
                country=safe_text("country") or "Unknown",
                political_group=safe_text("politicalGroup") or "Unknown",
                national_party=safe_text("nationalPoliticalGroup"),
                profile_url=f"{BASE_URL}/meps/en/{mep_id}",
                image_url=f"https://www.europarl.europa.eu/mepphoto/{mep_id}.jpg",
            )
        )
    return meps


def _fetch_mep_profile_page(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def _safe_get_text(tag, strip=True):
    """Safely extract text from HTML tag and clean invalid UTF-8 characters."""
    if not tag:
        return None
    text = tag.get_text(strip=strip)
    return _clean_text(text) if text else None


def _safe_get_attr(tag, attr):
    return tag.get(attr) if tag else None


def _parse_mep_profile(mep: Member, html_content: str) -> Member:
    soup = BeautifulSoup(html_content, "html.parser")
    mep.role = _safe_get_text(soup.find("p", class_="sln-political-group-role"))
    mep.birth_date = _safe_get_attr(soup.find("time", class_="sln-birth-date"), "datetime")
    mep.birth_place = _safe_get_text(soup.find("span", class_="sln-birth-place"))

    socials_div = soup.find("div", class_="erpl_social-share-horizontal")
    if socials_div:
        for a_tag in socials_div.find_all("a", href=True):
            title = a_tag.get("data-original-title", "").strip()
            href = a_tag["href"]
            if title and title.lower() not in ["e-mail"]:
                mep.socials[title] = href

    nav_accordion = soup.find("div", id="erplAccordion")
    if nav_accordion:
        for link in nav_accordion.find_all("a", href=True):
            if not link["href"].startswith("#"):
                text_span = link.find("span", class_="t-x")
                text = _safe_get_text(text_span)
                if text and link["href"]:
                    mep.navigation_links[text] = urljoin(BASE_URL, link["href"])

    status_list_div = soup.find("div", class_="erpl_meps-status-list")
    if status_list_div:
        for status_div in status_list_div.find_all("div", class_="erpl_meps-status"):
            role_title = _safe_get_text(status_div.find("h4", class_="erpl_title-h4"))
            if role_title:
                for badge_div in status_div.find_all("div", class_="badges"):
                    acronym = _safe_get_text(badge_div.find("a", class_="erpl_badge"))
                    full_name = _safe_get_text(badge_div.find("div", class_="erpl_committee"))
                    link = _safe_get_attr(badge_div.find("a", class_="erpl_badge"), "href")
                    if acronym and full_name and link:
                        mep.committees.append(
                            {
                                "role": role_title,
                                "name": full_name,
                                "acronym": acronym,
                                "link": urljoin(BASE_URL, link),
                            }
                        )

    contact_section = soup.find("section", id="contacts")
    if contact_section:
        for card in contact_section.find_all("div", class_="erpl_contact-card"):
            location = _safe_get_text(card.find("div", class_="erpl_title-h3"))
            address_span = card.find("div", class_="erpl_contact-card-list").find("span")
            address = " ".join(address_span.stripped_strings) if address_span else None
            phone_tag = card.find("a", href=lambda href: href and href.startswith("tel:"))
            phone = _safe_get_text(phone_tag)
            if location:
                mep.contacts.append(
                    {
                        "location": location,
                        "address": address,
                        "phone": phone,
                    }
                )
    return mep


def _parse_cv_page(html_content: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, "html.parser")
    cv_entries = []
    for section in soup.find_all("div", class_="erpl_meps-activity"):
        category = _safe_get_text(section.find("h4", class_="erpl_title-h4"))
        if category:
            for li in section.find_all("li"):
                period_tag = li.find("strong")
                period, description = (None, _safe_get_text(li))
                if period_tag:
                    period = _safe_get_text(period_tag)
                    period_tag.extract()
                    description = _safe_get_text(li).lstrip(": ").strip()
                if description:
                    cv_entries.append(
                        {
                            "category": category,
                            "period": period,
                            "description": description,
                        }
                    )
    return cv_entries


def _fetch_and_parse_cv(mep: Member):
    cv_url = mep.navigation_links.get("Curriculum vitae")
    if cv_url:
        try:
            response = requests.get(cv_url)
            response.raise_for_status()
            mep.cv = _parse_cv_page(response.text)
        except requests.RequestException as e:
            tqdm.write(
                f"Error fetching CV for {_safe_console_text(mep.full_name)}: {e}",
                file=sys.stderr,
            )


def _parse_assistants_page(html_content: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, "html.parser")
    assistants = []
    container = soup.find("div", class_="erpl_type-assistants-list")
    if container:
        for section in container.find_all("div", class_="erpl_type-assistants"):
            assistant_type = _safe_get_text(section.find("h4", class_="erpl_title-h4"))
            if assistant_type:
                for item in section.find_all("div", class_="erpl_type-assistants-item"):
                    name = " ".join(
                        _safe_get_text(item.find("span", class_="erpl_assistant")).split()
                    )
                    if name:
                        assistants.append({"name": name, "type": assistant_type})
    return assistants


def _fetch_and_parse_assistants(mep: Member):
    assistants_url = mep.navigation_links.get("Assistants")
    if assistants_url:
        try:
            response = requests.get(assistants_url)
            response.raise_for_status()
            mep.assistants = _parse_assistants_page(response.text)
        except requests.RequestException as e:
            tqdm.write(
                f"Error fetching assistants for {_safe_console_text(mep.full_name)}: {e}",
                file=sys.stderr,
            )


def _parse_declarations_page(html_content: str) -> List[Dict[str, Any]]:
    """Parse declarations from MEP profile page.

    Bronze stage: Extract metadata only (title, date, URL, type).
    PDF text extraction is deferred to Gold stage for on-demand processing.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    declarations = []
    for section in soup.find_all("div", class_="erpl_meps-declaration"):
        declaration_type = _safe_get_text(section.find("h4", class_="erpl_title-h4")) or "General"
        for li in section.find_all("li"):
            link = li.find("a", href=lambda href: href and ".pdf" in href)
            if link:
                full_title = " ".join(_safe_get_text(link.find("span", class_="t-x")).split())
                date_match = re.search(r"(\d{2}-\d{2}-\d{4})", full_title)
                declarations.append(
                    {
                        "title": re.sub(r"\s*\(\d+\s*KB\)\s*$", "", full_title).strip(),
                        "declaration_type": declaration_type,
                        "date": date_match.group(1) if date_match else None,
                        "url": urljoin(BASE_URL, link["href"]),
                        # Note: text_content extraction removed - happens in Gold stage
                    }
                )
    return declarations


def _fetch_and_parse_declarations(mep: Member):
    declarations_url = mep.navigation_links.get("Declarations")
    if declarations_url:
        try:
            response = requests.get(declarations_url)
            response.raise_for_status()
            mep.declarations = _parse_declarations_page(response.text)
        except requests.RequestException as e:
            tqdm.write(
                f"Error fetching declarations for {_safe_console_text(mep.full_name)}: {e}",
                file=sys.stderr,
            )


def _enrich_mep_data(mep: Member, logger: Optional[Any] = None) -> Optional[Member]:
    try:
        if not mep.profile_url:
            tqdm.write(
                f"No profile URL for {_safe_console_text(mep.full_name)}. Skipping.",
                file=sys.stderr,
            )
            return None
        profile_html = _fetch_mep_profile_page(mep.profile_url)
        enriched_mep = _parse_mep_profile(mep, profile_html)
        _fetch_and_parse_cv(enriched_mep)
        _fetch_and_parse_assistants(enriched_mep)
        _fetch_and_parse_declarations(enriched_mep)
        return enriched_mep
    except Exception as e:
        tqdm.write(
            f"An error occurred while processing {_safe_console_text(mep.full_name)}: {e}",
            file=sys.stderr,
        )
        return None


def _log_scraping_summary(logger, meps: List[Member]):
    """Log Bronze scraping success metrics."""
    total_meps = len(meps)
    meps_with_cv = len([m for m in meps if m.cv])
    meps_with_declarations = len([m for m in meps if m.declarations])
    meps_with_assistants = len([m for m in meps if m.assistants])
    meps_with_committees = len([m for m in meps if m.committees])

    # Count active vs inactive
    active_meps = len([m for m in meps if m.status == "active"])
    inactive_meps = len([m for m in meps if m.status == "inactive"])

    logger.info("=" * 50)
    logger.info("BRONZE SCRAPING SUMMARY:")
    logger.info(f"  Total MEPs processed: {total_meps}")
    logger.info(f"  Active MEPs: {active_meps}")
    logger.info(f"  Inactive MEPs: {inactive_meps}")
    logger.info(f"  MEPs with CV: {meps_with_cv} ({meps_with_cv/total_meps*100:.1f}%)")
    logger.info(
        f"  MEPs with declarations: {meps_with_declarations} ({meps_with_declarations/total_meps*100:.1f}%)"
    )
    logger.info(
        f"  MEPs with assistants: {meps_with_assistants} ({meps_with_assistants/total_meps*100:.1f}%)"
    )
    logger.info(
        f"  MEPs with committees: {meps_with_committees} ({meps_with_committees/total_meps*100:.1f}%)"
    )

    # Total declarations count
    total_declarations = sum(len(m.declarations) for m in meps if m.declarations)
    logger.info(f"  Total declarations scraped: {total_declarations}")
    if meps_with_declarations > 0:
        logger.info(
            f"  Average declarations per MEP (with declarations): {total_declarations/meps_with_declarations:.1f}"
        )

    logger.info("=" * 50)


def _save_meps_to_json(logger, meps: List[Member], filename: Path):
    meps_dict = [mep.model_dump() for mep in meps]
    with open(filename, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(meps_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved data for {len(meps)} MEPs to {filename}.")

    # Log comprehensive scraping metrics
    _log_scraping_summary(logger, meps)

    return meps_dict


class MepStage:
    """Scrape and process MEP data."""

    def __init__(
        self,
        out_dir: Path,
        max_meps: int = None,
        max_workers: int = 10,
        batch_size: int = None,
        browser_pool_size: int = None,
        logger: Optional[Any] = None,
    ):
        self.out_dir = out_dir
        self.max_meps = max_meps
        self.max_workers = max_workers
        self.logger = logger

        # Update performance config with custom values if provided
        self.performance_config = PERFORMANCE_CONFIG.copy()
        if batch_size:
            self.performance_config["batch_size"] = batch_size
        if browser_pool_size:
            self.performance_config["browser_pool_size"] = browser_pool_size
        if max_workers:
            self.performance_config["max_workers"] = max_workers

    def run(self) -> List[Member]:
        """Process all MEPs and return the enriched list."""
        # Fetch current MEP list from XML
        xml_data = _fetch_mep_list_xml(self.logger)
        all_meps = _parse_meps_from_xml(xml_data)

        if not all_meps:
            self.logger.info("No MEPs found. Exiting.")
            return []

        meps_to_process = (
            all_meps[: self.max_meps]
            if self.max_meps is not None and self.max_meps > 0
            else all_meps
        )
        self.logger.info(
            f"Starting to process {len(meps_to_process)} out of {len(all_meps)} total MEPs."
        )

        # Process MEPs in batches to manage memory
        if not meps_to_process:
            self.logger.info("No MEPs to process. Exiting.")
            return []

        batch_size = min(self.performance_config["batch_size"], len(meps_to_process))
        enriched_meps = []

        for i in range(0, len(meps_to_process), batch_size):
            batch = meps_to_process[i : i + batch_size]
            self.logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(meps_to_process)-1)//batch_size + 1} ({len(batch)} MEPs)"
            )

            batch_enriched = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(
                    self.max_workers or self.performance_config["max_workers"],
                    len(batch),
                )
            ) as executor:
                future_to_mep = {
                    executor.submit(_enrich_mep_data, mep, self.logger): mep for mep in batch
                }

                progress_bar = tqdm(
                    concurrent.futures.as_completed(future_to_mep),
                    total=len(batch),
                    desc=f"Batch {i//batch_size + 1}",
                )

                for future in progress_bar:
                    mep = future_to_mep[future]
                    progress_bar.set_description(f"Processing {_safe_console_text(mep.full_name)}")
                    start = time.time()
                    try:
                        enriched_mep = future.result()
                        elapsed = time.time() - start
                        print(
                            f"Processed {_safe_console_text(mep.full_name)} in {elapsed:.1f} seconds"
                        )
                        if enriched_mep:
                            batch_enriched.append(enriched_mep)
                    except Exception as e:
                        tqdm.write(
                            f"An exception occurred for MEP {_safe_console_text(mep.full_name)}: {e}",
                            file=sys.stderr,
                        )

            # Add batch results to main list
            enriched_meps.extend(batch_enriched)

            # Force garbage collection after each batch
            import gc

            gc.collect()

            # Small delay to prevent overwhelming the server
            time.sleep(self.performance_config["delay_between_batches"])

        if enriched_meps:
            self.logger.info(f"Successfully scraped {len(enriched_meps)} MEPs.")
            _log_scraping_summary(self.logger, enriched_meps)
        else:
            self.logger.info("No MEP data was successfully enriched.")

        return enriched_meps


def run_mep_pipeline(
    out_dir: Path,
    max_meps: int = None,
    max_workers: int = 10,
    batch_size: int = None,
    browser_pool_size: int = None,
    logger: Optional[Any] = None,
) -> List[Member]:
    """Run the MEP data pipeline.

    Args:
        out_dir: Directory (unused, kept for backward compatibility).
        max_meps: Maximum number of MEPs to process.
        max_workers: Number of concurrent threads.
        batch_size: Number of MEPs to process in each batch.
        browser_pool_size: Number of browser instances to keep in pool.
        logger: Optional logger instance (Dagster logger recommended).

    Returns:
        List of enriched MEP objects.
    """
    try:
        mep_stage = MepStage(
            out_dir,
            max_meps,
            max_workers,
            batch_size,
            browser_pool_size,
            logger,
        )
        return mep_stage.run()
    finally:
        # Clean up browser pool
        global _browser_pool
        if _browser_pool:
            _browser_pool.close_all()
            _browser_pool = None


def scrape_all_meps(
    logger: Optional[Any] = None,
    max_meps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simplified wrapper for Bronze stage MEP scraping.

    Scrapes all MEP data and returns as list of dictionaries in Member model format.
    This is the entry point used by Dagster assets.

    Args:
        logger: Optional logger instance
        max_meps: Maximum number of MEPs to scrape (None = all)

    Returns:
        List of MEP dictionaries in unified Member model format
    """
    import tempfile

    # Use temporary directory for backward compatibility (unused now)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)

        # Run the pipeline and get results directly (no file I/O)
        enriched_meps = run_mep_pipeline(
            out_dir=out_path,
            max_meps=max_meps,
            logger=logger,
        )

        # Convert Member objects to dictionaries
        return [mep.model_dump() for mep in enriched_meps]
