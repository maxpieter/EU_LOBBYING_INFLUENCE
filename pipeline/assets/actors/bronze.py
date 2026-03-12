"""EU Parliament actors extractor (Bronze layer).

Extracts EU institutional actors (Commissioners, Council members).
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from pipeline.assets.actors.commissioner_utils import (
    extract_biography,
    extract_com_id_from_calendar_url,
    extract_contacts,
    extract_declarations_from_documents,
    extract_documents,
    extract_latest_news,
    extract_numeric_id,
    extract_responsibilities,
    extract_speeches,
    extract_team_page_url,
    extract_transparency,
    extract_transparency_uuid,
    fetch_and_parse_team,
    fetch_calendar_items,
    fetch_html,
    fetch_meetings_excel,
    fetch_press_corner_items,
)

# Country mapping for commissioners (2024-2029 term)
COMMISSIONER_COUNTRIES: Dict[str, str] = {
    "Ursula von der Leyen": "Germany",
    "Teresa Ribera": "Spain",
    "Henna Virkkunen": "Finland",
    "Stéphane Séjourné": "France",
    "Kaja Kallas": "Estonia",
    "Raffaele Fitto": "Italy",
    "Roxana Mînzatu": "Romania",
    "Maroš Šefčovič": "Slovakia",
    "Valdis Dombrovskis": "Latvia",
    "Dubravka Šuica": "Croatia",
    "Olivér Várhelyi": "Hungary",
    "Wopke Hoekstra": "Netherlands",
    "Andrius Kubilius": "Lithuania",
    "Marta Kos": "Slovenia",
    "Jozef Síkela": "Czechia",
    "Dan Jørgensen": "Denmark",
    "Jessika Roswall": "Sweden",
    "Apostolos Tzitzikostas": "Greece",
    "Costas Kadis": "Cyprus",
    "Christophe Hansen": "Luxembourg",
    "Magnus Brunner": "Austria",
    "Michael McGrath": "Ireland",
    "Ekaterina Zaharieva": "Bulgaria",
    "Piotr Serafin": "Poland",
    "Hadja Lahbib": "Belgium",
    "Glenn Micallef": "Malta",
    "Maria Luís Albuquerque": "Portugal",
}


def enrich_commissioner_profile(
    commissioner: Dict[str, Any], logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Enrich commissioner data by scraping their profile page."""

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    url = commissioner.get("profile_url")
    if not url:
        return commissioner

    try:
        html = fetch_html(url, logger)
        soup = BeautifulSoup(html, "html.parser")

        # Extract role/portfolio
        page_header = soup.find("div", class_="ecl-page-header")
        if page_header:
            role_text = page_header.get_text(" ", strip=True).upper()
            if "EXECUTIVE VICE-PRESIDENT" in role_text:
                commissioner["role"] = "Executive Vice-President"
            elif "HIGH REPRESENTATIVE" in role_text:
                commissioner["role"] = "High Representative and Vice-President"
            elif "VICE-PRESIDENT" in role_text:
                commissioner["role"] = "Vice-President"
            elif "PRESIDENT" in role_text:
                commissioner["role"] = "President of the European Commission"

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc_content = meta_desc.get("content", "")
            if "responsible for" in desc_content.lower():
                match = re.search(r"responsible for ([^.]+)", desc_content, re.I)
                if match and not commissioner.get("portfolio"):
                    commissioner["portfolio"] = match.group(1).strip()

        # Set country from mapping
        name = commissioner.get("fullName", "")
        if not commissioner.get("country") and name in COMMISSIONER_COUNTRIES:
            commissioner["country"] = COMMISSIONER_COUNTRIES[name]

        # Contacts & Social
        if not commissioner.get("contacts"):
            commissioner["contacts"] = {}
        mailto_link = soup.find("a", href=lambda x: x and "mailto:" in x)
        if mailto_link:
            commissioner["contacts"]["email"] = mailto_link.get("href").replace("mailto:", "")

        social_media = {}
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "x.com" in href or "twitter.com" in href:
                social_media["x"] = href
            elif "linkedin.com" in href:
                social_media["linkedin"] = href
            elif "bsky.app" in href:
                social_media["bluesky"] = href
        if social_media:
            commissioner["contacts"]["social_media"] = social_media

        # Section Scraping
        responsibilities = extract_responsibilities(soup)
        if responsibilities:
            commissioner["responsibilities"] = responsibilities

        contacts = extract_contacts(soup)
        if contacts:
            commissioner["contacts"]["page_contacts"] = contacts

        # Press Corner News & Speeches
        numeric_id = extract_numeric_id(soup)
        if numeric_id:
            news = fetch_press_corner_items(numeric_id, "NEWS", logger)
            if news:
                commissioner["news"] = news
            speeches = fetch_press_corner_items(numeric_id, "SPEECH", logger)
            if speeches:
                commissioner["speeches"] = speeches
        else:
            # Fallback to HTML scraping for news/speeches if no ID found
            speeches = extract_speeches(soup)
            if speeches:
                commissioner["speeches"] = speeches
            news = extract_latest_news(soup)
            if news:
                commissioner["latest_news"] = news

        # Extract COM_ ID (required for all commissioners)
        com_id = extract_com_id_from_calendar_url(soup)
        if com_id:
            # CRITICAL: Set proper COM_ ID as actor_id
            commissioner["actor_id"] = com_id
            _log(f"Extracted COM ID: {com_id}")

            # Fetch calendar items using COM_ ID
            calendar = fetch_calendar_items(com_id, logger)
            if calendar:
                commissioner["calendar"] = calendar
        else:
            _log(f"WARNING: No COM ID found for {name}", "warning")

        # Meetings (Excel)
        transparency_uuid = extract_transparency_uuid(soup)
        if transparency_uuid:
            meetings = fetch_meetings_excel(transparency_uuid, logger)
            if meetings:
                commissioner["meetings"] = meetings

        # Documents & Team
        transparency = extract_transparency(soup)
        if transparency:
            commissioner["transparency"] = transparency

        biography = extract_biography(soup)
        if biography:
            commissioner["biography"] = biography

        documents = extract_documents(soup)
        if documents:
            commissioner["documents"] = documents
            declarations = extract_declarations_from_documents(documents, logger)
            if declarations:
                commissioner["declarations"] = declarations

        team_url = extract_team_page_url(soup)
        if team_url:
            team = fetch_and_parse_team(team_url, logger)
            if team:
                commissioner["team"] = team

        return commissioner
    except Exception as e:
        _log(f"Error enriching profile for {commissioner.get('fullName')}: {e}", "warning")
        return commissioner


def fetch_president(logger: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Scrape the President from dedicated page."""
    try:
        url = "https://commission.europa.eu/about/organisation/president_en"
        html = fetch_html(url, logger)
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        name = h1.get_text(strip=True) if h1 else "Ursula von der Leyen"

        # Extract COM_ ID from page (required)
        actor_id = None
        id_match = re.search(r"political-leader/(COM_[0-9A-Z]+)", html)
        if id_match:
            actor_id = id_match.group(1)
            if logger:
                logger.info(f"Extracted President COM ID: {actor_id}")
        else:
            if logger:
                logger.warning(f"No COM ID found for President {name}")
            return None  # President must have COM_ ID

        president = {
            "actor_id": actor_id,
            "fullName": name,
            "actor_type": "commissioner",
            "role": "President of the European Commission",
            "country": "Germany",
            "term_start": "2024-12-01",
            "portfolio": "Overall leadership",
            "profile_url": url,
            "contacts": {},
        }

        banner_img = soup.find("img", class_="ecl-banner__image")
        if banner_img and banner_img.get("src"):
            src = banner_img.get("src")
            president["image_url"] = (
                src if src.startswith("http") else f"https://commission.europa.eu{src}"
            )

        return president
    except Exception:
        return None


def fetch_commissioners(logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Scrape current European Commission."""
    try:
        url = "https://commission.europa.eu/about/organisation/college-commissioners_en"
        html = fetch_html(url, logger)
        soup = BeautifulSoup(html, "html.parser")
        EXCLUDED = [
            "calendar-items-president-and-commissioners_en",
            "former-colleges-commissioners_en",
        ]
        all_links = soup.find_all(
            "a", href=re.compile(r"/about/organisation/college-commissioners/[\w-]+_en$")
        )
        links = [
            link for link in all_links if not any(ex in link.get("href", "") for ex in EXCLUDED)
        ]

        commissioners = []
        for link in links:
            name = link.get_text(strip=True)
            if not name or len(name) < 3:
                continue
            href = link.get("href", "")
            profile_url = f"https://commission.europa.eu{href}" if href.startswith("/") else href

            image_url = None
            card = link.find_parent("article") or link.find_parent("li")
            if card:
                img = card.find("img")
                if img and img.get("src"):
                    src = img.get("src")
                    image_url = (
                        src if src.startswith("http") else f"https://commission.europa.eu{src}"
                    )

            commissioners.append(
                {
                    "actor_id": None,  # Will be set from COM_ ID during enrichment
                    "fullName": name,
                    "actor_type": "commissioner",
                    "profile_url": profile_url,
                    "image_url": image_url,
                    "term_start": "2024-12-01",  # Current commission term
                }
            )

        president = fetch_president(logger)
        if president:
            pres_article = soup.find(
                "article", attrs={"data-untranslated-label": "Ursula von der Leyen"}
            )
            if pres_article:
                img = pres_article.find("img")
                if img and img.get("src"):
                    src = img.get("src")
                    president["image_url"] = (
                        src if src.startswith("http") else f"https://commission.europa.eu{src}"
                    )
            commissioners.insert(0, president)

        # Enrich all commissioners (extracts COM_ IDs from profile pages)
        enriched = [enrich_commissioner_profile(c, logger) for c in commissioners]

        # Validate: Filter out any commissioners without proper COM_ IDs
        valid_commissioners = [
            c for c in enriched if c.get("actor_id") and c["actor_id"].startswith("COM_")
        ]

        if len(valid_commissioners) < len(enriched):
            missing = len(enriched) - len(valid_commissioners)
            if logger:
                logger.warning(f"Filtered out {missing} commissioners without valid COM_ IDs")

        return valid_commissioners
    except Exception:
        return []


def fetch_actors(
    date_range: Tuple[datetime, datetime],
    actor_type: Optional[str] = "all",
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Entry point for fetching institutional actors."""
    actors = []
    if actor_type in ["commissioner", "all"]:
        actors.extend(fetch_commissioners(logger))
    return actors
