"""Utilities and scrapers for European Commissioners."""

import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.core.pdf import extract_text_from_pdf


def clean_text(text: Optional[str]) -> str:
    """Clean whitespace from text."""
    if not text:
        return ""
    # Replace unicode non-breaking spaces
    text = text.replace("\xa0", " ")
    # Replace multiple whitespace characters with single space
    return re.sub(r"\s+", " ", text).strip()


def parse_date_string(date_str: Optional[str]) -> Optional[str]:
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return None

    try:
        date_str = clean_text(date_str)

        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")

        if re.match(r"\d{2}/\d{2}/\d{4}", date_str):
            dt = datetime.strptime(date_str, "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d")

        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str

        formats = [
            "%Y-%B-%d",
            "%d %B %Y",
            "%d %b %Y",
            "%B %d, %Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return date_str
    except Exception:
        return date_str


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def fetch_html(url: str, logger: Optional[Any] = None) -> str:
    """Fetch HTML content with retry."""
    response = requests.get(
        url,
        headers={"User-Agent": "Parl8-Pipeline/1.0 (EU Parliament Transparency Project)"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def extract_section_by_text(soup: BeautifulSoup, section_text: str) -> Optional[BeautifulSoup]:
    """Find a section by its heading text."""
    heading = soup.find("h2", id=section_text.lower())
    if not heading:
        heading = soup.find("h2", string=lambda text: text and section_text.lower() in text.lower())

    if not heading:
        return None

    section_container = heading.find_parent("div", class_="ecl-u-mb-2xl")
    if not section_container:
        section_container = heading.find_parent(
            "div", class_=lambda c: c and ("section" in c or "related-links" in c)
        )
    if not section_container:
        section_container = heading.find_parent("div")

    return section_container


def extract_responsibilities(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Extract the Responsibilities section."""
    section = extract_section_by_text(soup, "Responsibilities")
    if not section:
        priorities_banner = soup.find(id=re.compile(r"ecl-banner.*title"))
        if priorities_banner:
            title = priorities_banner.get_text(strip=True)
            if "Priorities" in title:
                return {"Priorities": [title]}
        return None

    responsibilities = {}
    subsections = section.find_all("h3")
    for h3 in subsections:
        subsection_title = h3.get_text(strip=True)
        content_parts = []
        for sibling in h3.find_next_siblings():
            if sibling.name == "h3":
                break
            if sibling.name == "p":
                content_parts.append(sibling.get_text(strip=True))
            elif sibling.name == "ul":
                items = [li.get_text(strip=True) for li in sibling.find_all("li")]
                content_parts.extend(items)
        if content_parts:
            responsibilities[subsection_title] = content_parts

    return responsibilities if responsibilities else None


def extract_contacts(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Extract the Contacts section."""
    section = extract_section_by_text(soup, "Contacts")
    if not section:
        return None

    contacts = {}
    contact_blocks = section.find_all("div", class_="ecl-content-block")
    for block in contact_blocks:
        title_elem = block.find("div", class_="ecl-content-block__title")
        if title_elem:
            link = title_elem.find("a")
            if link:
                title = link.get_text(strip=True)
                url = link.get("href", "")
                if url.startswith("/"):
                    url = f"https://commission.europa.eu{url}"
                desc_elem = block.find("div", class_="ecl-content-block__description")
                description = desc_elem.get_text(strip=True) if desc_elem else None
                contacts[title] = {"url": url, "description": description}

    return contacts if contacts else None


def extract_speeches(soup: BeautifulSoup) -> Optional[List[Dict[str, Any]]]:
    """Extract the Speeches section."""
    section = extract_section_by_text(soup, "Speeches")
    if not section:
        section = extract_section_by_text(soup, "Statements and speeches")
    if not section:
        return None

    speeches = []
    speech_blocks = section.find_all("article", class_="ecl-content-item")
    for block in speech_blocks:
        speech = {}
        time_elem = block.find("time")
        if time_elem:
            raw_date = time_elem.get("datetime")
            speech["date"] = parse_date_string(raw_date)
        title_elem = block.find("div", class_="ecl-content-block__title")
        if title_elem:
            link = title_elem.find("a")
            if link:
                speech["title"] = link.get_text(strip=True)
                speech["url"] = link.get("href", "")
        meta_items = block.find_all("li", class_="ecl-content-block__primary-meta-item")
        if meta_items:
            speech["type"] = meta_items[0].get_text(strip=True)
        if speech:
            speeches.append(speech)

    return speeches if speeches else None


def extract_latest_news(soup: BeautifulSoup) -> Optional[List[Dict[str, Any]]]:
    """Extract the Latest news section."""
    section = extract_section_by_text(soup, "Latest")
    if not section:
        section = extract_section_by_text(soup, "News")
    if not section:
        return None

    news_items = []
    news_blocks = section.find_all("article", class_="ecl-content-item")
    for block in news_blocks:
        item = {}
        time_elem = block.find("time")
        if time_elem:
            raw_date = time_elem.get("datetime")
            item["date"] = parse_date_string(raw_date)
        title_elem = block.find("div", class_="ecl-content-block__title")
        if title_elem:
            link = title_elem.find("a")
            if link:
                item["title"] = link.get_text(strip=True)
                item["url"] = link.get("href", "")
        meta_items = block.find_all("li", class_="ecl-content-block__primary-meta-item")
        if meta_items:
            item["type"] = meta_items[0].get_text(strip=True)
        if item:
            news_items.append(item)

    return news_items if news_items else None


def extract_transparency(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Extract the Transparency section."""
    section = extract_section_by_text(soup, "Transparency")
    transparency = {}
    if section:
        links = section.find_all("a", href=True)
        for link in links:
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "transparencyinitiative/meetings" in href:
                if "meetings" not in transparency:
                    transparency["meetings"] = []
                transparency["meetings"].append({"title": text, "url": href})
            elif "transparencyinitiative/meetings/mission" in href:
                if "missions" not in transparency:
                    transparency["missions"] = []
                transparency["missions"].append({"title": text, "url": href})

    if not transparency:
        transparency_link = soup.find(
            "a", string=lambda t: t and "Transparency" == t.strip(), href=True
        )
        if transparency_link:
            href = transparency_link.get("href", "")
            if href.startswith("/"):
                href = f"https://commission.europa.eu{href}"
            if "meetings" not in transparency:
                transparency["meetings"] = []
            transparency["meetings"].append({"title": "Transparency Register", "url": href})

    return transparency if transparency else None


def extract_biography(soup: BeautifulSoup) -> Optional[List[Dict[str, str]]]:
    """Extract the Biography timeline."""
    section = extract_section_by_text(soup, "Biography")
    if not section:
        section = extract_section_by_text(soup, "About the President")
    if not section:
        return None

    timeline_items = []
    timeline = section.find("ol", class_="ecl-timeline")
    if timeline:
        items = timeline.find_all("li", class_="ecl-timeline__item")
        for item in items:
            if "ecl-timeline__item--toggle" in item.get("class", []):
                continue
            tooltip = item.find("div", class_="ecl-timeline__tooltip")
            if tooltip:
                label_elem = tooltip.find("div", class_="ecl-timeline__label")
                content_elem = tooltip.find("div", class_="ecl-timeline__content")
                if label_elem and content_elem:
                    timeline_items.append(
                        {
                            "period": label_elem.get_text(strip=True),
                            "position": content_elem.get_text(strip=True),
                        }
                    )
    else:
        bio_link = section.find("a", href=True, string=lambda t: t and "Biography" in t)
        if bio_link:
            url = bio_link.get("href")
            if url.startswith("/"):
                url = f"https://commission.europa.eu{url}"
            try:
                sub_html = fetch_html(url, None)
                sub_soup = BeautifulSoup(sub_html, "html.parser")
                sub_timeline = sub_soup.find("ol", class_="ecl-timeline")
                if sub_timeline:
                    items = sub_timeline.find_all("li", class_="ecl-timeline__item")
                    for item in items:
                        if "ecl-timeline__item--toggle" in item.get("class", []):
                            continue
                        tooltip = item.find("div", class_="ecl-timeline__tooltip")
                        if tooltip:
                            label = tooltip.find("div", class_="ecl-timeline__label")
                            content = tooltip.find("div", class_="ecl-timeline__content")
                            if label and content:
                                timeline_items.append(
                                    {
                                        "period": label.get_text(strip=True),
                                        "position": content.get_text(strip=True),
                                    }
                                )
            except Exception:
                pass

    return timeline_items if timeline_items else None


def extract_documents(soup: BeautifulSoup) -> Optional[List[Dict[str, Any]]]:
    """Extract the Documents section."""
    section = extract_section_by_text(soup, "Documents")
    if not section:
        related = extract_section_by_text(soup, "Related links")
        if related:
            guidelines = related.find("a", string=lambda t: t and "Political Guidelines" in t)
            if guidelines:
                url = guidelines.get("href", "")
                if url.startswith("/"):
                    url = f"https://commission.europa.eu{url}"
                return [
                    {
                        "title": "Political Guidelines for the next European Commission 2024-2029",
                        "url": url,
                        "date": "2024-07-18",
                        "language": "EN",
                    }
                ]
    if not section:
        return None

    documents = []
    doc_blocks = section.find_all("div", class_="ecl-file")
    for block in doc_blocks:
        doc = {}
        title_elem = block.find("div", class_="ecl-file__title")
        if title_elem:
            doc["title"] = title_elem.get_text(strip=True)
        meta_items = block.find_all("li", class_="ecl-file__detail-meta-item")
        if meta_items:
            raw_date = meta_items[0].get_text(strip=True)
            doc["date"] = parse_date_string(raw_date)
        lang_elem = block.find("div", class_="ecl-file__language")
        if lang_elem:
            doc["language"] = lang_elem.get_text(strip=True)
        meta_elem = block.find("div", class_="ecl-file__meta")
        if meta_elem:
            doc["file_info"] = meta_elem.get_text(strip=True)
        download_link = block.find("a", class_="ecl-file__download")
        if download_link:
            url = download_link.get("href", "")
            if url.startswith("/"):
                url = f"https://commission.europa.eu{url}"
            doc["url"] = url
        if doc:
            documents.append(doc)

    return documents if documents else None


def extract_team_page_url(soup: BeautifulSoup) -> Optional[str]:
    """Extract the team page URL."""
    contacts_section = extract_section_by_text(soup, "Contacts")
    if not contacts_section:
        contacts_section = extract_section_by_text(soup, "Related links")
    if not contacts_section:
        contacts_section = extract_section_by_text(soup, "About the President")

    if contacts_section:
        team_link = contacts_section.find("a", href=lambda x: x and "team" in x.lower())
        if team_link:
            href = team_link.get("href", "")
            if href.startswith("/"):
                return f"https://commission.europa.eu{href}"
            return href
    return None


def fetch_and_parse_team(team_url: str, logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Fetch and parse cabinet/team members."""
    CABINET_ROLES = [
        "HEAD OF CABINET",
        "DEPUTY HEAD OF CABINET",
        "CABINET EXPERT",
        "MEMBER",
        "COMMUNICATION ADVISER",
        "POLICY ADVISER",
        "ASSISTANT",
    ]

    try:
        html = fetch_html(team_url, logger)
        soup = BeautifulSoup(html, "html.parser")
        team_members: List[Dict[str, Any]] = []
        main_content = soup.find("main") or soup.find("article") or soup
        text_content = main_content.get_text("\n", strip=True)
        lines = [line.strip() for line in text_content.split("\n") if line.strip()]

        current_member: Optional[Dict[str, Any]] = None
        for i, line in enumerate(lines):
            line_upper = line.upper()
            is_role_line = any(role in line_upper for role in CABINET_ROLES)

            if is_role_line and not line_upper.startswith("RESPONSIBILITIES"):
                if i > 0 and current_member is None:
                    name = lines[i - 1]
                    if len(name) > 2 and not name.startswith(("Email", "Phone", "Postal")):
                        current_member = {"name": name, "role": line}
                        team_members.append(current_member)
            elif current_member and line == "Responsibilities" and i + 1 < len(lines):
                responsibilities = []
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    if any(role in next_line.upper() for role in CABINET_ROLES):
                        break
                    if next_line in ("Email:", "Phone number:", "Postal address"):
                        break
                    responsibilities.append(next_line)
                if responsibilities:
                    current_member["responsibilities"] = "; ".join(responsibilities[:5])
                current_member = None

        seen_names: set = set()
        unique_members = []
        for member in team_members:
            name = member.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                unique_members.append(member)
        return unique_members
    except Exception:
        return []


def extract_declarations_from_documents(
    documents: List[Dict[str, Any]], logger: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Extract and enrich declaration documents."""
    declarations = []
    for doc in documents:
        title = doc.get("title", "").lower()
        if "declaration" in title and ("interest" in title or "interests" in title):
            declaration = doc.copy()
            declaration["declaration_type"] = "interests"
            if doc.get("url"):
                text_content = extract_text_from_pdf(doc["url"], logger)
                if text_content:
                    declaration["text_content"] = text_content
            declarations.append(declaration)
    return declarations


def extract_numeric_id(soup: BeautifulSoup) -> Optional[str]:
    """Extract the numeric ID used for Press Corner API."""
    link = soup.find("a", href=re.compile(r"presscorner.*commissioner=(\d+)"))
    if link:
        m = re.search(r"commissioner=(\d+)", link["href"])
        if m:
            return m.group(1)
    return None


def fetch_press_corner_items(
    numeric_id: str,
    filter_type: str,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Fetch items from Press Corner RSS API."""
    items = []
    url = f"https://ec.europa.eu/commission/presscorner/api/rss?search?language=en&commissioner={numeric_id}&pagesize=50"
    try:
        response = requests.get(url, headers={"User-Agent": "Parl8-Pipeline/1.0"}, timeout=20)
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
        channel = root.find("channel")
        if not channel:
            return []

        for item in channel.findall("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            date_el = item.find("pubDate")

            if title_el is not None and link_el is not None and date_el is not None:
                link_text = link_el.text or ""
                is_speech = "/speech_" in link_text or "/speech/" in link_text
                is_news = any(p in link_text for p in ["/ip_", "/statement_", "/ac_", "/mex_"])

                if filter_type == "SPEECH" and not is_speech:
                    continue
                if filter_type == "NEWS" and not is_news:
                    continue

                raw_date = date_el.text
                final_date = None
                try:
                    from email.utils import parsedate_to_datetime

                    final_date = parsedate_to_datetime(raw_date).strftime("%Y-%m-%d")
                except Exception:
                    final_date = raw_date

                desc_text = desc_el.text if desc_el is not None else ""
                items.append(
                    {
                        "title": title_el.text,
                        "url": link_text,
                        "date": final_date,
                        "description": (
                            desc_text[:500] + "..." if len(desc_text) > 500 else desc_text
                        ),
                        "type": "Speech" if is_speech else "News",
                    }
                )
        return items
    except Exception:
        return []


def extract_com_id_from_calendar_url(soup: BeautifulSoup) -> Optional[str]:
    """Extract COM_xxx ID from calendar link."""
    link = soup.find("a", href=re.compile(r"political-leader/COM_"))
    if link:
        m = re.search(r"(COM_[0-9A-F]+)", link["href"])
        if m:
            return m.group(1)
    return None


def fetch_calendar_items(com_id: str, logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Fetch calendar items."""
    import urllib.parse

    encoded_core = f"http://publications.europa.eu/resource/authority/political-leader/{com_id}"
    base_url = "https://commission.europa.eu/about/organisation/college-commissioners/calendar-items-president-and-commissioners_en"
    encoded_com = urllib.parse.quote(encoded_core, safe="")
    final_url = f"{base_url}?f[0]=commissioner_dynamic_commissioner_dynamic%3A{encoded_com}"

    try:
        html = fetch_html(final_url, logger)
        soup = BeautifulSoup(html, "html.parser")
        events = []
        articles = soup.find_all("article", class_="ecl-content-item")
        for art in articles:
            date_block = art.find("time", class_="ecl-date-block")
            final_date = None
            if date_block:
                day = date_block.find("span", class_="ecl-date-block__day")
                month = date_block.find("abbr", class_="ecl-date-block__month")
                year = date_block.find("span", class_="ecl-date-block__year")
                if day and month and year:
                    d_str = f"{day.get_text(strip=True)} {month.get_text(strip=True)} {year.get_text(strip=True)}"
                    final_date = parse_date_string(d_str)
            title_div = art.find("div", class_="ecl-content-block__title")
            title = title_div.get_text(strip=True) if title_div else "No title"
            loc_span = art.find("span", attrs={"translate": "no"})
            location = loc_span.get_text(strip=True) if loc_span else None
            if final_date:
                events.append(
                    {
                        "date": final_date,
                        "title": title,
                        "location": location,
                        "type": "Calendar Item",
                    }
                )
        return events
    except Exception:
        return []


def extract_transparency_uuid(soup: BeautifulSoup) -> Optional[str]:
    """Extract UUID for meetings."""
    link = soup.find("a", href=re.compile(r"transparencyinitiative.*host=[a-f0-9-]+"))
    if link:
        m = re.search(r"host=([a-f0-9-]+)", link["href"])
        if m:
            return m.group(1)
    return None


def fetch_meetings_excel(uuid: str, logger: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Fetch and parse meetings Excel file."""
    url = f"https://ec.europa.eu/transparencyinitiative/meetings/exportmeetings.do?host={uuid}"
    try:
        r = requests.get(url, headers={"User-Agent": "Parl8-Pipeline/1.0"}, timeout=30)
        if r.status_code != 200:
            return []
        with io.BytesIO(r.content) as f:
            df = pd.read_excel(f, header=1)
        meetings = []
        for _, row in df.iterrows():
            try:
                date_val = row.get("Date of meeting")
                location = row.get("Location")
                actors = row.get("Interest representative(s) met")
                subject = row.get("Subject(s)")
                fmt_date = (
                    date_val.strftime("%Y-%m-%d")
                    if isinstance(date_val, datetime)
                    else parse_date_string(date_val)
                )
                if fmt_date and subject:
                    meetings.append(
                        {
                            "date": fmt_date,
                            "title": subject,
                            "location": location if pd.notna(location) else None,
                            "organizations": actors.split(", ") if pd.notna(actors) else [],
                            "type": "Meeting",
                        }
                    )
            except Exception:
                continue
        return meetings
    except Exception:
        return []
