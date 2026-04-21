"""Title enrichment for procedures via EP Open Data v2 detail endpoint.

The catalog asset only has access to the /procedures listing endpoint, which
never returns human-readable titles — only the procedure reference ID. This
module hits the detail endpoint (one call per procedure) to fetch the real
title stored under ``data[0].process_title.en``.

Used by eu_procedures_titles. Idempotent by design: the asset only feeds
rows whose ``title`` still equals their reference ID, so rerunning after a
full enrichment is a no-op.
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Optional

import requests

V2_DETAIL_URL = "https://data.europarl.europa.eu/api/v2/procedures/{process_id}"
HEADERS = {
    "Accept": "application/ld+json",
    "User-Agent": "Parl8/1.0",
}
TIMEOUT = 30
MAX_ATTEMPTS = 5
RETRY_BACKOFF_BASE = 2.0  # Seconds, exponential


def fetch_title(
    process_id: str,
    session: Optional[requests.Session] = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[Optional[str], str]:
    """Fetch a single procedure's real title.

    Returns ``(title, reason)`` where reason is one of:
    ``ok`` | ``empty_data`` | ``no_title`` | ``http_<code>`` | ``timeout`` | ``network`` | ``invalid_json``.
    Retries on 429/5xx with exponential backoff before giving up.
    """
    sess = session or requests
    url = V2_DETAIL_URL.format(process_id=process_id)

    for attempt in range(max_attempts):
        try:
            resp = sess.get(url, headers=HEADERS, timeout=TIMEOUT)
        except requests.Timeout:
            if attempt < max_attempts - 1:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            return None, "timeout"
        except requests.RequestException:
            if attempt < max_attempts - 1:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            return None, "network"

        # EP's Cloudflare front returns 403 when rate-limited (not 429 as expected),
        # so treat 403 as a retryable throttle too.
        if resp.status_code in (403, 429) or 500 <= resp.status_code < 600:
            if attempt < max_attempts - 1:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else RETRY_BACKOFF_BASE ** attempt
                time.sleep(sleep_s)
                continue
            return None, f"http_{resp.status_code}"
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"

        try:
            payload = resp.json()
        except ValueError:
            return None, "invalid_json"
        data = payload.get("data") or []
        if not data:
            return None, "empty_data"
        titles = data[0].get("process_title") or {}
        if not isinstance(titles, dict):
            return None, "no_title"
        title = titles.get("en") or next(
            (v for v in titles.values() if isinstance(v, str) and v.strip()),
            None,
        )
        if not title:
            return None, "no_title"
        return title, "ok"

    return None, "exhausted"


def fetch_titles_concurrent(
    process_ids: Iterable[str],
    workers: int = 1,
    request_interval: float = 0.5,
    logger: Optional[Any] = None,
) -> dict[str, str]:
    """Fetch titles for many procedure IDs with strict pacing + retries.

    Returns a mapping of ``process_id -> title`` containing only the IDs
    where a real title was recovered. Missing IDs are absent — the calling
    asset keeps their DB placeholder so the next run retries them.

    Cloudflare in front of the EP API throttles aggressively and returns
    403/404 under load (not 429). Observed yield: ~100% at 1 req / 500ms,
    ~15% at 10 workers. We pace globally, not per-worker.
    """
    _log = logger.info if logger else print
    ids = list(dict.fromkeys(process_ids))
    if not ids:
        return {}

    _log(
        f"Fetching {len(ids)} titles with {workers} worker(s), "
        f"{request_interval}s between requests"
    )
    results: dict[str, str] = {}
    reasons: Counter[str] = Counter()
    session = requests.Session()
    last_request_at = [0.0]  # List-wrapped so the closure can mutate it
    import threading
    pace_lock = threading.Lock()

    def _worker(pid: str) -> tuple[str, Optional[str], str]:
        with pace_lock:
            gap = time.monotonic() - last_request_at[0]
            if gap < request_interval:
                time.sleep(request_interval - gap)
            last_request_at[0] = time.monotonic()
        title, reason = fetch_title(pid, session=session)
        return pid, title, reason

    report_every = max(1, len(ids) // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, pid): pid for pid in ids}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, title, reason = fut.result()
            reasons[reason] += 1
            if title:
                results[pid] = title
            if i % report_every == 0:
                _log(
                    f"  {i}/{len(ids)} processed "
                    f"({len(results)} titled, reasons: {dict(reasons)})"
                )

    _log(
        f"Fetched {len(results)}/{len(ids)} titles "
        f"(failure reasons: {dict(reasons)})"
    )
    return results
