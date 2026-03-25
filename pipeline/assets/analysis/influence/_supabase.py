"""Supabase helper utilities."""

from __future__ import annotations

from typing import Any


def fetch_all(
    client: Any,
    table: str,
    select: str = "*",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Paginated fetch from any Supabase table (1000 rows per page)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    batch = 1000
    while True:
        q = client.table(table).select(select).range(offset, offset + batch - 1)
        if filters:
            for col, val in filters.items():
                q = q.eq(col, val)
        resp = q.execute()
        rows.extend(resp.data)
        if len(resp.data) < batch:
            break
        offset += batch
    return rows
