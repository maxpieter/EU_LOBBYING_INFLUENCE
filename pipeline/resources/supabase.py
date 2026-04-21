"""Supabase client resource for Dagster."""

import os
import sys
from typing import Any, Dict, List, Optional

from dagster import ConfigurableResource
from pydantic import Field

# Hack: Remove CWD from sys.path if it contains a non-package 'supabase' folder
cwd = os.getcwd()
supabase_local = os.path.join(cwd, "supabase")
removed_cwd = False

if (
    cwd in sys.path
    and os.path.isdir(supabase_local)
    and not os.path.exists(os.path.join(supabase_local, "__init__.py"))
):
    try:
        sys.path.remove(cwd)
        removed_cwd = True
    except ValueError:
        pass

try:
    from supabase import Client, create_client
except ImportError:
    if removed_cwd:
        sys.path.insert(0, cwd)
    raise
finally:
    if removed_cwd:
        sys.path.insert(0, cwd)


class SupabaseResource(ConfigurableResource):
    """Dagster resource for Supabase client."""

    url: Optional[str] = Field(
        default=None,
        description="Supabase project URL. Falls back to SUPABASE_URL env var.",
    )
    service_role_key: Optional[str] = Field(
        default=None,
        description="Supabase service role key. Falls back to SUPABASE_SERVICE_ROLE_KEY env var.",
    )

    _client: Optional[Client] = None

    def _get_url(self) -> str:
        if self.url:
            return self.url
        url = os.getenv("SUPABASE_URL")
        if not url:
            raise ValueError("SUPABASE_URL environment variable is not set.")
        return url

    def _get_key(self) -> str:
        if self.service_role_key:
            return self.service_role_key
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable is not set.")
        return key

    def get_client(self) -> Client:
        if self._client is None:
            self._client = create_client(self._get_url(), self._get_key())
        return self._client

    def upsert(
        self, table: str, data: List[Dict[str, Any]], on_conflict: Optional[str] = None
    ) -> Any:
        client = self.get_client()
        return client.table(table).upsert(data).execute()

    def insert(self, table: str, data: List[Dict[str, Any]]) -> Any:
        client = self.get_client()
        return client.table(table).insert(data).execute()

    def select(
        self, table: str, columns: str = "*", filters: Optional[Dict[str, Any]] = None
    ) -> Any:
        client = self.get_client()
        query = client.table(table).select(columns)
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        return query.execute()

    def rpc(self, function_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        client = self.get_client()
        if params:
            return client.rpc(function_name, params).execute()
        return client.rpc(function_name).execute()

    def batch_upsert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100,
        on_conflict: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Upsert records in batches with individual retry fallback.

        Returns {"success", "failed", "failed_ids"}. `failed_ids` holds the
        `id` field of any record that could not be upserted even after the
        individual retry — useful for callers that need to drop dependent
        rows from a subsequent insert to avoid FK violations.
        """
        client = self.get_client()
        success_count = 0
        failed_count = 0
        failed_ids: List[str] = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_num = i // batch_size + 1
            try:
                if on_conflict:
                    client.table(table).upsert(batch, on_conflict=on_conflict).execute()
                else:
                    client.table(table).upsert(batch).execute()
                success_count += len(batch)
                if logger:
                    logger.debug(f"Batch {batch_num} upserted ({len(batch)} records)")
            except Exception as e:
                if logger:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    logger.warning(f"Retrying individually for batch {batch_num}")

                for record_idx, record in enumerate(batch):
                    try:
                        if on_conflict:
                            client.table(table).upsert(
                                [record], on_conflict=on_conflict
                            ).execute()
                        else:
                            client.table(table).upsert([record]).execute()
                        success_count += 1
                    except Exception as record_error:
                        failed_count += 1
                        record_id = record.get("id")
                        if record_id is not None:
                            failed_ids.append(record_id)
                        if logger:
                            log_id = record_id if record_id is not None else f"index_{i + record_idx}"
                            logger.error(f"Record {log_id} failed: {record_error}")

        return {"success": success_count, "failed": failed_count, "failed_ids": failed_ids}
