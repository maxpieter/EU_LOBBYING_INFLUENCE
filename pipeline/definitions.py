"""Dagster Definitions entry point for EU Lobby Influence pipeline.

Run with:
    dagster dev -f pipeline/definitions.py

Or for production:
    dagster-daemon run &
    dagster-webserver -h 0.0.0.0 -p 3001
"""

import os
from pathlib import Path

from dagster import Definitions, in_process_executor
from dotenv import load_dotenv

from pipeline.assets import all_assets
from pipeline.resources.http_client import HttpClientResource
from pipeline.resources.json_io_manager import JsonFileIOManager
from pipeline.resources.selenium import SeleniumResource
from pipeline.resources.supabase import SupabaseResource

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env", override=True)

# =============================================================================
# Resource Configurations
# =============================================================================

RESOURCES = {
    # Supabase client for database operations
    "supabase": SupabaseResource(),
    # Shared HTTP client with rate limiting for external APIs
    "http_client": HttpClientResource(
        rate_limit_delay=0.75,
        eurlex_delay=10.0,
        max_retries=3,
        timeout=30,
    ),
    # Selenium browser pool for web scraping
    "selenium": SeleniumResource(
        pool_size=3,
        page_load_timeout=30,
        implicit_wait=10,
        max_retries=3,
    ),
    # JSON file IO manager for data persistence
    "io_manager": JsonFileIOManager(
        base_path=str(project_root / "data"),
    ),
}

# =============================================================================
# Definitions
# =============================================================================

print(f"\n[OK] Total assets loaded: {len(all_assets)}")

defs = Definitions(
    assets=all_assets,
    resources=RESOURCES,
    executor=in_process_executor,
)
