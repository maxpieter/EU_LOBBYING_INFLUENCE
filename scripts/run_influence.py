"""Quick runner for the influence pipeline outside Dagster."""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from supabase import create_client
import os

from pipeline.assets.analysis.influence import run_influence_pipeline


def main():
    procedure_id = sys.argv[1] if len(sys.argv) > 1 else "2023/0212(COD)"
    regen = "--regen" in sys.argv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("influence")

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, key)

    report = run_influence_pipeline(
        procedure_id=procedure_id,
        client=client,
        regen_taxonomy=regen,
        logger=logger,
    )

    stats = report.get("summary_stats", {})
    print(f"\nDone. Amendments: {stats.get('total_amendments_parsed')}, "
          f"Positions: {len(report.get('positions', []))}, "
          f"Commission dossiers: {len(report.get('commission_evidence', []))}, "
          f"Amendment dossiers: {len(report.get('amendment_evidence', []))}")


if __name__ == "__main__":
    main()
