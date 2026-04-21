"""Search for procedures by proposal document number in Supabase.

Usage:
    python scripts/search_procedure.py <query>
    python scripts/search_procedure.py "COM(2023)206"
    python scripts/search_procedure.py "2023/0212"
    python scripts/search_procedure.py "52023PC"           # CELEX prefix
    python scripts/search_procedure.py --field id "COD"   # search a specific field
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from supabase import create_client


SEARCH_FIELDS = [
    "commission_document",
    "celex_number",
    "id",
    "process_id",
]

DISPLAY_FIELDS = [
    "id",
    "process_id",
    "commission_document",
    "celex_number",
    "title",
    "status",
    "stage",
    "proposal_date",
    "policy_area",
    "oeil_url",
]


def search_procedures(client, query: str, field: str | None = None):
    """Search procedures matching query across relevant fields (or a single field)."""
    fields_to_search = [field] if field else SEARCH_FIELDS
    seen_ids = set()
    results = []

    for f in fields_to_search:
        response = (
            client.table("procedures")
            .select(", ".join(DISPLAY_FIELDS))
            .ilike(f, f"%{query}%")
            .eq("is_deleted", False)
            .limit(50)
            .execute()
        )
        for row in response.data:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                results.append(row)

    return results


def print_results(results: list, query: str):
    if not results:
        print(f"No procedures found matching '{query}'.")
        return

    print(f"\nFound {len(results)} procedure(s) matching '{query}':\n")
    print("=" * 80)
    for proc in results:
        print(f"  ID               : {proc.get('id', '')}")
        print(f"  Process ID       : {proc.get('process_id', '')}")
        print(f"  Commission doc   : {proc.get('commission_document', '')}")
        print(f"  CELEX number     : {proc.get('celex_number', '')}")
        print(f"  Title            : {proc.get('title', '')[:100]}")
        print(f"  Status / Stage   : {proc.get('status', '')} / {proc.get('stage', '')}")
        print(f"  Proposal date    : {proc.get('proposal_date', '')}")
        print(f"  Policy area      : {proc.get('policy_area', '')}")
        if proc.get("oeil_url"):
            print(f"  OEIL URL         : {proc.get('oeil_url', '')}")
        print("-" * 80)


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    field = None
    if "--field" in args:
        idx = args.index("--field")
        if idx + 2 >= len(args):
            print("Usage: --field <field_name> <query>")
            sys.exit(1)
        field = args[idx + 1]
        if field not in SEARCH_FIELDS:
            print(f"Unknown field '{field}'. Valid fields: {', '.join(SEARCH_FIELDS)}")
            sys.exit(1)
        query = args[idx + 2]
    else:
        query = " ".join(args)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
        sys.exit(1)

    client = create_client(url, key)
    results = search_procedures(client, query, field)
    print_results(results, query)


if __name__ == "__main__":
    main()
