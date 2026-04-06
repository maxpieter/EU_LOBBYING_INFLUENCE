"""Actors Assets - Bronze → Diamond (no Gold/AI layer).

EU institutional actors pipeline for European Commissioners.
"""

import os
from typing import Any, Dict, List, Optional

from dagster import AssetExecutionContext, AssetIn, Config, asset

from pipeline.models.actors import Actor
from pipeline.resources.supabase import SupabaseResource

from .bronze import fetch_actors

PARLIAMENT_ID = "eu"


class ActorsBronzeConfig(Config):
    """Configuration for actors bronze scraping."""

    max_actors: Optional[int] = None
    skip_declarations: bool = False
    skip_meetings: bool = False


@asset(
    name="eu_actors_bronze",
    group_name="eu_bronze",
    description=(
        "Scrape EU institutional actors (Commissioners and cabinet members) from official EC "
        "sources. Enriches each actor with: team composition, declarations of interest, past "
        "meetings, speeches, calendar, biography, published documents, and latest news. "
        "Validates all records against the Actor Pydantic model."
    ),
    compute_kind="scraping",
)
def eu_actors_bronze(
    context: AssetExecutionContext,
    config: ActorsBronzeConfig,
) -> List[Dict[str, Any]]:
    """Bronze layer: Complete actor scraping."""
    from datetime import datetime

    max_actors = config.max_actors or (
        int(os.getenv("ACTOR_TEST_LIMIT")) if os.getenv("ACTOR_TEST_LIMIT") else None
    )

    if max_actors:
        context.log.info(f"TEST MODE: Limiting to {max_actors} actors")

    context.log.info("Scraping EU institutional actors")

    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()

    actors = fetch_actors(
        date_range=(start_date, end_date),
        actor_type="all",
        logger=context.log,
    )

    if max_actors and len(actors) > max_actors:
        actors = actors[:max_actors]

    # Normalize and add parliament ID
    normalized = []
    for actor in actors:
        actor["parliament"] = PARLIAMENT_ID
        actor.setdefault("team", [])
        actor.setdefault("declarations", [])
        actor.setdefault("past_meetings", [])
        actor.setdefault("speeches", [])
        actor.setdefault("calendar", [])
        actor.setdefault("biography", [])
        actor.setdefault("documents", [])
        actor.setdefault("latest_news", [])
        normalized.append(actor)

    # Validate with Actor model
    validated = []
    for actor in normalized:
        try:
            validated.append(Actor(**actor).model_dump())
        except Exception as e:
            context.log.warning(
                f"Validation error for actor {actor.get('name', 'unknown')}: {e}"
            )

    context.add_output_metadata(
        {
            "count": len(validated),
            "commissioners": sum(
                1 for a in validated if a.get("actor_type") == "commissioner"
            ),
            "with_team": sum(1 for a in validated if a.get("team")),
            "with_declarations": sum(1 for a in validated if a.get("declarations")),
        }
    )

    return validated


@asset(
    name="eu_actors_diamond",
    group_name="eu_diamond",
    description=(
        "Upsert validated actor records (Commissioners, cabinet members) to the Supabase "
        "actors table with deterministic primary keys for idempotent re-runs."
    ),
    compute_kind="upload",
    ins={"bronze_data": AssetIn("eu_actors_bronze")},
)
def eu_actors_diamond(
    context: AssetExecutionContext,
    supabase: SupabaseResource,
    bronze_data: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Diamond layer: Upload to Supabase."""
    if not bronze_data:
        context.log.info("No data to upload")
        return {"success": 0, "failed": 0}

    context.log.info(f"Uploading {len(bronze_data)} actors to Supabase")

    from .diamond import upload_actors

    result = upload_actors(
        actors=bronze_data,
        supabase_resource=supabase,
        logger=context.log,
    )

    context.add_output_metadata(
        {
            "uploaded": result.get("success", 0),
            "failed": result.get("failed", 0),
        }
    )

    return result


actors_assets = [
    eu_actors_bronze,
    eu_actors_diamond,
]
