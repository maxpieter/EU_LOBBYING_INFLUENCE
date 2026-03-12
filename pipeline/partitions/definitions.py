"""Partition definitions for Dagster assets."""

from datetime import datetime, timedelta

from dagster import WeeklyPartitionsDefinition

# Weekly partitions for lobbying and legislation data
# Matches parl8 configuration
weekly_partitions = WeeklyPartitionsDefinition(
    start_date="2024-07-16",
    timezone="Europe/Brussels",
    fmt="%Y-%m-%d",
    day_offset=5,
    end_offset=1,
)


def get_week_range_from_partition(partition_key: str) -> tuple[datetime, datetime]:
    """Get start and end datetime for a weekly partition."""
    start_date = datetime.strptime(partition_key, "%Y-%m-%d")
    start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = (start_date + timedelta(days=6)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    return start, end
