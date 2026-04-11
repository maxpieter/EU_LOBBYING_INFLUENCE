from .actors.definitions import actors_assets
from .analysis.definitions import analysis_assets
from .commission_meetings.definitions import commission_meetings_assets
from .feedback.hys_feedback_asset import hys_feedback_bronze, hys_feedback_chunks
from .legislation.definitions import legislation_assets
from .lobbying.definitions import lobbying_assets
from .meps.definitions import members_assets

all_assets = [
    *members_assets,
    *lobbying_assets,
    *legislation_assets,
    *actors_assets,
    *commission_meetings_assets,
    *analysis_assets,
    hys_feedback_bronze,
    hys_feedback_chunks,
]
