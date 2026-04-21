from .actors.definitions import actors_assets
from .analysis.definitions import analysis_assets
from .commission_meetings.definitions import commission_meetings_assets
from .legislation.definitions import legislation_assets
from .lobbying.definitions import lobbying_assets
from .meps.definitions import members_assets
from .organizations.definitions import organization_assets
from .procedures.definitions import procedure_matching_assets

# HYS feedback assets loaded separately (optional deps: pdfplumber, langchain)
try:
    from .feedback.hys_feedback_asset import hys_feedback_bronze, hys_feedback_chunks
    _feedback_assets = [hys_feedback_bronze, hys_feedback_chunks]
except ImportError:
    _feedback_assets = []

all_assets = [
    *members_assets,
    *organization_assets,
    *lobbying_assets,
    *legislation_assets,
    *actors_assets,
    *commission_meetings_assets,
    *procedure_matching_assets,
    *analysis_assets,
    *_feedback_assets,
]
