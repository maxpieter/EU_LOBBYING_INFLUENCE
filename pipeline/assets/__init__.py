from .actors.definitions import actors_assets
from .commission_meetings.definitions import commission_meetings_assets
from .legislation.definitions import legislation_assets
from .lobbying.definitions import lobbying_assets
from .meps.definitions import members_assets

all_assets = [
    *members_assets,
    *lobbying_assets,
    *legislation_assets,
    *actors_assets,
    *commission_meetings_assets,
]
