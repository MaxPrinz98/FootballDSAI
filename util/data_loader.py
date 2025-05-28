import pandas as pd
from statsbombpy import sb

def load_competitions():
    return sb.competitions()

def load_matches(competition_id=9, season_id=27):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches.drop([
        "match_status_360", "last_updated", "last_updated_360",
        "match_status", "data_version", "shot_fidelity_version", "xy_fidelity_version"
    ], axis=1, inplace=True)
    matches['match_date'] = pd.to_datetime(matches['match_date'])
    return matches.sort_values('match_date')