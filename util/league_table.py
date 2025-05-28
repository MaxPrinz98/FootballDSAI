import pandas as pd

def _get_points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0

def generate_league_table(matches):
    home_df = matches[['home_team', 'away_team', 'home_score', 'away_score']].copy()
    
    # Process home stats
    home_stats = home_df.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    home_stats['goals_for'] = home_df['home_score']
    home_stats['goals_against'] = home_df['away_score']
    home_stats['points'] = home_stats.apply(
        lambda row: _get_points(row['goals_for'], row['goals_against']), axis=1
    )
    
    # Process away stats
    away_stats = home_df.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    away_stats['goals_for'] = home_df['away_score']
    away_stats['goals_against'] = home_df['home_score']
    away_stats['points'] = away_stats.apply(
        lambda row: _get_points(row['goals_for'], row['goals_against']), axis=1
    )
    
    # Combine and aggregate
    full_stats = pd.concat([home_stats, away_stats])
    final_table = full_stats.groupby('team').agg(
        Matches=('team', 'count'),
        Wins=('points', lambda x: (x == 3).sum()),
        Draws=('points', lambda x: (x == 1).sum()),
        Losses=('points', lambda x: (x == 0).sum()),
        Goals_For=('goals_for', 'sum'),
        Goals_Against=('goals_against', 'sum'),
        Points=('points', 'sum')
    ).reset_index()
    
    final_table['Goal_Difference'] = final_table['Goals_For'] - final_table['Goals_Against']
    return final_table.sort_values(
        by=['Points', 'Goal_Difference', 'Goals_For'], 
        ascending=False
    ).reset_index(drop=True)