import pandas as pd
from numpy.typing import ArrayLike
from typing import Optional
import random

def calculate_team_stats(event):
    """Calculate team statistics from events DataFrame and return merged stats."""
    
    # All shots by team
    all_shots = (
        event[event['type'] == 'Shot']
        .dropna(axis=1, how='all')
        .groupby(['team'])
        .size()
        .reset_index(name='Shots')
        .sort_values(['team','Shots'], ascending=False)
    )

    xg = (
    event[
        (event['type'] == 'Shot') & 
        (event['shot_statsbomb_xg'].notna())
    ]
    .groupby('team')['shot_statsbomb_xg']
    .sum()
    .reset_index(name='xG')
    
    )
    possession = (
    event[event['possession_team'].notna()]
                .groupby('team')
                .size()
                .reset_index(name='Event Count')
    )

    # Calculate possession percentage
    #total_events = possession['Event Count'].sum()
    #possession['Possession'] = (possession['Event Count'] / total_events) * 100
    #possession['Possession'] = possession['Possession'].round(3)

    # Drop the raw count if not needed
    #possession = possession[['team', 'Possession']]

    # Goals scored by team
    if event[event['type'] == 'Shot'].empty:
        goals = pd.DataFrame(columns=['team', 'Goals'])
        goals_conceded = pd.DataFrame(columns=['team', 'Goals Conceded'])
    else:
        goal_events = event[
            (event['type'] == 'Shot') &
            (event['shot_outcome'] == 'Goal')
        ]
        # kann man hier oben evtl anstatt event eif all_shots nehmen für bessere runtime?
        # Goals scored
        goals = (
            goal_events
            .groupby('team')
            .size()
            .reset_index(name='Goals')
        )

        # Get teams in the match
        teams = event['team'].dropna().unique()

        # Build goals conceded by assuming the conceding team is the *other* team
        goal_events = goal_events.copy()
        goal_events['conceding_team'] = goal_events['team'].apply(
            lambda x: [team for team in teams if team != x][0] if len(teams) == 2 else None
        )

        # Group by conceding team
        goals_conceded = (
            goal_events
            .groupby('conceding_team')
            .size()
            .reset_index(name='Goals Conceded')
            .rename(columns={'conceding_team': 'team'})
        )


    #Still wrong (Duels: 128,  Source: Kicker https://www.kicker.de/bayern-gegen-hsv-2015-bundesliga-2854915/spieldaten)
    duels = duels_per_team = event[(event['type'] == 'Duel')].dropna(axis=1, how='all').shape[0]

    won_duels = event[
        (event['type'] == 'Duel') |
        #(events['duel_outcome'] == 'Success In Play') |
        (event['type'] == '50/50')
    ].dropna(axis=1, how='all')


    # Calculate total passes
    total_passes = (event[event['type'] == 'Pass']
                .groupby('team')
                .size()
                .reset_index(name='total_passes'))
    
    '''total_poss = (event['possession_team']
                .groupby('team')
                .size()
                .reset_index(name='total_possession')
                )'''
    
    

    # Calculate completed passes (successful passes with no outcome)
    completed_passes = (event[(event['type'] == 'Pass') & 
                            (event['pass_outcome'].isnull())]
                    .groupby('team')
                    .size()
                    .reset_index(name='completed_passes'))


    #starting_formations = event[event['type'] =='Starting XI'][['team', 'tactics']]
        
    # Merge all stats
    team_stats = pd.merge(completed_passes, total_passes, on='team')
    team_stats['pass_completion_percentage'] = (team_stats['completed_passes'] / 
                                              team_stats['total_passes']) * 100
    #team_stats['possession'] = team_stats['total_poss'] / team_stats[]
    team_stats = pd.merge(team_stats, all_shots, on='team')
    team_stats['duels'] = duels  # Note: this adds total duels to all teams
    team_stats = pd.merge(team_stats, xg, on='team', how='outer')
    team_stats = pd.merge(team_stats, goals, on='team', how='outer')
    team_stats = pd.merge(team_stats, goals_conceded, on='team', how='outer')
    team_stats['xG'] = team_stats['xG'].round(2)
    team_stats = pd.merge(team_stats, possession, on='team', how='outer')
    team_stats = team_stats.fillna(0)

    # team_stats = (pd.merge(team_stats, starting_formations, on='team')
    #              .rename(columns={'tactics': 'starting_formation'}))
    
    return team_stats

# TODO: Implemet Hyperparameter, wich considers various methods of combining like for example weight the more soon games more
def calculate_chances_from_played_games(team_stats : pd.DataFrame, considered_matchweeks: ArrayLike, round_decimals: Optional[int] = None ) -> pd.DataFrame :
    team_stats_considered = team_stats[team_stats['match_week'].isin(considered_matchweeks)]
    chances = team_stats_considered.groupby('team').agg(
        avg_completed_passes=('completed_passes', 'mean'),
        avg_total_passes=('total_passes', 'mean'),
        avg_pass_comp_pct=('pass_completion_percentage', 'mean'),
        avg_shots=('Shots', 'mean'),
        avg_duels=('duels', 'mean'),
        avg_goals=('Goals', 'mean')
    )
    
    if round_decimals is not None:
        chances = chances.round(round_decimals)
    
    return chances

def create_test_dataset_entry_based_on_certain_matchweeks(home_team, away_team, chances, matches):

    # 1. Extract rows (keep as DataFrames)
    home_stats = chances.loc[[home_team]].reset_index()  # Removes 'team' index
    away_stats = chances.loc[[away_team]].reset_index()  # Removes 'team' index

    # 2. Rename columns to avoid duplicates (e.g., 'completed_passes' → 'home_completed_passes')
    home_stats = home_stats.add_prefix('home_')
    away_stats = away_stats.add_prefix('away_')

    # 3. Merge into one row
    prediction_stats = pd.concat([home_stats, away_stats], axis=1)

    # The the actual result:
    match_result = matches[
        (matches['home_team'] == home_team) & 
        (matches['away_team'] == away_team)
    ][['home_score', 'away_score']]

    home_score = match_result['home_score'].iloc[0]  # Extract scalar value
    away_score = match_result['away_score'].iloc[0]

    if home_score > away_score:
        actual_result = 'Home_Victory'
    elif home_score == away_score:
        actual_result = 'Draw'
    else:
        actual_result = 'Home_Defeat'

    actual_result

    prediction_stats['actual_result'] = actual_result

    return prediction_stats


def create_home_away_pairs(team_stats: pd.DataFrame) -> pd.DataFrame:
    # Get all unique teams
    all_teams = team_stats['team'].unique().tolist()
    
    if len(all_teams) < 18:
        raise ValueError("team_stats must contain at least 18 unique teams")
    
    # Randomly select 9 home teams
    home_teams = random.sample(all_teams, 9)
    
    # Remaining teams (potential away teams)
    away_pool = [team for team in all_teams if team not in home_teams]
    
    # Shuffle and assign one unique away team per home team
    random.shuffle(away_pool)
    away_teams = away_pool[:9]  # Take first 9
    
    # Create DataFrame
    pairings = pd.DataFrame({
        'home_team': home_teams,
        'away_team': away_teams
    })
    
    return pairings



def generate_unique_random_array(min_length=1, max_length=34, min_value=1, max_value=34):
    length = random.randint(min_length, max_length)
    return random.sample(range(min_value, max_value + 1), length)



def generate_dataset(team_stats : pd.DataFrame, matches : pd.DataFrame, amount_of_weeks_to_simulate: int = 1) -> pd.DataFrame:
    all_data_points = []
    
    for _ in range(amount_of_weeks_to_simulate):  # Simulate multiple weeks
        # Generate random matchweeks for this simulation
        random_training_array = generate_unique_random_array()
        
        # Calculate team stats for these matchweeks
        chances = calculate_chances_from_played_games(
            considered_matchweeks=random_training_array,
            team_stats=team_stats,
            round_decimals=None
        )
        
        # Create home-away pairings for this week
        pairings = create_home_away_pairs(team_stats=team_stats)
        
        # Process each pairing
        for _, pair in pairings.iterrows():
            data_point = create_test_dataset_entry_based_on_certain_matchweeks(
                home_team=pair['home_team'],
                away_team=pair['away_team'],
                chances=chances,
                matches=matches
            )
            all_data_points.append(data_point)
    
    # Combine all data points from all weeks
    dataset = pd.concat(all_data_points, ignore_index=True)
    return dataset