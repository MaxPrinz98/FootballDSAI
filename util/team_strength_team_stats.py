
import pandas as pd

def calculate_team_stats(event):
    """Calculate team statistics from events DataFrame and return merged stats."""
    
    all_shots = event[event['type'] == 'Shot'].dropna(axis=1, how='all').groupby(['team']).size().reset_index(name='Shots').sort_values(['team','Shots'], ascending=False)

    if event[(event['type'] == 'Shot')].empty:
        goals = pd.DataFrame(columns=['team', 'Goals'])
    else:
        goals = (
            event[
                (event['type'] == 'Shot') & 
                (event['shot_outcome'] == 'Goal')
            ]
            .groupby('team')
            .size()
            .reset_index(name='Goals')
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

    # Calculate completed passes (successful passes with no outcome)
    completed_passes = (event[(event['type'] == 'Pass') & 
                            (event['pass_outcome'].isnull())]
                    .groupby('team')
                    .size()
                    .reset_index(name='completed_passes'))


    starting_formations = event[event['type'] =='Starting XI'][['team', 'tactics']]
        
    # Merge all stats
    team_stats = pd.merge(completed_passes, total_passes, on='team')
    team_stats['pass_completion_percentage'] = (team_stats['completed_passes'] / 
                                              team_stats['total_passes']) * 100
    
    team_stats = pd.merge(team_stats, all_shots, on='team')
    team_stats['duels'] = duels  # Note: this adds total duels to all teams
    team_stats = pd.merge(team_stats, goals, on='team', how='outer').fillna(0)
    team_stats = (pd.merge(team_stats, starting_formations, on='team')
                 .rename(columns={'tactics': 'starting_formation'}))
    
    return team_stats

# Usage:
# team_stats = calculate_team_stats(events)