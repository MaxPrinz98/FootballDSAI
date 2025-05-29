import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

def initialize_team_stats(matches, starting_elo = 2000):
    unique_teams = pd.concat([matches['home_team'], matches['away_team']]).unique()
    return pd.DataFrame({
        'team': unique_teams,
        'elo_rating': starting_elo,
        'total_games': 0,
        'total_wins': 0,
        'total_draws': 0,
        'total_lost': 0,
        'total_goals_scored': 0,
        'total_goals_conceded': 0
    }).set_index('team')


def calculate_elo_ratings(team_stats, matches, K_FACTOR=30, HOME_ADVANTAGE=0.00):
    elo_ratings = team_stats['elo_rating'].copy()
    elo_history = []
    
    for idx, match in matches.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]
        
        expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400)) + HOME_ADVANTAGE
        expected_away = 1 - expected_home
        
        # Determine actual results
        if home_score > away_score:
            actual_home = 1
            actual_away = 0
            home_result = 'win'
            away_result = 'loss'
        elif home_score < away_score:
            actual_home = 0
            actual_away = 1
            home_result = 'loss'
            away_result = 'win'
        else:
            actual_home = 0.5
            actual_away = 0.5
            home_result = 'draw'
            away_result = 'draw'
            
        # Update Elo ratings
        elo_ratings[home_team] += K_FACTOR * (actual_home - expected_home)
        elo_ratings[away_team] += K_FACTOR * (actual_away - expected_away)
        
        # Append result and Elo to history
        elo_history.append({
            'date': match['match_date'],  # for pivoting and Elo logic
            'plot_date': match['match_date'],  # for plotting
            'team': home_team,
            'elo': elo_ratings[home_team] ,#- HOME_ADVANTAGE,
            'match_id': idx,
            'result': home_result
        })
        elo_history.append({
            'date': match['match_date'],
            'plot_date': match['match_date'] + timedelta(minutes=5),  # slight offset for dot
            'team': away_team,
            'elo': elo_ratings[away_team],
            'match_id': idx,
            'result': away_result
        })
    
    return elo_ratings, pd.DataFrame(elo_history)

def loss_from_comparing_tables(actual_standings,elo_standing):
    total_loss = 0
    elo_standings_tmp = elo_standing.copy().reset_index()
    for index , row in elo_standings_tmp.iterrows():
        elo_team_name = row['team']
        index_actual = actual_standings[actual_standings['team']==elo_team_name].index[0]
        index_elo = index
        total_loss += abs(index_elo - index_actual)
    return total_loss
