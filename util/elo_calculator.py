import pandas as pd
from collections import defaultdict
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from itertools import product


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
    })


def calculate_elo_ratings(MATCHES, K_FACTOR=30., SCALING_FACTOR = 400., HOME_ADVANTAGE=0., STARTING_ELO=2000.):
    elo_history = []
    
    team_stats = initialize_team_stats(MATCHES, starting_elo = STARTING_ELO).set_index('team')

    for idx, match in MATCHES.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        
        home_elo = team_stats.loc[home_team, 'elo_rating']
        away_elo = team_stats.loc[away_team, 'elo_rating']
        
        expected_home = 1 / (1 + 10**((away_elo - home_elo) / SCALING_FACTOR)) + HOME_ADVANTAGE
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
        home_elo_diff = K_FACTOR * (actual_home - expected_home)
        away_elo_diff = K_FACTOR * (actual_away - expected_away)

        team_stats.loc[home_team, 'elo_rating'] += home_elo_diff
        team_stats.loc[away_team, 'elo_rating'] += away_elo_diff
        
        # Append result and Elo to history
        elo_history.append({
            'date': match['match_date'],  
            'plot_date': match['match_date'],
            'team': home_team,
            'elo': team_stats.loc[home_team, 'elo_rating'] ,#- HOME_ADVANTAGE,
            'elo_diff': home_elo_diff,
            'match_id': match['match_id'],
            'result': home_result
        })
        elo_history.append({
            'date': match['match_date'],
            'plot_date': match['match_date'] + timedelta(minutes=5),  # slight offset for dot
            'team': away_team,
            'elo': team_stats.loc[away_team, 'elo_rating'],
            'elo_diff': away_elo_diff,
            'match_id': match['match_id'],
            'result': away_result
        })


    return team_stats.sort_values('elo_rating', ascending=False).reset_index(), pd.DataFrame(elo_history)

def loss_from_comparing_tables(actual_standings,elo_standing):
    total_loss = 0
    elo_standings_tmp = elo_standing.copy().reset_index()
    for index , row in elo_standings_tmp.iterrows():
        elo_team_name = row['team']
        index_actual = actual_standings[actual_standings['team']==elo_team_name].index[0]
        index_elo = index
        total_loss += abs(index_elo - index_actual)
    return total_loss



# 1. 3-way probability model (logistic-style with draw modeling)
def elo_three_way_probs(home_elo, away_elo, alpha, beta, gamma):
    diff = home_elo - away_elo
    p_home = np.exp(alpha * diff)
    p_away = np.exp(-alpha * diff)
    p_draw = np.exp(beta - gamma * abs(diff))
    Z = p_home + p_draw + p_away
    return p_home / Z, p_draw / Z, p_away / Z


# 2. Brier score computation per match week
def brier_scores_by_matchweek(matches, alpha=0.01, beta=1.0, gamma=0.01, R_0=1000,K=20):
    elo_ratings = {team: R_0 for team in pd.concat([matches['home_team'], matches['away_team']]).unique()}
    brier_records = []

    # Sort matches by match week
    matches = matches.sort_values(by='match_week')

    for _, match in matches.iterrows():
        home = match['home_team']
        away = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        week = match['match_week']

        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]

        # 3-way Elo prediction
        p_home, p_draw, p_away = elo_three_way_probs(home_elo, away_elo, alpha, beta, gamma)

        # True outcome
        if home_score > away_score:
            actual = [1, 0, 0]
        elif home_score == away_score:
            actual = [0, 1, 0]
        else:
            actual = [0, 0, 1]

        predicted = [p_home, p_draw, p_away]

        # Brier score for this match
        brier = np.sum((np.array(predicted) - np.array(actual)) ** 2)

        # Store match-level Brier score with match_week info
        brier_records.append({'match_week': week, 'brier_score': brier})

        # Optionally: Elo update (if you want to refine ratings)
        # Simplified update using K-factor
        
        elo_ratings[home] += K * (actual[0] - p_home)
        elo_ratings[away] += K * (actual[2] - p_away)

    # Create DataFrame and average per week
    brier_df = pd.DataFrame(brier_records)
    weekly_brier = brier_df.groupby('match_week')['brier_score'].mean().reset_index()

    return weekly_brier




def elo_grid_search(matches, param_grid):
    results = []
    best_score = float('inf')
    best_params = None
    
    # Generate all combinations of parameters
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        # Run ELO calculation with current parameters
        '''team_stats, _ = calculate_elo_ratings(
            matches,
            K_FACTOR = params.get('K_FACTOR', 1),
            HOME_ADVANTAGE = params.get('HOME_ADVANTAGE', 0),
            SCALING_FACTOR = params.get('SCALING_FACTOR', 400),
            STARTING_ELO = params.get('STARTING_ELO', 1000)
        )'''
        
        # Compute loss average brier score 
        
        loss_df = brier_scores_by_matchweek(matches = matches , K = params.get('K_FACTOR', 1),R_0=params.get('STARTING_ELO', 1000),
                                            alpha = (np.log(10) / params.get('SCALING_FACTOR', 400) ) )
        current_brier = 0
        n = 0
        for _,row in loss_df.iterrows():
            current_brier += row['brier_score']
            n += 1
        current_brier = current_brier/n

        
        # Store results
        results.append({
            'params': params,
            'loss': current_brier
        })
        
        # Update best parameters if current loss is better
        if current_brier < best_score:
            best_score = current_brier
            best_params = params
    
    results_df = pd.DataFrame(results)
    params_df = pd.json_normalize(results_df['params'])
    final_results_df = pd.concat([params_df, results_df['loss']], axis=1)


    return best_params, best_score, final_results_df