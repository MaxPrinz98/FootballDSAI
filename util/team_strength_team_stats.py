import pandas as pd
from numpy.typing import ArrayLike
from typing import Optional
import random


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns



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
                .groupby('possession_team')
                .size()
                .reset_index(name='ball_possession')
                .rename(columns={'possession_team': 'team'})
    )

    # Calculate possession percentage
    total_events = possession['ball_possession'].sum()
    possession['Possession'] = (possession['ball_possession'] / total_events) * 100
    possession['Possession'] = possession['Possession'].round(0)

    # Drop the raw count if not needed
    possession = possession[['team', 'Possession']]

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
    team_stats.drop(columns=['completed_passes']) # dont need and correlation of 1 to total passes
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

def calculate_features_based_on_certain_matchweeks(home_team, away_team, chances, matches):

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
    ][['home_score', 'away_score','match_week']]

    home_score = match_result['home_score'].iloc[0]  # Extract scalar value
    away_score = match_result['away_score'].iloc[0]

    if home_score > away_score:
        actual_result = 'Home_Victory'
    elif home_score == away_score:
        actual_result = 'Draw'
    else:
        actual_result = 'Home_Defeat'

    actual_result

    prediction_stats['match_week'] = match_result['match_week'].iloc[0]
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
    match_week = 1
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
            data_point = calculate_features_based_on_certain_matchweeks(
                home_team=pair['home_team'],
                away_team=pair['away_team'],
                chances=chances,
                matches=matches
            )
            data_point['match_week'] = match_week
            all_data_points.append(data_point)
        match_week += 1

    
    # Combine all data points from all weeks
    dataset = pd.concat(all_data_points, ignore_index=True)
    return dataset

def generate_real_dataset(team_stats, matches,) -> pd.DataFrame:
    all_features = []
    for i in range(1,matches['match_week'].max()+1):
        team_pairings = matches[matches['match_week'] == i]
        # Cheating the stats data for first match_week because we cannot predict a game, when no game before is played
        if i == 1:
            real_chances = calculate_chances_from_played_games(team_stats=team_stats, considered_matchweeks=list(range(1, matches['match_week'].max()+1)), round_decimals=None)
            for _, match in team_pairings.iterrows():
                features = calculate_features_based_on_certain_matchweeks(home_team=match['home_team'],away_team=match['away_team'], chances= real_chances, matches=matches)
                all_features.append(features)
        else:
            real_chances = calculate_chances_from_played_games(team_stats=team_stats, considered_matchweeks=list(range(1, i)), round_decimals=None)
            for _, match in team_pairings.iterrows():
                features = calculate_features_based_on_certain_matchweeks(home_team=match['home_team'],away_team=match['away_team'], chances= real_chances, matches=matches)
                all_features.append(features)

    final_features = pd.concat(all_features, ignore_index=True)
    return final_features


def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple models using GridSearchCV with weighted F1 scoring.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion for test split
        random_state: Random seed for reproducibility
        
    Returns:
        Fitted GridSearchCV object containing all results
    """
    # Initial split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False
    )

    # Define preprocessing
    categorical_features = ['home_team', 'away_team']
    numerical_features = X_train.columns.difference(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    # Safe F1 scorer to handle edge cases
    def safe_f1_weighted(y_true, y_pred):
        labels_in_data = list(set(y_true) | set(y_pred))
        if len(labels_in_data) < 2:
            return 0.0
        return f1_score(y_true, y_pred, average='weighted', labels=labels_in_data)

    f1_weight_scorer = make_scorer(safe_f1_weighted)

    print("Class distribution in y_train:", pd.Series(y_train).value_counts())

    # Create pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', None)  # Placeholder
    ])

    # Parameter grid
    param_grid = [
        {
            'model': [RandomForestClassifier()],
            'model__n_estimators': [100, 400],
            'model__max_depth': [None, 10]
        },
        {
            'model': [LogisticRegression(max_iter=1000)],
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2']
        },
        {
            'model': [GradientBoostingClassifier()],
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.1, 0.5]
        },
        {
            'model': [SVC(probability=True)],
            'model__C': [0.1, 1],
            'model__kernel': ['linear', 'rbf']
        }
    ]

    # Configure GridSearch
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring=f1_weight_scorer,
        error_score='raise',
        verbose=1,
        n_jobs=-1,
        refit=True
    )

    # Fit models
    grid.fit(X_train, y_train)

    # Print results
    print(f"\nBest model: {grid.best_estimator_.named_steps['model']}")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.3f}")
    
    return grid, X_test, y_test, X_train, y_train, preprocessor


def create_brier_scores(X_test, y_test, model, dataset) -> pd.DataFrame:
    probabilities_array = model.predict_proba(X_test)
    probabilities_array.shape  #'Draw', 'Home_Defeat', 'Home_Victory'

    x_probabilities = pd.DataFrame(
        data=probabilities_array,
        columns=['Draw', 'Home_Defeat', 'Home_Victory'],
        index=X_test.index
    )
    x_probabilities = x_probabilities.add_prefix('predicted_')

    y_test_proba = pd.get_dummies(y_test, dtype='float64')
    y_test_proba = y_test_proba.add_prefix('actual_')

    prediction_and_outcome = pd.merge(x_probabilities, y_test_proba, left_index=True, right_index=True)
    prediction_and_outcome = pd.merge(prediction_and_outcome, dataset['match_week'], left_index=True, right_index=True)

    prediction_and_outcome['brier_score'] = (
        (prediction_and_outcome['predicted_Draw'] - prediction_and_outcome['actual_Draw']) ** 2 +
        (prediction_and_outcome['predicted_Home_Defeat'] - prediction_and_outcome['actual_Home_Defeat']) ** 2 +
        (prediction_and_outcome['predicted_Home_Victory'] - prediction_and_outcome['actual_Home_Victory']) ** 2
    )

    return prediction_and_outcome

def evaluate_all_models(grid_search, X_test, y_test,X_train, y_train, dataset):
    """
    Evaluate all models from GridSearchCV and return DataFrame with Brier scores
    
    Args:
        grid_search: Fitted GridSearchCV object
        X_test: Test features
        y_test: True labels
        dataset: Original dataset with match_week
        
    Returns:
        DataFrame with Brier scores for all models
    """
    all_scores = []
    
    # Get all candidate estimators from grid search
    for i, (params, mean_score) in enumerate(zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score']
    )):
        # Get the fitted pipeline
        fitted_pipeline = grid_search.best_estimator_
        
        # Update the pipeline with current parameters
        current_pipeline = fitted_pipeline.set_params(**params)
        
        # Refit the pipeline with these parameters
        current_pipeline.fit(X_train, y_train)  # You'll need X_train and y_train in scope
        
        # Calculate Brier scores
        try:
            _, brier_scores = create_brier_scores(X_test, y_test, current_pipeline, dataset)
            
            # Get model name with parameters
            model_name = f"{params['model'].__class__.__name__}"
            for param, value in params.items():
                if param != 'model':
                    param_name = param.split('__')[-1]
                    model_name += f" {param_name}={value}"
            
            # Add to results
            scores_df = brier_scores.reset_index()
            scores_df['model'] = model_name
            scores_df['cv_score'] = mean_score
            all_scores.append(scores_df)
        except Exception as e:
            print(f"Failed to evaluate model {i}: {str(e)}")
            continue
    
    return pd.concat(all_scores) if all_scores else pd.DataFrame()