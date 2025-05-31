import matplotlib.pyplot as plt

from util.elo_calculator import calculate_elo_ratings, initialize_team_stats, loss_from_comparing_tables

def plot_kfactor_loss_curve_range(k_start, k_end, matches, final_table, starting_elo=1000, home_advantage=0.0):
    """
    Plots the loss curve for K_FACTOR values from k_start to k_end (inclusive).

    Parameters:
    - k_start: starting value of K_FACTOR (int)
    - k_end: ending value of K_FACTOR (int)
    - matches: DataFrame of matches
    - final_table: DataFrame of final league rankings (must include 'team' and 'rank')
    - starting_elo: initial Elo rating for all teams (default: 1000)
    - home_advantage: Elo bonus for home teams (default: 0.0)
    """
    losses = []
    k_values = list(range(k_start, k_end + 1))
    
    # Ensure final_table is indexed by team
    final_table = final_table.set_index('team')
    
    for k in k_values:
        team_stats = initialize_team_stats(matches, starting_elo=starting_elo)
        elo_ratings, _ = calculate_elo_ratings(team_stats.copy(), matches, K_FACTOR=k, HOME_ADVANTAGE=home_advantage)
        team_stats['elo_rating'] = team_stats.index.map(elo_ratings)
        loss = loss_from_comparing_tables(final_table, team_stats)
        losses.append(loss)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, losses, marker='o', linestyle='-')
    plt.title("Loss vs. K Factor")
    plt.xlabel("K Factor")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
