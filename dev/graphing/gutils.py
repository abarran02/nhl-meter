import json

import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray

# load teams and colors
import importlib.resources

with importlib.resources.open_text('graphing', 'teams.json') as f:
    teams = json.load(f)

def team_name_color(team_code: str) -> tuple[str, str]:
    """Finds the hex color for a given NHL team by team code

    Args:
        team_code (str): 3 letter team code like `FLA` or `L.A`

    Raises:
        AttributeError: given team code not found

    Returns:
        tuple[str, str]: full team name, color hex in format `#ffffff`
    """

    for t in teams:
        if t["team_code"] == team_code:
            idx = next(i for i, color in enumerate(t["colors"]["hex"]) if color != '010101')
            color = t["colors"]["hex"][idx]
            return (t["name"], f'#{color}')
        
    # no team color found
    raise AttributeError

def graph_probabilities(time_remaining: pd.Series | ndarray, probabilities: pd.Series | ndarray, home: tuple[str, str], away: tuple[str, str]) -> None:
    """Generate and display win probability over time from home team perspective.

    Args:
        time_remaining (pd.Series | ndarray): Time remaining in seconds
        probabilities (pd.Series | ndarray): Win probability for home team
        home (tuple[str, str]): Home team name and hex color for shading
        away (tuple[str, str]): Away team name and hex color for shading
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_remaining, probabilities, color='black', linewidth=1)

    plt.fill_between(time_remaining, probabilities, 0, color=home[1], alpha=0.4, label=home[0])
    plt.fill_between(time_remaining, probabilities, 1, color=away[1], alpha=0.4, label=away[0])

    plt.xlim(time_remaining.min(), time_remaining.max())
    plt.ylim(0, 1.0)
    plt.grid(False)
    plt.gca().invert_xaxis()
    plt.title("Home Team Win Probability Over Time")
    plt.xlabel("Time Remaining (s)")
    plt.ylabel("Win Probability")
    plt.legend()
    plt.grid()
    plt.show()
