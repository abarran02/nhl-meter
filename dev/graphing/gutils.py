import importlib.resources
import json
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import ndarray

# broken for dev/ for now, only works from app.py
with importlib.resources.open_text('dev.graphing', 'teams.json') as f:
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
            color = t["colors"]["hex"][0]
            return (t["name"], f'#{color}')

    # no team color found
    raise AttributeError(f"Invalid team code: {team_code}")

def graph_probabilities(time_remaining: pd.Series | ndarray,
                        probabilities: pd.Series | ndarray,
                        home: tuple[str, str],
                        away: tuple[str, str]) -> None:
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

def convert_seconds_to_time_format(time_remaining: pd.Series | np.ndarray) -> list[tuple[str, int]]:
    tuples = []
    for t in time_remaining:
        period = ceil(t / 1200)
        minutes = int(t // 60)
        seconds = int(t % 60)

        time_str = f"{minutes}:{seconds:02d}"
        tuples.append((time_str, period))

    return tuples

def graph_probabilities_plotly(time_elapsed: pd.Series | np.ndarray,
                                probabilities: pd.Series | np.ndarray,
                                home: tuple[str, str],
                                away: tuple[str, str]) -> go.Figure:
    """Generate and display win probability over time from home team perspective using Plotly.

    Args:
        time_elapsed (pd.Series | ndarray): Time elapsed in seconds
        probabilities (pd.Series | ndarray): Win probability for home team
        home (tuple[str, str]): Home team name and hex color for shading
        away (tuple[str, str]): Away team name and hex color for shading
    """
    fig = go.Figure()

    # Fill for home team
    fig.add_trace(go.Scatter(
        x=time_elapsed,
        y=probabilities,
        mode='lines',
        line=dict(color=home[1]),
        name=home[0],
        hoverinfo="none",
        stackgroup="one"
    ))

    # Fill for away team
    fig.add_trace(go.Scatter(
        x=time_elapsed,
        y=(np.ones(probabilities.shape) - probabilities),
        mode='lines',
        line=dict(color=away[1]),
        name=away[0],
        hoverinfo="none",
        stackgroup="one"
    ))
    
    # ugly but I got it to work and do not want to change it
    hover_text = [f"Time: {x[0]}<br>Period: {x[1]}<br>Value: {y:.4f}<extra></extra>" for x, y in zip(convert_seconds_to_time_format(time_elapsed), probabilities)]
    
    # Line for probabilities
    fig.add_trace(go.Scatter(
        x=time_elapsed,
        y=probabilities,
        mode='lines',
        line=dict(color='black', width=1),
        name='Win Probability',
        hovertemplate=hover_text
    ))

    fig.add_hline(y=0.5, line=dict(color="Red", width=2, dash="dash"))

    fig.update_layout(
        title=f"{home[0]} Win Probability Over Time",
        xaxis=dict(
            title="Time Elapsed (s)",
        ),
        yaxis=dict(
            title="Win Probability",
            range=[0, 1.0]
        ),
        legend=dict(title="Teams"),
        template="simple_white"
    )

    return fig
