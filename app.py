import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from keras.models import load_model

from dev import meter
from dev.graphing import gutils

current_file_path = Path(__file__).resolve()
data_path = current_file_path.parent / "data"
model_path = current_file_path.parent / "dev" / "models"

games = pd.read_parquet(data_path / "games.parquet")
slices = pd.read_parquet(data_path / "time_slices.parquet")
ot_pbp = pd.concat([
    pd.read_parquet(data_path / "regular_ot_pbp.parquet"),
    pd.read_parquet(data_path / "playoff_ot_pbp.parquet")
])

with open(model_path / "one_hot_columns.json", 'r') as f:
    one_hot_columns = json.load(f)

model_regulation = load_model(model_path / "meter_lstm16d2.keras")
model_overtime = load_model(model_path / "meter_ot_lstm16d1.keras")

teams = games["Home_Team"].unique()
teams.sort()

app = Dash()

app.layout = html.Div([
    html.Div(style={"flex": "10%"}),
    html.Div([
        html.H3("Teams and Games"),
        html.Div([
            html.Label("Home Team"),
            dcc.Dropdown(
                options=[{"label": team, "value": team} for team in teams],
                value=teams[0],
                id="home-dropdown"
            ),
        ], className="dropdown-container"),
        html.Div([
            html.Label("Away Team"),
            dcc.Dropdown(
                options=[{"label": team, "value": team} for team in teams],
                id="away-dropdown"
            ),
        ], className="dropdown-container"),
        html.Button("Switch", id="switch-button"),
        html.Div([
            html.Label("Games"),
            dcc.Dropdown(id="game-dropdown"),
        ], className="dropdown-container", style={"margin-top": "20px"})
    ], style={"flex": "15%"}),
    html.Div([
        dcc.Graph(id="probability-graph")
    ], style={"flex": "auto"}),
    html.Div(style={"flex": "10%"}),
], style={"display": "flex"})


@app.callback(
    Output("away-dropdown", "options"),
    [Input("home-dropdown", "value")]
)
def update_away_dropdown(home):
    indices = np.where(teams == home)
    return np.delete(teams, indices)

@app.callback(
    [Output("home-dropdown", "value"),
     Output("away-dropdown", "value")],
    Input("switch-button", "n_clicks"),
    State("home-dropdown", "value"),
    State("away-dropdown", "value"),
    prevent_initial_call=True
)
def switch_home_away(n_clicks, home, away):
    return away, home

@app.callback(
    Output("game-dropdown", "options"),
    [Input("home-dropdown", "value"),
     Input("away-dropdown", "value")]
)
def update_game_dropdown(home, away):
    mask = (games["Home_Team"] == home) & (games["Away_Team"] == away)
    games_reduced = games[mask]

    return [{
        "label": f'{home} {g["Home_Score"]} - {away} {g["Away_Score"]} -- {g["Date"].strftime("%d %b %Y")}',
        "value": f'{g["Game_Id"]}.{g["Season"]}'
    } for idx, g in games_reduced.iterrows()]

@app.callback(
    Output("probability-graph", "figure"),
    [Input("home-dropdown", "value"),
     Input("away-dropdown", "value"),
     Input("game-dropdown", "value")]
)
def update_figure(home, away, game_season):
    if not game_season:
        # prevent drawing incomplete dropdown selection
        return go.Figure()

    game, season = [int(x) for x in game_season.split('.')]

    # find team full names and colors
    idx = 0
    home_name_color = gutils.team_name_color(home, idx)
    away_name_color = gutils.team_name_color(away, idx)
    while away_name_color[1] == home_name_color[1]:  # ensure colors are different
        idx += 1
        away_name_color = gutils.team_name_color(away, idx)

    time_elapsed, probabilities = meter.predict_regulation(game, season, slices, model_regulation)

    # handle overtime games
    mask = (games["Game_Id"] == game) & (games["Season"] == season)
    game_data = games[mask].iloc[0]
    if game_data["Period"] > 3:
        time_elapsed_ot, probabilities_ot = meter.predict_overtime(game, season, ot_pbp, model_overtime, one_hot_columns)

        # remove last data point of regulation to prevent overlap
        time_elapsed = pd.concat([time_elapsed[:-1], time_elapsed_ot])
        probabilities = np.concatenate((probabilities[:-1], probabilities_ot))

    fig = gutils.graph_probabilities_plotly(
        time_elapsed,
        probabilities,
        home_name_color,
        away_name_color
    )
    fig.update_layout(height=600)

    return fig

if __name__ == "__main__":
    app.run(debug=True)
