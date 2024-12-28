import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from keras.models import load_model

from dev.graphing import gutils

games = pd.read_parquet("data/games.parquet")
slices = pd.read_parquet("data/time_slices.parquet")
model = load_model("dev/models/meter_lstm16d2.keras")

teams = games["Home_Team"].unique()
teams.sort()

app = Dash()

app.layout = [
    html.H2("Teams and Games"),
    html.Div([
        html.Label("Home Team"),
        dcc.Dropdown(teams, teams[0], id="home-dropdown"),
    ], className="dropdown-container"),
    html.Div([
        html.Label("Away Team"),
        dcc.Dropdown(id="away-dropdown"),
    ], className="dropdown-container"),
    html.Div([
        html.Label("Games"),
        dcc.Dropdown(id="game-dropdown"),
    ], className="dropdown-container"),
    dcc.Graph(id="probability-graph")
]

@app.callback(
    Output("away-dropdown", "options"),
    [Input("home-dropdown", "value")]
)
def update_away_dropdown(home):
    indices = np.where(teams == home)
    return np.delete(teams, indices)

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
        return go.Figure()

    game, season = game_season.split('.')
    filtered_games = slices[(slices["game"] == int(game)) & (slices["season"] == int(season))]

    X = filtered_games.drop(columns=["winner", "game", "season"])

    probabilities = model.predict(X)

    home = gutils.team_name_color(home)
    away = gutils.team_name_color(away)

    fig = gutils.graph_probabilities_plotly(filtered_games["time_remaining"] * 3600, probabilities.flatten(), home, away)
    return fig

if __name__ == "__main__":
    app.run(debug=True)
