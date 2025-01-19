import pandas as pd
from dash import dcc, html

games = pd.read_parquet("data/games.parquet")
teams = games["Home_Team"].unique()
teams.sort()

app_layout = html.Div([
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
