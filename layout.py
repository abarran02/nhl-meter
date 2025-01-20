import pandas as pd
from dash import dcc, html

games = pd.read_parquet("data/games.parquet")
teams = games["Home_Team"].unique()
teams.sort()

linkedin_url = "https://www.linkedin.com/in/abarran/"
github_url = "https://github.com/abarran02/nhl-meter"

footer_style = {
    "text-align": "center",
    "display": "flex",
    "align-items": "right",
    "justify-content": "right",
    "gap": "10px"
}

app_layout = html.Div([
    html.Header([
        html.A(html.I(className="fab fa-linkedin fa-lg"), href=linkedin_url, target="_blank"),
        html.A(html.I(className="fab fa-github fa-lg"), href=github_url, target="_blank")
    ], style=footer_style),

    html.Div([
        dcc.Location(id="page-load", refresh=False),
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
            dcc.Loading(
                id="loading",
                children=[dcc.Graph(id="probability-graph")],
                type="circle"
            )
        ], style={"flex": "auto"}, id="graph"),
        html.Div(style={"flex": "10%"}),
    ], style={"display": "flex", "flex-direction": "row"})
], style={"height": "100vh"})
