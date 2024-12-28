import numpy as np
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

df = pd.read_parquet("data/games.parquet")
teams = df["Home_Team"].unique()
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
    html.Div(id="display-selected-values"),
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
    mask = (df["Home_Team"] == home) & (df["Away_Team"] == away)
    games = df[mask]

    return [{
        "label": f'{home} {g["Home_Score"]} - {away} {g["Away_Score"]} -- {g["Date"].strftime("%m %b %Y")}',
        "value": f'{g["Game_Id"]}.{g["Season"]}'
    } for idx, g in games.iterrows()]

@app.callback(
    Output("display-selected-values", "children"),
    [Input("game-dropdown", "value")]
)
def set_display_children(selected_value):
    return f"Game {selected_value} selected"

if __name__ == "__main__":
    app.run(debug=True)
