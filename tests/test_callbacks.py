from typing import Callable

import pytest
from dash import Dash
from plotly.graph_objects import Figure

from callbacks import register_callbacks, teams


def get_callback(app: Dash, callback: str) -> Callable:
    # use __wrapped__ on callbacks to get underlying function
    return app.callback_map[callback]["callback"].__wrapped__

def test_update_away_dropdown():
    app = Dash(__name__)
    register_callbacks(app)
    update_away_callback = get_callback(app, "away-dropdown.options")

    home = teams[0]  # should be ANA
    updated_options = update_away_callback(home)

    assert home not in updated_options

def test_update_game_dropdown():
    app = Dash(__name__)
    register_callbacks(app)
    update_game_callback = get_callback(app, "game-dropdown.options")

    home = "ANA"
    away = "PHI"  # both arbitrary
    updated_games = update_game_callback(home, away)

    assert updated_games  # should not be empty

def test_update_game_dropdown_empty():
    app = Dash(__name__)
    register_callbacks(app)
    update_game_callback = get_callback(app, "game-dropdown.options")

    home = "ATL"
    away = "SEA"  # teams that never played
    updated_games = update_game_callback(home, away)

    assert not updated_games  # should be empty

def test_update_figure_regulation():
    app = Dash(__name__)
    register_callbacks(app)
    update_figure_callback = get_callback(app, "probability-graph.figure")

    home = "BOS"
    away = "PHI"
    game_season = "30227.2009"
    try:
        update_figure_callback(home, away, game_season)
    except Exception as e:
        pytest.fail(e)

def test_update_figure_overtime():
    app = Dash(__name__)
    register_callbacks(app)
    update_figure_callback = get_callback(app, "probability-graph.figure")

    home = "CAR"
    away = "FLA"
    game_season = "30311.2022"
    try:
        update_figure_callback(home, away, game_season)
    except Exception as e:
        pytest.fail(e)

def test_update_figure_empty():
    app = Dash(__name__)
    register_callbacks(app)
    update_figure_callback = get_callback(app, "probability-graph.figure")

    home = "BOS"
    away = "PHI"      # both arbitrary
    game_season = ""  # none selected
    updated_figure = update_figure_callback(home, away, game_season)

    assert updated_figure == Figure()  # blank Figure
