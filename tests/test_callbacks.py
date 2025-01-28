# use __wrapped__ on callbacks to get underlying function

from dash import Dash

from callbacks import register_callbacks, teams


def test_update_figure_callback():
    app = Dash(__name__)
    register_callbacks(app)
    update_figure_callback = app.callback_map["away-dropdown.options"]["callback"].__wrapped__

    home = teams[0]  # should be ANA
    updated_options = update_figure_callback(home)

    assert home not in updated_options
