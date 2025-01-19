from dash import Dash
from layout import app_layout
from callbacks import register_callbacks

app = Dash(title="NHL Meter")
app.layout = app_layout

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
