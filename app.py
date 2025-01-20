from dash import Dash
from layout import app_layout
from callbacks import register_callbacks

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
]

app = Dash(
    title="NHL Meter",
    external_stylesheets=external_stylesheets
)
app.layout = app_layout

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
