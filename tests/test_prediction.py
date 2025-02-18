import json
import pytest

import pandas as pd
from keras.models import load_model

from dev import meter

# Load data and models
games = pd.read_parquet("data/games.parquet")
slices = pd.read_parquet("data/time_slices.parquet")
ot_pbp = pd.concat([
    pd.read_parquet("data/regular_ot_pbp.parquet"),
    pd.read_parquet("data/playoff_ot_pbp.parquet")
])
one_hot_columns = json.load(open("dev/models/one_hot_columns.json", 'r'))
model_regulation = load_model("dev/models/meter_lstm16d2.keras")
model_overtime = load_model("dev/models/meter_ot_lstm16d1.keras")

def test_regulation():
    errors = []
    for game in games.itertuples(index=False):
        try:
            meter.predict_regulation(game.Game_Id, game.Season, slices, model_regulation)
        except Exception as e:
            errors.append(f"Error for game {game.Game_Id}.{game.Season}: {e}")

    if errors:
        pytest.fail("\n".join(errors))

def test_overtime():
    errors = []
    for game in games.itertuples(index=False):
        if game.Period < 4:
            # limit to overtime games
            continue

        try:
            meter.predict_overtime(game.Game_Id, game.Season, ot_pbp, model_overtime, one_hot_columns)
        except Exception as e:
            errors.append(f"Error for game {game.Game_Id}.{game.Season}: {e}")

    if errors:
        pytest.fail("\n".join(errors))
