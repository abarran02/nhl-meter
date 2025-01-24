import pandas as pd
import numpy as np


def plays_preceding_goals(pbp: pd.DataFrame, window_size: int, include_goal: bool = True, convert_winner: bool = False) -> pd.DataFrame:
    """
    Reduces play-by-play DataFrame to plays preceding each goal.

    Args:
        pbp (pd.DataFrame): NHL play-by-play data.
        window_size (int): Number of plays preceding goal to include.
        include_goal (bool, optional): Whether to include the GOAL play. Defaults to True.
        convert_winner (bool, optional): For regular season games, converts the `winner`
            column to the team of each goal scored. Also converts game IDs to random,
            negative values so that each goal is treated as a separate game. Defaults to False.

    Returns:
        pd.DataFrame: Reduced DataFrame.
    """

    goal_indices = pbp.index[pbp['event'] == 'GOAL']

    windows = []
    targets = []

    result_indices = []
    for idx in goal_indices:
        # may include GOAL row for use with sliding window
        end = idx + 1 if include_goal else idx
        window = range(max(idx - window_size, 0), end)

        if convert_winner:
            windows.append(window)
            targets.append(pbp.iloc[window[-1]]["team"])

        result_indices.extend(window)

    if not convert_winner:
        return pbp.loc[result_indices]
    else:
        # prevent changing original dataframe
        conv = pbp.copy()

        # change winner column in each window to goal scorer
        # convert game ID to arbitrary value
        winners = np.array(pbp["winner"].values)
        ids = np.array(pbp["game"].values)

        # use negative index as "game ID" since real IDs are positive
        for idx, (rng, value) in enumerate(zip(windows, targets)):
            winners[list(rng)] = value
            ids[list(rng)] = -idx

        # set DataFrame columns
        conv["winner"] = winners
        conv["game"] = ids

        return conv.loc[result_indices]

def sliding_window_game_pbp(pbp: pd.DataFrame, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform sliding window preprocessing on games' play-by-play data.

    Args:
        pbp (pd.DataFrame): NHL play-by-play data.
        window_size (int): Size of the sliding window.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of play-by-play windows and targets (winners), respectively.
    """
    grouped = pbp.groupby(["season", "game"])  # unnecessary for individual games but don't want to cross over
    windows = []
    targets = []

    for group_name, group in grouped:
        temp_window = []
        target = group["winner"].iloc[0]  # same for all in group

        for row in group.drop("winner", axis=1).itertuples(index=False):
            feature_values = list(row)[2:]  # skip season and game columns
            temp_window.append(feature_values)

            if len(temp_window) == window_size:
                windows.append(temp_window.copy())
                targets.append(target)
                temp_window.pop(0)

    return np.array(windows), np.array(targets)

def predict_regulation(game: int, season: int, slices: pd.DataFrame, model) -> tuple[pd.Series, np.ndarray]:
    """Predict regulation win probabilities for given NHL game.

    Args:
        game (int): NHL Game ID
        season (int): NHL season
        ot_pbp (pd.DataFrame): regulation time slice data
        model (Model): keras Model

    Returns:
        tuple[pd.Series, np.ndarray]: Time series and win probability
    """
    selected_game = slices[(slices["game"] == game) & (slices["season"] == season)]

    X = selected_game.drop(columns=["winner", "game", "season"])

    probabilities = model.predict(X)

    # convert from normalized 1 to 0
    time_elapsed = 3600 - (selected_game["time_remaining"] * 3600)

    return (time_elapsed, probabilities.flatten())

def predict_overtime(game: int, season: int, ot_pbp: pd.DataFrame, model, one_hot_columns: list[str]) -> tuple[pd.Series, np.ndarray]:
    """Predict overtime win probabilities for given NHL game.

    Args:
        game (int): NHL Game ID
        season (int): NHL season
        ot_pbp (pd.DataFrame): overtime play-by-play data
        model (Model): keras Model
        one_hot_columns (list[str]): one-hot encoding columns from training

    Returns:
        tuple[pd.Series, np.ndarray]: Time series and win probability
    """
    window_size = 3
    mask = (ot_pbp["game"] == game) & (ot_pbp["season"] == season)
    selected_game = ot_pbp[mask]

    X = selected_game.drop(["seconds_elapsed"], axis=1)
    # overtime finished in 2 plays? (minimum FAC, then GOAL)
    while len(X) < window_size:
        # copy game, season, and Elo columns
        # generate remaining blank row and prepend
        blank_row = X.iloc[:, :4].copy()
        for col in X.columns[4:]:
            blank_row[col] = None

        X = pd.concat([blank_row, X], ignore_index=True)

    # one-hot encode play-by-play and match columns to training data
    X_encoded = pd.get_dummies(X, columns=["event", "team", "event_zone", "home_zone", "strength"])
    X_encoded = X_encoded.reindex(columns=one_hot_columns, fill_value=False)

    windows, targets = sliding_window_game_pbp(X_encoded, window_size)

    probabilities = model.predict(windows)

    return (3600 + selected_game["seconds_elapsed"], probabilities.flatten())
