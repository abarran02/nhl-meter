import pandas as pd
from tqdm import tqdm
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

    for group_name, group in tqdm(grouped, total=len(grouped)):
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
