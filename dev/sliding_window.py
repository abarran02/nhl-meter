import pandas as pd
from tqdm import tqdm
import numpy as np


def plays_preceding_goals(df: pd.DataFrame, window_size: int, include_goal: bool = True, convert_winner: bool = False) -> pd.DataFrame:
    goal_indices = df.index[df['event'] == 'GOAL']

    result_indices = []
    for idx in goal_indices:
        # may include GOAL row for use with sliding window
        end = idx + 1 if include_goal else idx
        window = range(max(idx - window_size, 0), end)

        if convert_winner:
            # set winner column to GOAL scorer
            df.loc[window]["winner"] = df.loc[window]["winner"][-1]

        result_indices.extend(window)

    return df.loc[result_indices]

def sliding_window_on_groups(grouped: pd.DataFrame, window_size: int) -> tuple[np.ndarray, np.ndarray]:
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
