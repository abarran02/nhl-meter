import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from numpy import nan

def fix_player_id(pbp: pd.DataFrame, shifts: pd.DataFrame, game_id: str, player_name: str, name_col: str, id_col: str) -> pd.DataFrame:
    id_almost = shifts.loc[(shifts["Game_Id"] == game_id) & (shifts["Player"] == player_name)]  # rows matching given game with player name
    try:
        id = id_almost["Player_Id"].iloc[0]  # player has only one id, just grab first instance
    except IndexError:
        id = nan  # still missing

    pbp.loc[(pbp["Game_Id"] == game_id) & (pbp[name_col] == player_name), id_col] = id  # replace all matching player null id with id from shift df
    return pbp

def fix_missing_ids(pbp: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    # build list of player column labels to fix IDs for all players on ice
    s = ["homePlayer", "awayPlayer"]
    players = [f"{team}{ct}" for team in s for ct in range(1,7)]

    for p in players:
        id_col = p + "_id"
        rows = pbp.loc[(pbp[id_col].isnull()) & (pbp[p].notnull())]  # players with name but null id

        for i, r in rows.iterrows():
            pbp = fix_player_id(pbp, shifts, r["Game_Id"], r[p], p, id_col)

    # fix IDs for players involved in event
    for p in ["p1", "p2", "p3"]:
        id_col = p + "_ID"
        name_col = p + "_name"
        rows = pbp.loc[pbp[id_col].isnull() & pbp[name_col].notnull()]

        for i, r in rows.iterrows():
            if (name := r[p + "_name"]) == "Team":  # exclude team penalties like too many men
                continue

            pbp = fix_player_id(pbp, shifts, r["Game_Id"], name, name_col, id_col)

    return pbp

#  global scope folder path and name pattern
folder_path = './pbp'
pbp_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet') and 'game' in f]
shift_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet') and 'shift' in f]

def process_files(f_idx):
    pbp_df = pd.read_parquet(pbp_files[f_idx])
    shift_df = pd.read_parquet(shift_files[f_idx])

    # convert 'Period' column to string and ID columns to int
    pbp_df['Period'] = pbp_df['Period'].astype(int)
    shift_df['Period'] = shift_df['Period'].astype(int)
    for col in pbp_df.columns:
        if col.endswith('_id') or col.endswith('_ID'):
            pbp_df[col] = pbp_df[col].fillna(0).astype(int)

    pbp_df = fix_missing_ids(pbp_df, shift_df)
    return pbp_df, shift_df

if __name__ == "__main__":
    dataframes = {'pbp': [], 'shift': []}

    with ProcessPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(process_files, range(len(pbp_files))), total=len(pbp_files)))

    for pbp_df, shift_df in results:
        dataframes["pbp"].append(pbp_df)
        dataframes["shift"].append(shift_df)

    for k, v in dataframes.items():
        pd.concat(v).to_parquet(f'{k}_combined.parquet')
