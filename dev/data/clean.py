# for the current ML-focused state of this project, fixing IDs is not necessary as the model ignores individual players
# my implementation is also extremely slow, which is why I had to throw 12 cores at it

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from numpy import nan
from tqdm import tqdm


def fix_player_id(pbp: pd.DataFrame, shifts: pd.DataFrame, game_id: str, player_name: str, name_col: str, id_col: str) -> pd.DataFrame:
    id_almost = shifts.loc[(shifts['Game_Id'] == game_id) & (shifts['Player'] == player_name)]  # rows matching given game with player name
    try:
        id = id_almost['Player_Id'].iloc[0]  # player has only one id, just grab first instance
    except IndexError:
        id = nan  # still missing

    pbp.loc[(pbp['Game_Id'] == game_id) & (pbp[name_col] == player_name), id_col] = id  # replace all matching player null id with id from shift df
    return pbp

def fix_missing_ids(pbp: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    # build list of player column labels to fix IDs for all players on ice
    s = ['homePlayer', 'awayPlayer']
    players = [f'{team}{ct}' for team in s for ct in range(1,7)]

    for p in players:
        id_col = p + '_id'
        rows = pbp.loc[(pbp[id_col].isnull()) & (pbp[p].notnull())]  # players with name but null id

        for row in rows.itertuples(index=False):
            pbp = fix_player_id(pbp, shifts, row.Game_Id, getattr(row, p), p, id_col)

    # fix IDs for players involved in event
    for p in ['p1', 'p2', 'p3']:
        id_col = p + '_ID'
        name_col = p + '_name'
        rows = pbp.loc[pbp[id_col].isnull() & pbp[name_col].notnull()]

    for row in rows.itertuples(index=False):  # Use index=True to access the row index if needed
        if (name := getattr(row, p + '_name')) == 'Team':  # Exclude team penalties
            continue

        pbp = fix_player_id(pbp, shifts, row.Game_Id, name, name_col, id_col)

    return pbp

def clean_season(pbp_df: pd.DataFrame, shift_df: pd.DataFrame, fix_ids: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    # convert 'Period' column to string and ID columns to int
    pbp_df['Period'] = pbp_df['Period'].astype(int)
    shift_df['Period'] = shift_df['Period'].astype(int)
    for col in pbp_df.columns:
        if col.endswith('_id') or col.endswith('_ID'):
            pbp_df[col] = pbp_df[col].fillna(0).astype(int)

    if fix_ids:
        pbp_df = fix_missing_ids(pbp_df, shift_df)

    return pbp_df, shift_df

if __name__ == '__main__':
    dataframes = {'pbp': [], 'shift': []}

    current_file_path = Path(__file__).resolve()
    data_path = current_file_path.parent / '..' / '..' / 'data'
    pbp_path = data_path / 'pbp'

    pbp_files = [f for f in Path(pbp_path).iterdir() if f.suffix == '.parquet' and 'game' in f.name]
    shift_files = [f for f in Path(pbp_path).iterdir() if f.suffix == '.parquet' and 'shift' in f.name]

    # fix_ids True, will be slow
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = executor.map(clean_season, pbp_files, shift_files)
        results = list(tqdm(futures, total=len(pbp_files)))

    for pbp_df, shift_df in results:
        dataframes['pbp'].append(pbp_df)
        dataframes['shift'].append(shift_df)

    for k, v in dataframes.items():
        pd.concat(v).to_parquet(data_path / f'{k}_combined.parquet')
