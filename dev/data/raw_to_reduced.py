# if there is any error, the cleaning scripts may need to be run individually

from pathlib import Path

import clean
import elo
import pandas as pd
import slice_and_reduce
import tidy

current_file_path = Path(__file__).resolve()
data_path = current_file_path.parent / '..' / '..' / 'data'

# clean
print("Importing and concatenating season files...")
pbp_path = data_path / 'pbp'
pbp_files = [f for f in Path(pbp_path).iterdir() if f.suffix == '.parquet' and 'game' in f.name]
shift_files = [f for f in Path(pbp_path).iterdir() if f.suffix == '.parquet' and 'shift' in f.name]

pbp_list = []

for pbp, shift in zip(pbp_files, shift_files):
    pbp_df, shift_df = clean.clean_season(
        pd.read_parquet(pbp),
        pd.read_parquet(shift),
        fix_ids=False
    )

    pbp_list.append(pbp_df)

pbp = pd.concat(pbp_list, ignore_index=True)
pbp.to_parquet(data_path / 'pbp_combined.parquet')

# tidy
print("Tidying games...")
pbp = pd.read_parquet(data_path / 'pbp_combined.parquet')

pbp, games, playerframe = tidy.tidy_pbp(pbp)

games.to_parquet(data_path / 'games.parquet', index=False)
playerframe.to_parquet(data_path / 'players.parquet', index=False)
pbp.to_parquet(data_path / 'pbp_reduced.parquet', index=False)

print("Generating Elo...")
# generate Elo column
games = elo.add_and_run_elo_by_season(games)
games.to_parquet(data_path / 'game_elo.parquet', index=False)

# slice and reduce
print("Generating regulation game state slices...")
slices = slice_and_reduce.slice_regulation(games, pbp)
slices.to_parquet(data_path / "time_slices.parquet")

print("Reducing play-by-play for overtime training...")
regulation = games[games["Period"] < 4]
events = slice_and_reduce.reduce_regulation(regulation, pbp)
events.to_parquet(data_path / "regulation_pbp.parquet")

regular_ot = games[(games["Period"] >= 4) & (games["Playoff"] == False)]
events = slice_and_reduce.reduce_regular_overtime(regular_ot, pbp)
events.to_parquet(data_path / "regular_ot_pbp.parquet")

playoff_ot = games[(games["Period"] >= 4) & (games["Playoff"] == True)]
events = slice_and_reduce.reduce_playoff_overtime(playoff_ot, pbp)
events.to_parquet(data_path / "playoff_ot_pbp.parquet")
