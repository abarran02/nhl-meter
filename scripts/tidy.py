# Tidies and splits the PBP into more focused tables
import pandas as pd
import re
from datetime import datetime, timedelta
from tqdm import tqdm

shifts = pd.read_parquet('./data/shift_combined.parquet')
pbp = pd.read_parquet('./data/pbp_combined.parquet')

pbp['Date'] = pd.to_datetime(pbp['Date'])

games = pbp[['Game_Id', 'Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach', 'Event', 'Period', "Home_Score", "Away_Score"]]
games = games[games["Event"] == "GEND"]
games.drop(columns=['Event'], inplace=True)

players = [[f"homePlayer{pNum}", f"homePlayer{pNum}_id"] for pNum in range(1, 6)]
players += [[f"awayPlayer{pNum}", f"awayPlayer{pNum}_id"] for pNum in range(1, 6)]
players += [[f"p{pNum}_name", f"p{pNum}_ID"] for pNum in range(1, 4)]

#fix goalie weirdness
pbp[['Home_Goalie_Id', 'Away_Goalie_Id']] = pbp[['Home_Goalie_Id', 'Away_Goalie_Id']].fillna(0).astype('int64')
players += [['Home_Goalie', 'Home_Goalie_Id'], ['Away_Goalie', 'Away_Goalie_Id']]


playerframe = pd.DataFrame()
for pair in tqdm(players):
    playerframe = pd.concat([pbp[pair].rename(columns={pair[0]: 'player', pair[1]: 'playerId'}).drop_duplicates(), playerframe])
playerframe.drop_duplicates(inplace=True)


pbp['Game_Id'] = pbp['Game_Id'].astype('int64')
nameColumns = [f"{team}Player{num}" for team in ['home', 'away'] for num in range(1, 7)]
pbp.drop(columns=nameColumns + ["Home_Goalie", "Away_Goalie"], inplace=True)

pbp.columns = [re.sub(r'(away|home)Player(\d)_id', lambda m: f"{m.group(1).title()}_p{m.group(2)}", col) for col in pbp.columns]
pbp.columns = [re.sub(r'p(\d)_(ID|name)', lambda m: f"p{m.group(1)}_{m.group(2).title()}", col) for col in pbp.columns]

season_list = []
season = 2007
comparison_date = datetime.strptime('2008-08-31', '%Y-%m-%d')
for row in games.itertuples(index=True):
    if row.Date > comparison_date:
        season += 1
        comparison_date += timedelta(days=365)

    season_list.append(season)

games['Season'] = season_list

games.to_parquet('./data/games.parquet', index=False)
playerframe.to_parquet('./data/players.parquet', index=False)
pbp.to_parquet('./data/pbp_reduced.parquet', index=False)
