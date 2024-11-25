# Tidies and splits the PBP into more focused tables
import pandas as pd
import re

shifts = pd.read_parquet('./data/shift_combined.parquet')
pbp = pd.read_parquet('./data/pbp_combined.parquet')

games = pbp[['Game_Id', 'Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach']].drop_duplicates()
pbp.drop(columns=['Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach'], inplace=True)


gend_events = pbp[(pbp['Event'] == 'GEND')]
gend_events = gend_events[['Game_Id', 'Period', 'Away_Score', 'Home_Score']]
games = games.merge(gend_events, on='Game_Id', how='left')

players = [[f"homePlayer{pNum}", f"homePlayer{pNum}_id"] for pNum in range(1, 6)]
players += [[f"awayPlayer{pNum}", f"awayPlayer{pNum}_id"] for pNum in range(1, 6)]
players += [[f"p{pNum}_name", f"p{pNum}_ID"] for pNum in range(1, 4)]

#fix goalie weirdness
pbp[['Home_Goalie_Id', 'Away_Goalie_Id']] = pbp[['Home_Goalie_Id', 'Away_Goalie_Id']].fillna(0).astype('int64')
players += [['Home_Goalie', 'Home_Goalie_Id'], ['Away_Goalie', 'Away_Goalie_Id']]


playerframe = pd.DataFrame()
for pair in players:
     playerframe = pd.concat([pbp[pair].rename(columns={pair[0]: 'player', pair[1]: 'playerId'}).drop_duplicates(), playerframe])
playerframe.drop_duplicates(inplace=True)


pbp['Game_Id'] = pbp['Game_Id'].astype('int64')
nameColumns = [f"{team}Player{num}" for team in ['home', 'away'] for num in range(1, 7)]
pbp.drop(columns=nameColumns + ["Home_Goalie", "Away_Goalie"], inplace=True)

pbp.columns = [re.sub(r'(away|home)Player(\d)_id', lambda m: f"{m.group(1).title()}_p{m.group(2)}", col) for col in pbp.columns]
pbp.columns = [re.sub(r'p(\d)_(ID|name)', lambda m: f"p{m.group(1)}_{m.group(2).title()}", col) for col in pbp.columns]


games.to_parquet('./data/games.parquet')
playerframe.to_parquet('./data/players.parquet')
pbp.to_parquet('./data/pbp_reduced.parquet')