# Tidies and splits the PBP into more focused tables
import pandas as pd

shifts = pd.read_parquet('../data/shift_combined.parquet')
pbp = pd.read_parquet('../data/pbp_combined.parquet')

games = pbp[['Game_Id', 'Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach']].drop_duplicates()
pbp.drop(columns=['Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach'], inplace=True)


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


games.to_parquet('games.parquet')
players.to_parquet('players.parquet')
pbp.to_parquet('pbp_reduced.parquet')