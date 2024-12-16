import pandas as pd
from elosports.elo import Elo

def add_and_run_elo_by_season(df: pd.DataFrame) -> pd.DataFrame:
    # add Elo columns and initialize all to zero
    df["Away_Starting_Elo"] = 0.0
    df["Home_Starting_Elo"] = 0.0
    df["Away_Ending_Elo"] = 0.0
    df["Home_Ending_Elo"] = 0.0

    # create league
    league = Elo(k = 20)
    for t in df['Home_Team'].unique():
        league.addPlayer(t)

    season = df["Season"].min()  # likely 2007
    for row in df.itertuples(index=True):
        # see https://github.com/ddm7018/Elo/blob/master/tutorial/elo_simulations.py
        if row.Season > season:
            for key in league.ratingDict.keys():
                # year that Thrashers moved to Winnipeg, copy Elo
                if key == 'WPG' and season == 2011:
                    league.ratingDict['WPG'] = league.ratingDict['ATL']
                league.ratingDict[key] = league.ratingDict[key] - ((league.ratingDict[key] - 1500) * (1/3.))
            season += 1

        df.loc[row.Index, 'Away_Starting_Elo'] = league.ratingDict[row.Away_Team]
        df.loc[row.Index, 'Home_Starting_Elo'] = league.ratingDict[row.Home_Team]

        if row.Away_Score > row.Home_Score:
            league.gameOver(row.Away_Team, row.Home_Team, False)
        else:
            league.gameOver(row.Home_Team, row.Away_Team, True)

        df.loc[row.Index, 'Away_Ending_Elo'] = league.ratingDict[row.Away_Team]
        df.loc[row.Index, 'Home_Ending_Elo'] = league.ratingDict[row.Home_Team]

    return df

if __name__ == "__main__":
    df = pd.read_parquet('./scripts/data/games.parquet')
    df.to_parquet('./scripts/data/game_elo.parquet', index=False)
