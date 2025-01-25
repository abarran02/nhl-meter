# Tidies and splits the PBP into more focused tables
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

playoff_dates = {
    2007: "2008-04-09",
    2008: "2009-04-15",
    2009: "2010-04-14",
    2010: "2011-04-13",
    2011: "2012-04-11",
    2012: "2013-04-30",
    2013: "2014-04-16",
    2014: "2015-04-15",
    2015: "2016-04-13",
    2016: "2017-04-12",
    2017: "2018-04-11",
    2018: "2019-04-10",
    2019: "2020-08-01",
    2020: "2021-05-20",
    2021: "2022-05-02",
    2022: "2023-04-17",
    2023: "2024-04-20"
}
playoff_dates = {key: datetime.strptime(value, "%Y-%m-%d") for key, value in playoff_dates.items()}

def add_season_column(pbp: pd.DataFrame):
    """Modifies pbp DataFrame in-place to determine season for each game.
    Notated for starting year of season (e.g. `2007` for 2007-2008 season).
    Uses August 31 as cutoff date between seasons.

    Args:
        pbp (pd.DataFrame): NHL play-by-play data
    """
    season_list = []
    season = 2007
    comparison_date = datetime.strptime('2008-08-31', '%Y-%m-%d')
    for row in pbp.itertuples(index=False):
        if row.Date > comparison_date:
            season += 1
            comparison_date += timedelta(days=365)

        season_list.append(season)
    pbp['Season'] = season_list

def reduce_pbp(pbp: pd.DataFrame):
    """Reduce pbp DataFrame in-place dropping player names and fixing goalie IDs.

    Args:
        pbp (pd.DataFrame): NHL play-by-play data
    """    
    # idk ask Alex Hagood
    # fix goalie weirdness
    pbp[['Home_Goalie_Id', 'Away_Goalie_Id']] = pbp[['Home_Goalie_Id', 'Away_Goalie_Id']].fillna(0).astype('int64')
    
    nameColumns = [f"{team}Player{num}" for team in ['home', 'away'] for num in range(1, 7)]
    pbp.drop(columns=nameColumns + ["Home_Goalie", "Away_Goalie"], inplace=True)

    pbp.columns = [re.sub(r'(away|home)Player(\d)_id', lambda m: f"{m.group(1).title()}_p{m.group(2)}", col) for col in pbp.columns]
    pbp.columns = [re.sub(r'p(\d)_(ID|name)', lambda m: f"p{m.group(1)}_{m.group(2).title()}", col) for col in pbp.columns]

def extract_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Extract game data including:
    
    Game ID, Season, Date, Home Team, Home Coach, Away Team, Away Coach, Event, Period, Home Score, Away Score, Playoff

    Args:
        pbp (pd.DataFrame): NHL play-by-play data

    Returns:
        pd.DataFrame: NHL game metadata
    """
    reduced = pbp[['Game_Id', 'Season', 'Date', 'Home_Team', 'Home_Coach', 'Away_Team', 'Away_Coach', 'Period', 'Home_Score', 'Away_Score']]

    # extract the last rows of each group and determine playoffs boolean
    # previously used GEND event but this is missing from some games, like 20003.2007
    games = []
    for name, game in reduced.groupby(['Season', 'Game_Id']):
        last_play = game.iloc[-1].copy()
        last_play['Playoff'] = last_play.Date >= playoff_dates[last_play.Season]
        games.append(last_play)

    return pd.DataFrame(games)

def extract_players(pbp: pd.DataFrame) -> pd.DataFrame:
    """Mapping between NHL player names and IDs

    Args:
        pbp (pd.DataFrame): NHL play-by-play data
    """
    players = [[f"homePlayer{pNum}", f"homePlayer{pNum}_id"] for pNum in range(1, 6)]
    players += [[f"awayPlayer{pNum}", f"awayPlayer{pNum}_id"] for pNum in range(1, 6)]
    players += [[f"p{pNum}_name", f"p{pNum}_ID"] for pNum in range(1, 4)]
    players += [['Home_Goalie', 'Home_Goalie_Id'], ['Away_Goalie', 'Away_Goalie_Id']]

    playerframe = pd.DataFrame()
    for pair in tqdm(players):
        playerframe = pd.concat([pbp[pair].rename(columns={pair[0]: 'player', pair[1]: 'playerId'}).drop_duplicates(), playerframe])
    playerframe.drop_duplicates(inplace=True)

    return playerframe

def tidy_pbp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pbp = df.copy()
    
    # type conversions for convenience
    pbp['Date'] = pd.to_datetime(pbp['Date'])
    pbp['Game_Id'] = pbp['Game_Id'].astype('int64')

    add_season_column(pbp)
    
    # must extract games and players before reduction
    games = extract_games(pbp)
    playerframe = extract_players(pbp)
    
    reduce_pbp(pbp)
    
    return (pbp, games, playerframe)

if __name__ == "__main__":
    # data folder path
    current_file_path = Path(__file__).resolve()
    data_path = current_file_path.parent / '..' / '..' / 'data'

    pbp = pd.read_parquet(data_path / 'pbp_combined.parquet')

    pbp, games, playerframe = tidy_pbp(pbp)
    
    games.to_parquet(data_path / 'games.parquet', index=False)
    playerframe.to_parquet(data_path / 'players.parquet', index=False)
    pbp.to_parquet(data_path / 'pbp_reduced.parquet', index=False)
