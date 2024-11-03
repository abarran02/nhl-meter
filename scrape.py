import hockey_scraper
import pandas as pd


def fix_player_id(pbp: pd.DataFrame, shifts: pd.DataFrame, game_id: str, player_name: str, name_col: str, id_col: str) -> pd.DataFrame:
    id_almost = shifts.loc[(shifts["Game_Id"] == game_id) & (shifts["Player"] == player_name)]  # rows matching given game with player name
    id = id_almost["Player_Id"].iloc[0]  # player has only one id, just grab first instance
    pbp.loc[(pbp["Game_Id"] == game_id) & (pbp[name_col] == player_name), id_col] = id  # replace all matching player null id with id from shift df
    return pbp

def fix_missing_ids(pbp: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    # build list of player column labels to fix IDs for all players on ice
    s = ["homePlayer", "awayPlayer"]
    players = [f"{team}{ct}" for team in s for ct in range(1,7)]

    for p in players:
        id_col = p + "_id"
        rows = pbp.loc[pbp[id_col].isnull() & pbp[p].notnull()]  # players with name but null id
        
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

if __name__ == "__main__":
    for y in range(2007,2024):
        try:
            scraped = hockey_scraper.scrape_seasons([y], True, data_format="pandas")
            # scraped = hockey_scraper.scrape_date_range('2017-12-08', '2017-12-08', True, data_format="pandas")  # includes four players with missing IDs for testing

            shifts = scraped["shifts"]
            pbp = scraped["pbp"]
            # pbp = fix_missing_ids(pbp, shifts)

            shifts['Period'] = pd.to_numeric(shifts['Period'], errors='coerce')
            pbp['Period'] = pd.to_numeric(pbp['Period'], errors='coerce')

            shifts.to_parquet(f"shift_{y}.parquet")
            pbp.to_parquet(f"game_{y}.parquet")
        except TimeoutError:
            continue
