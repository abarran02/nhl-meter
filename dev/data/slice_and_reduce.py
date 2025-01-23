import copy
import re
from pathlib import Path
from typing import Callable, NamedTuple

import pandas as pd
from tqdm import tqdm


def convert_strength_to_int(strength: str) -> int:
    players = strength.split('x')
    # home - away
    return int(players[0]) - int(players[1])

def slice_regulation(games: pd.DataFrame, pbp: pd.DataFrame, slice_length: int = 30) -> pd.DataFrame:
    slices = []

    for game in tqdm(games.itertuples(index=False), total=len(games)):
        cur_cutoff = slice_length

        game_totals = {
            "game": game.Game_Id,
            "season": game.Season,
            "time_remaining": 1,  # three periods of 20 minutes, in seconds, divided by 3600
            "away_elo": game.Away_Starting_Elo,
            "home_elo": game.Home_Starting_Elo,
            "away_score": 0,
            "home_score": 0,
            "away_pim": 0,
            "home_pim": 0,
            "away_hits": 0,
            "home_hits": 0,
            "away_shots": 0,
            "home_shots": 0,
            "strength": 0,
            "winner": "home" if game.Home_Score > game.Away_Score else "away"
        }

        slices.append(copy.deepcopy(game_totals))  # initial based purely on Elo

        reduced = pbp[(pbp["Game_Id"] == game.Game_Id) & (pbp["Date"] == game.Date)]
        for play in reduced.itertuples(index=False):
            elapsed = play.Seconds_Elapsed + ((play.Period - 1) * 1200)
            if elapsed > 3600:
                # ignore overtime, will use different model and concatenate
                break

            if elapsed >= cur_cutoff:
                # convert to normalized time remaining
                game_totals["time_remaining"] = (3600 - cur_cutoff) / 3600
                slices.append(copy.deepcopy(game_totals))
                cur_cutoff += slice_length

            game_totals["strength"] = convert_strength_to_int(play.Strength)  # always update strength

            if play.Ev_Team == game.Home_Team:
                team = "home"
            elif play.Ev_Team == game.Away_Team:
                team = "away"
            else:
                # timing event like PSTR or GEND, or STOP for rink repair, etc
                continue

            match play.Event:
                case "SHOT":
                    game_totals[f"{team}_shots"] += 1
                case "HIT":
                    game_totals[f"{team}_hits"] += 1
                case "PENL":
                    try:
                        text = play.Type.split("(")[1]  # some penalty descriptions have player number
                        if "maj" in text:
                            game_totals[f"{team}_pim"] += 5
                        else:
                            mins = re.search(r'\d+', text)
                            game_totals[f"{team}_pim"] += int(mins.group())
                    except (IndexError, AttributeError):
                        pass  # some penalties are missing descriptions
                case "GOAL":
                    game_totals[f"{team}_score"] += 1

    return pd.DataFrame(slices)

def reduce_plays(game: NamedTuple, pbp: pd.DataFrame, time_key: str, time_function: Callable[[int, int], int]) -> list[dict]:
    valid_events = ["FAC", "BLOCK", "SHOT", "GOAL", "MISS", "HIT", "GIVE", "TAKE"]

    events = []
    for play in pbp.itertuples(index=False):
        # first event will always be a faceoff
        if play.Event in valid_events:
            event = {
                "game": game.Game_Id,
                "season": game.Season,
                "away_elo": game.Away_Starting_Elo,
                "home_elo": game.Home_Starting_Elo,
                time_key: time_function(play.Period, play.Seconds_Elapsed),
                "event": play.Event,
                "team": "home" if play.Ev_Team == play.Home_Team else "away",
                "event_zone": play.Ev_Zone,
                "home_zone": play.Home_Zone,
                "strength": play.Strength,
                "winner": "home" if game.Home_Score > game.Away_Score else "away"
            }

            events.append(event)

    return events

def reduce_regulation(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    events = []
    for game in tqdm(games.itertuples(index=False), total=len(games)):
        mask = ((pbp["Game_Id"] == game.Game_Id)
                & (pbp["Season"] == game.Season)
                & (pbp["Period"] < 4))
        reduced = pbp[mask]

        time_remaining = lambda x, y: 3600 - y + ((x - 1) * 1200)
        events.extend(reduce_plays(game, reduced, "time_remaining", time_remaining))

    return pd.DataFrame(events)

def reduce_regular_overtime(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    events = []
    for game in tqdm(games.itertuples(index=False), total=len(games)):
        mask = ((pbp["Game_Id"] == game.Game_Id)
                & (pbp["Season"] == game.Season)
                & (pbp["Period"] == 4))  # ignore shootout
        reduced = pbp[mask]

        seconds_elapsed = lambda x, y: y
        events.extend(reduce_plays(game, reduced, "seconds_elapsed", seconds_elapsed))

    return pd.DataFrame(events)

def reduce_playoff_overtime(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    events = []
    for game in tqdm(games.itertuples(index=False), total=len(games)):
        mask = ((pbp["Game_Id"] == game.Game_Id)
                & (pbp["Season"] == game.Season)
                & (pbp["Period"] >= 4))
        reduced = pbp[mask]

        seconds_elapsed =  lambda x, y: ((x - 4) * 1200) + y
        events.extend(reduce_plays(game, reduced, "seconds_elapsed", seconds_elapsed))

    return pd.DataFrame(events)

if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    data_path = current_file_path.parent / '..' / '..' / 'data'

    games = pd.read_parquet(data_path / "game_elo.parquet")
    pbp = pd.read_parquet(data_path / "pbp_reduced.parquet")

    slices = slice_regulation(games, pbp)
    slices.to_parquet(data_path / "time_slices.parquet")

    regulation = games[games["Period"] < 4]
    events = reduce_regulation(regulation, pbp)
    events.to_parquet(data_path / "regulation_pbp.parquet")

    regular_ot = games[(games["Period"] >= 4) & (games["Playoff"] == False)]
    events = reduce_regular_overtime(regular_ot, pbp)
    events.to_parquet(data_path / "regular_ot_pbp.parquet")

    playoff_ot = games[(games["Period"] >= 4) & (games["Playoff"] == True)]
    events = reduce_playoff_overtime(playoff_ot, pbp)
    events.to_parquet(data_path / "playoff_ot_pbp.parquet")
