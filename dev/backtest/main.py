import pandas as pd
import numpy as np
import math

class NHL_METER:
    def __init__(self):
        # self.games = pd.read_parquet('../data/games.parquet')
        # self.pbp = pd.read_parquet('../data/pbp_reduced.parquet')
        # self.games["Game_Id"] = self.games["Game_Id"].astype("Int64")
        pass

    def get_pbp(self, game_id, season):
        return self.pbp[(self.pbp['Game_Id'] == game_id) & (self.pbp['Season'] == season)]
    
    def add_won_column(self, games, pbp):
        pbp_w = pd.merge(pbp, games[["Season", "Game_Id", "Away_Score", "Home_Score"]], how='left', on=['Game_Id', 'Season'], suffixes = ["", "_Final"])
        pbp_w["Home_Won"] = pbp_w["Away_Score_Final"] < pbp_w["Home_Score_Final"]
        pbp_w.drop(columns=["Away_Score_Final", "Home_Score_Final"], inplace=True)
        return pbp_w
    
    # create some random winprob for testing yknow
    def random_winprob(self, pbp):
        pbp["Win_Prob"] = np.random.rand(pbp.shape[0])
        return pbp
    
    def bin_winprob(self, pbp, binCount=10):
        bins = np.linspace(0, 1, binCount)
        digitized = np.digitize(pbp["Win_Prob"], bins)
        pbp["bin"] = digitized
        binned = []
        for x in range(binCount):
            binned.append(pbp[pbp["bin"] == x])
        return binned
    
    # returns [(actual, predicted)]
    # pass in result from bin_winprob
    def analyze_accuracy(self, binned, nans=False):
        trueWin = []
        for i, df in enumerate(binned):
            winProp = df["Home_Won"].mean()
            if (math.isnan(winProp) and not nans):
                winProp = 0.0
            est = (i + .5) / len(binned)
            #print(f"Estimated: {est}, Actual: {winProp}")
            trueWin.append((est, winProp))
        return trueWin
    
    def accBins(self, data, bc=10):
        return self.analyze_accuracy(self.bin_winprob(data, bc), nans=True)
    
    def least_squares(self, analyzed):
        return sum([(x[0]-x[1])**2 for x in analyzed])

    def log_loss(self, analyzed):
        total = 0
        for predicted, actual  in analyzed:
            total += (actual * math.log(predicted) + (1 - actual) * math.log(1 - predicted))
        res = -1 / len(analyzed)
        return res * total
    
    def ranked_probability(self, data):
        all_rps = []
        for i, row in data.iterrows():
            rps = 1/2 * (row["Win_Prob"] - row["Home_Won"])**2
            all_rps.append(rps)
        return sum(all_rps) / len(data)
    
    def backtest(self, plays, loss_func):
        binned_plays = self.bin_winprob(plays, binCount=100)
        analyzed = self.analyze_accuracy(binned_plays)
        return loss_func(analyzed)
