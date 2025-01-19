import pandas as pd
from dev.analysis.main import NHL_METER
from elosports.elo import Elo
import matplotlib.pyplot as plt

df = pd.read_parquet('../data/game_elo.parquet')
df["Home_Won"] = df["Home_Score"] > df["Away_Score"]
df["Win_Prob"] = Elo.expectResult(Elo, df["Home_Starting_Elo"], df["Away_Starting_Elo"])


m = NHL_METER()

print(f"Sum of Least Squares: {m.backtest(df, m.least_squares):.4f}", )
print(f"Cross Entropy: {m.backtest(df, m.log_loss):.4f}")
print(f"Ranked Probability Score: {m.ranked_probability(df):.4f}")



xy = [list(z) for z in m.accBins(df, 100)]
dy = [x - y for x, y in xy]
x, y = zip(*xy)
x = list(x)
y = list(y)
dy = [y + dy for y, dy in zip(y, dy)]

plt.figure
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.plot(y, y, color='black')
ax.scatter(x, y, color='red')
ax.vlines(x, y, dy, color='black')
plt.title("Binned Elo Win Probability Residuals")

plt.show()