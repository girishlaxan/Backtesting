from backtesting import Backtest, Strategy
from backtesting.test import GOOG
from backtesting.lib import crossover, plot_heatmaps, resample_apply
import matplotlib.pyplot as plt
import seaborn as sns

import talib

# print(GOOG)

def optim_func(series):
    if series["# Trades"] < 10:
        return -1

    return series["Equity Final [$]"] / series["Exposure Time [%]"]

class RsiOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    rsi_window = 14

    def init(self):
        self.daily_rsi = self.I(talib.RSI, self.data.Close, self.rsi_window)
        # multi time frame strategies
        # self.weekly_rsi = resample_apply(
        #     "W-FRI", talib.RSI, self.data.Close, self.rsi_window
        # )
    
    def next(self):
        if crossover(self.daily_rsi, self.upper_bound):
            self.position.close()  

        elif crossover(self.lower_bound, self.daily_rsi):
            self.buy()

bt = Backtest(GOOG, RsiOscillator, cash = 10_000)
stats, heatmap = bt.optimize(
    upper_bound = range(55, 85, 5),
    lower_bound = range(10, 45, 5),
    rsi_window = range(10, 30, 2),
    maximize= "Sharpe Ratio",
    constraint= lambda param: param.upper_bound > param.lower_bound,
    # max_tries = 100 // to deal with very huge sets of configs
    return_heatmap= True,

)

stats = bt.run()
# print(heatmap)
# print(hm)
print(stats)

# Method 1:
hm = heatmap.groupby(["upper_bound", "lower_bound"]).mean().unstack()
sns.heatmap(hm, cmap="plasma")
plt.show()

# Method 2:
# plot_heatmaps(heatmap, agg="mean")


bt.plot()   