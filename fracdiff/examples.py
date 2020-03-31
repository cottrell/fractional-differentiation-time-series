"""
From Prado Ch. 5

In practice, I suggest you experiment with the following transformation of your
features: First, compute a cumulative sum of the time series. This guarantees that
some order of differentiation is needed. Second, compute the FFD(d) series for var-
ious d ∈ [0, 1]. Third, determine the minimum d such that the p-value of the ADF
statistic on FFD(d) falls below 5%. Fourth, use the FFD(d) series as your predictive
feature.
"""

import numpy as np
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller

from .fracdiff import fast_frac_diff, frac_diff_ffd


def get_data(symbols="aapl", start="2010-01-03"):
    return pdr.yahoo.daily.YahooDailyReader(symbols=symbols, start=start).read()


def example(df=None):
    if df is None:
        df = get_data()
    s = np.log(df["Adj Close"])
    return s


def plot_min_ffd(s):
    """Finding the minimum d value that passes the ADF test"""
    import matplotlib.pyplot as plt
    import pandas as pd

    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])
    for d in np.linspace(0, 2, 11):
        # df1 -> x, df2 -> dx
        x = np.log(s)
        # dx = frac_diff_ffd(x, d, thres=0.01)
        dx = fast_frac_diff(x, d)
        corr = np.corrcoef(x, dx)[0, 1]
        adf = adfuller(dx, maxlag=1, regression="c", autolag=None)
        out.loc[d] = list(adf[:4]) + [adf[4]["5%"]] + [corr]  # with critical value
    fig = plt.figure(1)
    fig.clf()
    plt.ion()
    out[["adfStat", "corr"]].plot(secondary_y="adfStat", ax=fig.gca())
    plt.axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    plt.show()
    return locals()


def old_main():
    # from the forked repo
    import pandas as pd
    from utils import plot_multi

    close = pd.read_csv("sp500.csv", index_col=0, parse_dates=True)[["Close"]]
    close = close["1993":]
    import matplotlib.pyplot as plt

    fracs = frac_diff_ffd(close.apply(np.log), d=0.4, thres=1e-5)
    a = pd.DataFrame(data=np.transpose([np.array(fracs), close["Close"].values]), columns=["Fractional differentiation FFD", "SP500"])

    # burn the first 1500 days where the weights are not defined.
    plot_multi(a[1500:])
    plt.show()
