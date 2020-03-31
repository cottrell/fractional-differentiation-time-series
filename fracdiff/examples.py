"""
From Prado Ch. 5

In practice, I suggest you experiment with the following transformation of your
features: First, compute a cumulative sum of the time series. This guarantees that
some order of differentiation is needed. Second, compute the FFD(d) series for var-
ious d ∈ [0, 1]. Third, determine the minimum d such that the p-value of the ADF
statistic on FFD(d) falls below 5%. Fourth, use the FFD(d) series as your predictive
feature.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller

from .fracdiff import fast_frac_diff, frac_diff_ffd, get_weights_ffd, get_weights


def get_data(symbols="aapl", start="2010-01-03"):
    return pdr.yahoo.daily.YahooDailyReader(symbols=symbols, start=start).read()


def example(symbols="aapl", df=None):
    if df is None:
        df = get_data(symbols=symbols)
    s = np.log(df["Adj Close"])
    l = plot_min_ffd(s)
    return locals()


def plot_weights():
    fig = plt.figure(1)
    fig.clf()
    fig.gca().set_title('d in [0, 1]')
    _plot_weights(dRange=[0, 1], nPlots=6, size=6)
    fig = plt.figure(2)
    fig.clf()
    fig.gca().set_title('d in [1, 2]')
    _plot_weights(dRange=[1, 2], nPlots=6, size=6)


def _plot_weights(dRange, nPlots, size):
    w = list()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w.append(w_)
    w = pd.concat(w, axis=1)
    ax = plt.gca()
    w.plot(ax=plt.gca())
    ax.legend(loc="upper left")
    plt.show()
    return


def plot_min_ffd(s):
    """Finding the minimum d value that passes the ADF test.
    Input s should be prices."""

    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])
    for d in np.linspace(0, 2, 50):
        # df1 -> x, df2 -> dx
        x = np.log(s.values)
        dx = frac_diff_ffd(x, d, thres=0.01)
        dx = np.nan_to_num(dx, 0)
        # dx = fast_frac_diff(x, d)
        corr = np.corrcoef(x, dx)[0, 1]
        adf = adfuller(dx, maxlag=1, regression="c", autolag=None)
        out.loc[d] = list(adf[:4]) + [adf[4]["5%"]] + [corr]  # with critical value
    fig = plt.figure(1)
    fig.clf()
    plt.ion()
    out[["adfStat", "corr"]].plot(secondary_y="adfStat", ax=fig.gca(), style=".-")
    plt.axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    plt.show()
    return locals()
