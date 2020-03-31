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

import os

from joblib import Memory

_joblib_dir = os.path.expanduser('~/joblib_cache')
memory = Memory(_joblib_dir, verbose=10)

@memory.cache
def get_data(symbols="aapl", start="2010-01-03"):
    return pdr.yahoo.daily.YahooDailyReader(symbols=symbols, start=start).read()


def example_1(symbols="aapl", df=None):
    if df is None:
        df = get_data(symbols=symbols)
    s = np.log(df["Adj Close"])
    x = s.values
    xdf = pd.Series(x, index=s.index)
    plt.ion()
    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    xdf.plot(ax=ax, alpha=1, style='--', color='k')
    ax = ax.twinx()
    for d in np.linspace(0.1, 1.3, 13):
        d = (d * 10).round() / 10
        dx = frac_diff_ffd(x, d)
        dx = pd.Series(dx, index=s.index)
        dx.plot(ax=ax, alpha=0.5, label=f'd={d}')
    ax.legend()
    ax.set_title(symbols)
    return locals()

def example_2(symbols="aapl", df=None, thres=1e-3, maxlag=1, autolag=None, lim=None, regression='c'):
    if df is None:
        df = get_data(symbols=symbols)
    s = np.log(df["Adj Close"])
    l = plot_min_ffd(s, symbols=symbols, thres=thres, maxlag=maxlag, autolag=autolag, lim=lim, regression='c')
    ax = plt.gca()
    ax.set_title(symbols)
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

import scipy

def plot_min_ffd(s, symbols=None, thres=1e-2, maxlag=1, autolag=None, lim=None, regression='c'):
    """Finding the minimum d value that passes the ADF test.
    Input s should be prices."""

    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "99% conf", "corr"])
    x = np.log(s.values)
    for d in np.linspace(0, 0.6, 50):
        # df1 -> x, df2 -> dx
        dx = frac_diff_ffd(x, d, thres=thres, lim=lim)

        # this bit is important
        i = ~np.isnan(dx)
        # i = slice(None, None)

        if dx[i].shape[0] < 10:
            print(f'SKIPPING: data size after slicing is {x[i].shape[0]} from {x.shape[0]}')
            continue

        dx = np.nan_to_num(dx, 0)
        # # dx = fast_frac_diff(x, d)
        corr = np.corrcoef(x[i], dx[i])[0, 1]
        adf = adfuller(dx[i], maxlag=maxlag, regression=regression, autolag=autolag)
        out.loc[d] = list(adf[:4]) + [adf[4]["5%"], adf[4]["1%"]] + [corr]  # with critical value

    # TODO: need to get the best differences  and then plot them

    params = np.polyfit(
        out.index.values,
        out['adfStat'].values, 
        deg=3)

    roots = (np.poly1d(params) - out['99% conf'].iloc[0]).roots
    real_root = [x for x in roots if np.imag(x) == 0.0 and np.real(x) > 0]
    assert len(real_root) == 1
    real_root = np.real(real_root[0])

    plt.ion()
    fig = plt.figure(1)
    fig.clf()
    ax = fig.subplots(3, 1)
    # a = out[["adfStat", "corr"]].plot(secondary_y="corr", ax=ax[0], style=".-")
    ax[0].plot(out.index, out['adfStat'], label='abfStat')
    ax[0].axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted", label='95% conf')
    ax[0].axhline(out["99% conf"].mean(), linewidth=1, color="g", linestyle="dotted", label='99% conf')

    # yy = np.linspace(*ax[0].get_ylim(), 100)
    # ax[0].plot(np.polyval(params, yy), yy, color='r', label='fit', linestyle='--')
    xx = np.linspace(*ax[0].get_xlim(), 100)
    ax[0].plot(xx, np.polyval(params, xx), color='r', label='fit', linestyle='--')
    ax[0].plot([real_root] * 2, ax[0].get_ylim(), color='y', linewidth=2, alpha=0.5)
    
    ax2 = ax[0].twinx()
    ax2.plot(out.index, out['corr'], color='g', alpha=0.5, label='corr')
    ax[0].legend()
    ax2.legend()
    ax2.set_ylabel('corr')
    ax[0].set_ylabel('adfStat')
    ax[0].grid()
    ax[1].plot(s.index, s.values)
    ax[1].grid()
    ax[1].set_ylabel('price')

    dx = frac_diff_ffd(x, real_root, thres=thres, lim=lim)
    dx = pd.Series(dx, s.index)
    dx.plot(ax=ax[2])
    ax[2].set_title(f'best d={real_root}')
    ax[2].grid()

    plt.tight_layout()
    plt.show()
    print(out)
    if symbols is not None:
        plt.savefig(os.path.expanduser(f'~/{symbols}.png'))
    return locals()
