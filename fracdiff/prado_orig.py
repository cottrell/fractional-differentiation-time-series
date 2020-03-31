import numpy as np


def fracDiff_original_impl(series, d, thres=0.01):
    from .fracdiff import get_weights
    import pandas as pd

    if isinstance(series, pd.Series):
        series = pd.DataFrame(series)
    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = 0
    if thres is not None:
        skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    # df = {}
    output = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method="ffill").dropna()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            output[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        # df[name] = df_.copy(deep=True)
    # df = pd.concat(df, axis=1)
    output = pd.Series(output).sort_index()
    return output


def fracDiff_FFD_original_impl(series, d, thres=1e-5):
    import pandas as pd
    from .fracdiff import get_weight_ffd

    if isinstance(series, pd.Series):
        series = pd.DataFrame(series)
    # 1) Compute weights for the longest series
    w = get_weight_ffd(d, thres, len(series))
    width = len(w) - 1
    # df = {}
    output = []
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        output.extend([0] * width)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            # df_[loc1] =
            output.append(np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0])
        # df[name] = df_.copy(deep=True)
    # df = pd.concat(df, axis=1)
    output = np.array(output)
    return output


def fracDiff_FFD_prado_original(series, d, thres=1e-5):
    """Constant width window (new solution)
    Note 1: thresh determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].

    Prado's implementation is totally broken for fractional d.
    """
    import pandas as pd
    from .fracdiff import get_weight_ffd

    if isinstance(series, pd.Series):
        series = pd.DataFrame(series)
    # 1) Compute weights for the longest series
    w = get_weight_ffd(d, thres, len(series))
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}
    # FRACTIONALLY DIFFERENTIATED FEATURES
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df
