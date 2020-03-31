import numpy as np
from numpy.fft import fft, ifft

# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py


def fast_frac_diff(x, d):
    """expanding window version using fft form"""
    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = ifft(fft(z1) * fft(z2))
    return np.real(dx[0:T])


def get_weights(d, size):
    """Expanding window fraction difference weights."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def get_weight_ffd(d, thres, lim):
    """Fixed width window fraction difference weights."""
    w, k = [1.0], 1
    for i in range(1, lim):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(x, d, thres=1e-5):
    """
    d is any positive real
    """
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width: i + 1])[0])
    return np.array(output)


# TESTS


def test_all():
    for d in [1, 1.5, 2, 2.5]:
        test_fast_frac_diff_equals_fracDiff_original_impl(d=d)
        test_frac_diff_ffd_equals_original_impl(d=d)
        # test_frac_diff_ffd_equals_prado_original(d=d) # his implementation is busted for fractional d


def test_frac_diff_ffd_equals_prado_original(d=3):
    # ignore this one for now as Prado's version does not work
    from .prado_orig import fracDiff_FFD_prado_original
    import pandas as pd

    x = np.random.randn(100)
    a = frac_diff_ffd(x, d, thres=1e-5)
    b = fracDiff_FFD_prado_original(pd.DataFrame(x), d, thres=1e-5)
    b = np.squeeze(b.values)
    a = a[d:]  # something wrong with the frac_diff_ffd gives extra entries of zero
    assert np.allclose(a, b)
    # return locals()


def test_frac_diff_ffd_equals_original_impl(d=3):
    from .prado_orig import fracDiff_FFD_original_impl
    import pandas as pd

    x = np.random.randn(100)
    a = frac_diff_ffd(x, d, thres=1e-5)
    b = fracDiff_FFD_original_impl(pd.DataFrame(x), d, thres=1e-5)
    assert np.allclose(a, b)
    # return locals()


def test_fast_frac_diff_equals_fracDiff_original_impl(d=3):
    from .prado_orig import fracDiff_original_impl
    import pandas as pd

    x = np.random.randn(100)
    a = fast_frac_diff(x, d)
    b = fracDiff_original_impl(pd.DataFrame(x), d, thres=None)
    b = b.values
    assert a.shape == b.shape
    assert np.allclose(a, b)
    # return locals()


if __name__ == "__main__":
    test_all()
