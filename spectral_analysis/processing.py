# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division
# ------------------------------------------------------------------------------

def detrend(timeseries, type="linear"):
    """
    Function to detrend the time series.

    Parameters
    ----------

    timeseries : 1D array
        time series to be detrendet
    type : string
        linear or ...?

    Yields
    ------

    timeseries : 1D array
        Detrendet time series.
    """
    from scipy.signal import detrend

    return detrend(timeseries, type=type)


def cut(x, y, threshold, keep="before"):
    """
    Cut a series x at given threshold and cut corresponding series y
    at same index. Threshold will be kept in series when 'before' is chosen.
    If 'after' was chosen, threshold will be not in list any more.

    Parameters
    ----------
    x : 1D array
        Series to which the threshold refers. Should be sorted either
        ascending or descending.
    y : 1D array
        Series which will be cut at same index as x.
    threshold : float
        Threshold where x series will be cut (excluding)
    keep : string
        'before' : values with index smaller than the threshold will be kept
        'after' : values with index after threshold will be kept

    Yields
    ------
    x_cut : 1D array, list
        Cut series.
    y_cut : 1D array, list
        Cut series.
    """

    import numpy as np

    if np.shape(x) != np.shape(y):
        raise ValueError
        print("x and y must have same length.")
    if np.asarray(x).ndim != 1:
        raise ValueError
        print("x and y must have dimension = 1.")

    if [i for i in sorted(x)] == [i for i in x]:
        if threshold < x[0]:
            raise ValueError
            print("Your threshold is to low. Not cutting list.")
        if threshold > x[-1]:
            raise ValueError
            print("Your threshold is to high. Not cutting list.")
        for i, item in enumerate(x):
            if item > threshold:
                if keep == "before":
                    return x[:i], y[:i]
                elif keep == "after":
                    return x[i:], y[i:]
    elif [i for i in sorted(x, reverse=True)] == [i for i in x]:
        if threshold > x[0]:
            raise ValueError
            print("Your threshold is to high. Not cutting list.")
        if threshold < x[-1]:
            raise ValueError
            print("Your threshold is to low. Not cutting list.")
        for i, item in enumerate(x):
            if item < threshold:
                if keep == "before":
                    return x[:i], y[:i]
                elif keep == "after":
                    return x[i:], y[i:]
    else:
        raise ValueError(
            "Your series x is not sorted. Sort it either ascending or descending."
        )


def percent_fraction(a, b):
    """
    Returns fraction of a in b as %.
    a / b * 100

    Parameters
    ----------
    a : float

    b : float

    """
    return a / b * 100


def percent_difference_fraction(a, b):
    """
    Returns fraction of difference of a and b to a as %.
    (a - b)/b * 100

    Parameters
    ----------
    a : float

    b : float

    """
    return (a - b) / a * 100
