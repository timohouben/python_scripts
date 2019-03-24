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

def combine_results(path_to_multiple_projects, filename="results.csv"):
    """
    Searches for results.csv in all subdirectories and combines these files. All csv files must have the same header!

    Parameters
    ----------

    path_to_multiple_projects : string
        Path in which you want to start searching. Results will be stored on this level.
    filename : string
        Name of .csv file. Default: results.csv
    """

    import os
    import os.path

    if os.path.exists(path_to_multiple_projects + "/combined_results") == False:
        os.mkdir(path_to_multiple_projects + "/combined_results")

    file_paths = []
    for dirpath, dirnames, filenames in os.walk(path_to_multiple_projects):
        for filename in [f for f in filenames if f.endswith(str(filename))]:
            file_paths.append(dirpath + "/" + str(filename))

    # get header from first file in list
    with open(file_paths[0]) as f:
        header = f.readline()
        f.close()

    csv_merge = open(path_to_multiple_projects + "/combined_results" + "/" + "csv_merge.csv", 'w')
    csv_merge.write(header)

    for file in file_paths:
        csv_in = open(file)
        for line in csv_in:
            if line.startswith(header):
                continue
            csv_merge.write(line)
        csv_in.close()
    csv_merge.close()
    print('Created consolidated CSV file : ' + "csv_merge.csv")

if __name__ == "__main__":
    # test for combine_results()
    combine_results("/Users/houben/Desktop/TEST_combine")
