#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def calc_t_c(L, S, T):
    return L**2 * S / 3 / T / 86400

aquifer_length = 1000
vmin = 0.001
vmax = 1000000

# define parameter space
S = [
    5.0e-1,
    4.5e-1,
    4.0e-1,
    3.5e-1,
    3.0e-1,
    2.5e-1,
    2.0e-1,
    1.5e-1,
    1.0e-1,
    9.0e-2,
    8.0e-2,
    7.0e-2,
    6.0e-2,
    5.0e-2,
    4.0e-2,
    3.0e-2,
    2.0e-2,
    1.0e-2,
    9.0e-3,
    8.0e-3,
    7.0e-3,
    6.0e-3,
    5.0e-3,
    4.0e-3,
    3.0e-3,
    2.0e-3,
    1.0e-3,
    9.0e-4,
    8.0e-4,
    7.0e-4,
    6.0e-4,
    5.0e-4,
    4.0e-4,
    3.0e-4,
    2.0e-4,
    1.0e-4,
    9.0e-5,
    8.0e-5,
    7.0e-5,
    6.0e-5,
    5.0e-5,
    4.0e-5,
    3.0e-5,
    2.0e-5,
    1.0e-5
    ]

T = [
    9.0e-2,
    8.0e-2,
    7.0e-2,
    6.0e-2,
    5.0e-2,
    4.0e-2,
    3.0e-2,
    2.0e-2,
    1.0e-2,
    9.0e-3,
    8.0e-3,
    7.0e-3,
    6.0e-3,
    5.0e-3,
    4.0e-3,
    3.0e-3,
    2.0e-3,
    1.0e-3,
    9.0e-4,
    8.0e-4,
    7.0e-4,
    6.0e-4,
    5.0e-4,
    4.0e-4,
    3.0e-4,
    2.0e-4,
    1.0e-4,
    9.0e-5,
    8.0e-5,
    7.0e-5,
    6.0e-5,
    5.0e-5,
    4.0e-5,
    3.0e-5,
    2.0e-5,
    1.0e-5,
    9.0e-6,
    8.0e-6,
    7.0e-6,
    6.0e-6,
    5.0e-6,
    4.0e-6,
    3.0e-6,
    2.0e-6,
    1.0e-6,
    ]


#plt.semilogy(S)
#plt.semilogy(T)



def plot_heatmap(S, T, function):
    """
    Plot S vs T and colorize the heatmap accodring to function.

    Parameters
    ----------

    S : 1D array
    T : 1D array
    function : functon to calculate the Z value

    Yields
    ------

    """

    import seaborn as sns
    import os
    import pandas as pd
    from matplotlib.colors import LogNorm
    import math

    def plot(pivot, vmin, vmax):
        # normalize the ticks
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(vmin)), 1+math.ceil(math.log10(vmax)))]
        ax = sns.heatmap(pivot, cmap="Spectral_r",norm=log_norm,cbar_kws={"ticks": cbar_ticks})
        ax.invert_yaxis()
        fig = ax.get_figure()
        plt.show()

    # make list of lists with all combinations of input parameters
    sample_space =[]
    for i in S:
        for j in T:
            sample_space.append([i,j,calc_t_c(aquifer_length,i,j)])
    # create a dictionary from input parameters and sample the space
    data = {
        'S': [sample_space[i][0] for i in np.arange(len(sample_space))],
        'T': [sample_space[i][1] for i in np.arange(len(sample_space))],
        'tc' : [sample_space[i][2] for i in np.arange(len(sample_space))]
        }

    #np.save("/Users/houben/PhD/modelling/20190318_spectral_analysis_homogeneous/ogs_input_data",data)

    # create data frame from dictionary
    dataframe = pd.DataFrame(data,columns=['S','T','tc'])
    # make pivot table from data frame to plot as heatmap
    pivot = dataframe.pivot("S","T","tc")
    # plot as heatmap
    plot(pivot,vmin,vmax)
    plt.show


if __name__ == "__main__":
    plot_heatmap(S, T, calc_t_c)
