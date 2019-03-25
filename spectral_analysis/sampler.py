#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def calc_t_c(L, S, T):
    return L**2 * S / 3 / T / 86400


# define parameter space
S = np.power(10,np.linspace(-5,-1,41))
T = np.power(10,np.linspace(-6,-2,41))
aquifer_length = 1000
vmin = min(S)
vmax = min(T)
barmax = 1e6
barmin = 1e-2

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

        #achsisticks_y = [1e-5,1e-4,1e-3,1e-2,1e-1]
        #achsislabel_y = [["%1.0e" % i for i in S][j-1] for j in achsisticks_y]
        #achsisticks_x = [1,5,10,15,20]#,25,30,35,40,45]
        #achsislabel_x = [["%1.0e" % i for i in S][j-1] for j in achsisticks_x]

        # normalize the ticks
        log_norm = LogNorm(vmin=barmin, vmax=barmax)
        cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(barmin)), 1+math.ceil(math.log10(barmax)))]
        ax = sns.heatmap(pivot, cmap="Spectral_r",norm=log_norm,cbar_kws={"ticks": cbar_ticks})#,yticklabels=achsislabel_y)#,xticklabels=achsislabel_x)
        #ax.set_yticks(achsisticks_y)
        #ax.set_xticks(achsisticks_x)
        ax.invert_yaxis()
        import matplotlib.ticker as ticker
        #y_fmt = ticker.FormatStrFormatter('%2.2e')
        #ax.yaxis.set_major_formatter(y_fmt)
        tick_locator = ticker.MaxNLocator(12)
        #ax.xaxis.set_major_locator(tick_locator)
        #ax.yaxis.set_major_locator(tick_locator)

        #ax.invert_yaxis()
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

    import os
    dir_path = os.path.dirname(os.path.realpath("__file__"))
    np.save(dir_path + "/samples",data)

    # create data frame from dictionary
    dataframe = pd.DataFrame(data,columns=['S','T','tc'])
    # make pivot table from data frame to plot as heatmap
    pivot = dataframe.pivot("S","T","tc")
    # plot as heatmap
    plot(pivot,vmin,vmax)
    plt.show


if __name__ == "__main__":
    plot_heatmap(S, T, calc_t_c)
