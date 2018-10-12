#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Script to make time series analysis with multiple folders comparing ogs and gw_model

Created on Wed Oct  3 15:26:55 2018

@author: houben
"""
import numpy as np
path_to_multiple_projects = '/Users/houben/PhD/modelling/ogs_vs_derooij12/con_transient_compare/run_3'
obs_points = ['obs_0100', 'obs_0500', 'obs_0950', 'obs_0990']

def pearson_corr(path_to_multiple_projects=path_to_multiple_projects, obs_points=obs_points):
    
    import numpy as np
    import os
    from scipy.stats import pearsonr
    
    # sort key
    def sort_key_1(elem):
        return elem[7]
    def sort_key_2(elem):
        return elem[0]
    #list_dir = [f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith('.')]
    list_dir = [f for f in os.listdir(str(path_to_multiple_projects)) if f.endswith('28')]
    list_dir.sort(key=sort_key_2, reverse=True)
    list_dir.sort(key=sort_key_1)
    list_dir.reverse()

    pearson_corr = np.zeros((len(obs_points), len(list_dir)))
    nse = np.zeros((len(obs_points), len(list_dir)))
    rmse = np.zeros((len(obs_points), len(list_dir)))
    for i,curr_dir in enumerate(list_dir):
        path_to_project = str(path_to_multiple_projects) + '/' + str(curr_dir)
        #print('Creating timeseries for: ' + str(path_to_project) + '. ' + str(i+1) + ' of ' + str(len(list_dir)) + ' in progress...')
    
        for j,obs_point in enumerate(obs_points):
            head_gw_model = np.loadtxt(str(path_to_project) + '/' + 'head_gw_model_' + str(obs_point[5:-1] + '.txt'))
            head_ogs = np.loadtxt(str(path_to_project) + '/' + 'head_ogs_' + str(obs_point) + '_max.txt')
    

            # calculation of RMSE
            rmse[j,i] = np.sqrt((sum((head_ogs[:]-head_gw_model[:])**2) / len(head_gw_model)))
            #rmse[:] = rmse[:] * 1000


            # calculation of the NSE
            nse[j,i] = 1 - (sum((head_ogs[:]-head_gw_model[:])**2) / sum((head_ogs[:]-np.mean(head_ogs))**2))
#            nse[j,i] = (sum((head_gw_model[:]-head_ogs[:])**2) / sum((head_gw_model[:]-np.mean(head_gw_model))**2))
            pearson_corr[j,i] = pearsonr(head_gw_model, head_ogs)[0]
            #pearson_corr = np.round(pearson_corr,6)
    #return pearson_corr[:,-15:], obs_points, list_dir[-15:]
    return nse, obs_points, list_dir



####
# plot the heatmap
####
import matplotlib.pyplot as plt
import matplotlib

harvest, vegetables, farmers = pearson_corr()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(data_for_colors, im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data_for_colors = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data_for_colors.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.N
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data_for_colors.shape[0]):
        for j in range(data_for_colors.shape[1]):
            kw.update(color=textcolors[im.norm(data_for_colors[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data_for_colors[i, j], None), **kw)
            texts.append(text)

    return texts


fig, ax = plt.subplots()

#im, cbar = heatmap(harvest[:,:13], vegetables, farmers[:13], ax=ax,
im, cbar = heatmap(harvest[:,-13:], vegetables, farmers[-13:], ax=ax,
#im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="RdYlGn", cbarlabel="NSE")
#YlGn
#texts = annotate_heatmap(im, valfmt="{x:.1f} t")
texts = annotate_heatmap(harvest, im, valfmt="{x:.2f}")

fig.tight_layout()
plt.show()