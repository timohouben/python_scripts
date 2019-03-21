# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# set some parameters for the analysis manually
dpi = 200
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def plot_errors_vs_loc(results, path_to_results):
    """
    Plot errors of input and output parameters (S, T, tc) vs the observation
    location in the aquifer (2D transect). This results in three plots per OGS
    model run.

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results generated by multi_psd.py.

    Yields
    ------

    see docstring description
    """

    import matplotlib.pyplot as plt

    def plot(x, y, path, label, title):
        plt.plot(x, y, label=label)
        plt.xlabel("Location of observation point [m]")
        plt.ylabel("Error in %")
        plt.title(title)
        plt.legend()
        plt.savefig(path, dpi=dpi)
        plt.close()

    for project_folder in results.name.unique():
        err_S = results[results["name"] == project_folder]["err_S"]
        err_T = results[results["name"] == project_folder]["err_T"]
        err_tc = results[results["name"] == project_folder]["err_tc"]
        obs_loc = results[results["name"] == project_folder]["obs_loc"]
        plot(
            obs_loc,
            err_S,
            path_to_results + "/" + project_folder + "_err_S.png",
            label=project_folder,
            title="Relative error in storativity",
        )
        plot(
            obs_loc,
            err_T,
            path_to_results + "/" + project_folder + "_err_T.png",
            label=project_folder,
            title="Relative error in transmissivity",
        )
        plot(
            obs_loc,
            err_tc,
            path_to_results + "/" + project_folder + "_err_tc.png",
            label=project_folder,
            title="Relative error in characteristic time",
        )


def plot_heatmap(results, path_to_results, abs = True):
    """
    Plot errors of input and output parameters (S, T, tc) vs the parameter
    range as heatmap. This results in three plots per location.

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results generated by multi_psd.py.
    path_to_results : string
        Where to store the heatmaps.
    abs : bool
        Absolute error in %.

    Yields
    ------

    see docstring description
    """

    import seaborn as sns
    import os

    # extract input values for achsis limits
    achsisticks_x = results["T_in"].unique()
    print(achsisticks_x)
    achsislabel_x = ["%1.2e" % i for i in achsisticks_x]
    achsisticks_y = results["S_in"].unique()
    achsislabel_y = ["%1.2e" % i for i in achsisticks_y]
    print(achsislabel_y)

    def plot(pivotted, error):
        import numpy as np
        from matplotlib.colors import LogNorm
        import math
        # set axismin and axis max based on input space (hard coded, BAD SOLUTION)
        #achsismin_y, achsismax_y = 1e-6, 1e-1
        #achsismin_x, achsismax_x = 1e-5, 1
        #achsisticks_x = [math.pow(10, i) for i in range(math.floor(math.log10(achsismin_x)), 1+math.ceil(math.log10(achsismax_x)))]
        #achsisticks_y = [math.pow(10, i) for i in range(math.floor(math.log10(achsismin_y)), 1+math.ceil(math.log10(achsismax_y)))]
        barmin, barmax = 1, 1000
        cbar_ticks = [1,10,100,1000]
        log_norm = LogNorm(vmin=barmin, vmax=barmax)
        plot = sns.heatmap(pivotted, cmap="Spectral_r",cbar_kws={"ticks": cbar_ticks}, norm=log_norm, vmax=barmax, vmin=barmin, yticklabels=achsislabel_y, xticklabels=achsislabel_x, fmt="1.3e")
        plot.set_yticks(achsisticks_y)
        plot.set_xticks(achsisticks_x)
        fig = plot.get_figure()
        if not os.path.exists(path_to_results + "/heatmap"):
            os.mkdir(path_to_results + "/heatmap")

        fig.savefig(
            path_to_results + "/heatmap" + "/" + str(obs_loc) + "_" + error, dpi=dpi
        )
        fig.clf()

    for obs_loc in results["obs_loc"]:
        # extract only rows with obs_loc==obs_loc
        df_obs_loc = results[results.obs_loc == obs_loc]
        # extract columns for plotting
        for error in ["err_S", "err_T", "err_tc"]:
            # absolute erros, NOT A GOOD SOLUTION
            if abs == True:
                results[error] = results[error].apply(lambda x: x*(-1) if x < 0 else x)
            df_obs_loc_cut = df_obs_loc[["S_in", "T_in", error]]
            # pivot this table
            pivot_df_obs_loc_cut = df_obs_loc_cut.pivot("S_in", "T_in", error)
            # plot heatmap
            plot(pivot_df_obs_loc_cut, error)


if __name__ == "__main__":
    import pandas as pd

    results = pd.read_csv(
        "/Users/houben/PhD/modelling/20190304_spectral_analysis_homogeneous/models/fitting_results/results.csv"
    )
    path_to_results = "/Users/houben/PhD/modelling/20190304_spectral_analysis_homogeneous/models/fitting_results"
    plot_errors_vs_loc(results, path_to_results)
    plot_heatmap(results, path_to_results)
