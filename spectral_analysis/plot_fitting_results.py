# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# set some parameters for the analysis manually
dpi = 300
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def plot_parameter_vs_location_block(results, path_to_results, borders, S_in, recharge1, recharge2, threshold = 0.1, comment="", saveimg=True):
    """
    Plot the derived parameter (T or S) along the different observation points
    and for a selection of positions of the boarder between a high and a low
    conductive zone.

    CHANGE MANUALLY: The RECHARGE, THE PARAMETER and THE VALUE OF THE PARAMETER

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results from spectral analysis.
    path_to_results : string
        Path where to store the images.
    borders : list of integers
        List of values for the boarder between two zones.
    S_in : float
        For which S input.
    recharge1 : string
        Name of the first recharge.
    recharge2 : string
        Name of the second recharge.
    threshold : float
        Threshold for variance from curve_fit. All other entries will be dropped from data frame.
    comment : string
        Give a commment.
    saveimg : bool, default: True
        True: Saves a .png image like it has been modified in the interactive backend. Will be saved after you have closed the window.

    Yields
    ------

    A/multiple plot(s) in path_to_results directory.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from processing import identify_numbers_from_string

    # convert strings to numbers
    results["cov_numbers"] = results["cov"].apply(identify_numbers_from_string)
    results["sigma_S"] = results["cov_numbers"].apply(lambda x: float(x[0]) if x != [] else np.nan)
    results["sigma_T"] = results["cov_numbers"].apply(lambda x: float(x[3]) if x != [] else np.nan)
    # remove all lines where the covariance is higher than threshold
    results = results[results["sigma_T"] > -threshold]
    results = results[results["sigma_T"] < threshold]

    fig, axs = plt.subplots(len(borders), 1, sharex=True, figsize=(15,10))
    for i, border in enumerate(borders):
        results_temp = results
        border_str = "border_" + str(border) + "_"
        # get a new column with border values
        results_temp = results_temp[results_temp.name.str.contains(border_str)]
        # only values with specific S_in
        results_temp = results_temp[results_temp["S_in"] == S_in]
        # plot the T in
        T_in = [float(results_temp.T_in_1.unique()) if obs < border else float(results_temp.T_in_2.unique()) for obs in np.arange(0,1010,10)]
        axs[i].semilogy(np.arange(0,border,10), T_in[:len(np.arange(0,border,10))], linestyle="--", linewidth=5, marker="", label="high conductive part", color="#1f77b4")
        axs[i].semilogy(np.arange(border,1010,10), T_in[len(np.arange(0,border,10)):], linestyle="--", linewidth=5, marker="", label="low conductive part", color="#ff7f0e")
        axs[i].grid(which="major", color="white", linestyle=":")
        # for both recharges in one plot
        results_temp_r1 = results_temp[results_temp["recharge"] == recharge1]
        results_temp_r2 = results_temp[results_temp["recharge"] == recharge2]
        axs[i].plot(results_temp_r1.obs_loc, results_temp_r1.T_out, label="derived Transmissivity, white noise", linewidth=3, marker="^", color="#2ca02c")
        axs[i].plot(results_temp_r2.obs_loc, results_temp_r2.T_out, label="derived Transmissivity, mHM", linewidth=1, marker="*", color="#d62728")
        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0.2)
        #axs[i].set_title("Border at " + str(border) + " m")
        axs[i].annotate("Boundary \n" + str(border), (border+10, 5e-4))
        axs[i].axvline(x=border, color='black')
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(True)
        #axs[i].set_ylim(ymin=np.min(T_in + results_temp.T_out.tolist())*0.5, ymax=np.max(T_in + results_temp.T_out.tolist())*4)
        axs[i].set_ylim(bottom=7e-5, top=5e-1)
        axs[i].set_facecolor('#C0C0C0')
        # set second y achsis
        #axs_twin = axs[i].twinx()
        #axs_twin.bar(results_temp.obs_loc, results_temp.sigma_T)
        #axs_twin.errorbar(results_temp.obs_loc, results_temp.T_out, results_temp.sigma_T, label="Border at " + str(border) + " m", marker="+")
    fig.text(0.04, 0.5, 'Transmissivity $[m^2/s]$', va='center', rotation='vertical')
    plt.xlabel("location [m]")
    fig.suptitle("Derived Transmissivity vs input Transmissivity\n" + comment)
    plt.legend(loc="lower left")
    axs[len(borders)-1].legend(loc='upper center', bbox_to_anchor=(0.5, -1),
          fancybox=True, shadow=True, ncol=5)
    if saveimg == True:
        plt.savefig(path_to_results + "/" + comment + "T_vs_location.png", dpi=dpi)
    #plt.show()

def plot_error_vs_tc(results, path_to_results, comment="", abs=True):
    """
    Plot errors of input and output parameters (S, T, tc) vs input tc.
    This results in 3 (S,T,tc) plots with multiple graphs according to the
    different observation points.

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results generated by multi_psd.py.
    path_to_results : string
        Path where to store the images
    abs : bool
        Take the mean over absolute errors or not.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    from cycler import cycler

    # ['#f1eef6', '#d0d1e6', '#a6bddb', '#74a9cf','#3690c0','#0570b0','#034e7b']
    plt.rc(
        "axes",
        prop_cycle=(
            cycler(
                "color",
                [
                    #    '#edf8b1',
                    "#c7e9b4",
                    #    '#7fcdbb',
                    "#41b6c4",
                    "#1d91c0",
                    "#225ea8",
                    "#253494",
                ],
            )
        ),
    )
    font = {"family": "normal", "weight": "normal", "size": 14}
    plt.rc("font", **font)
    if not os.path.exists(path_to_results + "/error_vs_tc"):
        os.mkdir(path_to_results + "/error_vs_tc")
    for error in ["err_S", "err_T", "err_tc"]:
        tc_agg = results["tc_in"].apply(lambda x: np.around(x, 2)).unique()
        tc_agg.sort()
        for loc in [200, 400, 600, 800, 990]:  # results.obs_loc.unique():
            results_loc = results[error][results["obs_loc"] == loc]
            err_vs_tc_at_loc = []
            for tc in tc_agg:
                print(
                    "...currently grouping values for tc = " + str(tc),
                    "location: " + str(loc),
                    "Error: " + error,
                )
                # append error to list for specific tc
                if abs == True:
                    err_vs_tc_at_loc.append(
                        np.mean(
                            np.abs(
                                results_loc[
                                    results["tc_in"].apply(lambda x: np.around(x, 2))
                                    == tc
                                ]
                            )
                        )
                    )
                if abs == False:
                    err_vs_tc_at_loc.append(
                        np.mean(
                            results_loc[
                                results["tc_in"].apply(lambda x: np.around(x, 2)) == tc
                            ]
                        )
                    )
            plt.loglog(tc_agg, err_vs_tc_at_loc, label=str(loc) + " m")

        plt.ylabel(error + " [%]")
        plt.xlabel("t_c [days]")
        plt.legend()
        plt.savefig(
            path_to_results
            + "/error_vs_tc/"
            + error
            + "_vs_tc_abs_"
            + str(abs)
            + "_"
            + comment
            + ".png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()


def plot_errors_vs_loc_aggregate(
    results, path_to_results, error, aggregate, bins, abs, comment=""
):
    """
    Plot errors of input and output parameters (S, T, tc) vs the observation
    location in the aquifer (2D transect) and aggregated for specific ranges of tc.
    This results in three plots over all model runs.

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results generated by multi_psd.py.
    path_to_results : string
        Path where to store the images
    error : string
        Select the column from data frame to plot along locations.
    aggregate : string
        Must be an existing column in dataframe. Lines in plot will be aggregated based on the chosen column.
    bins : list of floats
        Lines in plot will be aggregated according to the values in the list and the chosen column.
    abs : bool
        True: Neglect the sign of errors an avegage only positive deviations.
    comment : string
        Give a meaningful comment for the file name. Otherwise it will override existing files.


    Yields
    ------

    see docstring description
    """
    # print(results)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    print("Your have provided the following bins: ", bins)
    # get list of locations from df
    obs_locs = results["obs_loc"].unique()
    obs_locs = [int(i) for i in obs_locs]
    # make arrays with errors for obs_locs
    err = np.zeros((len(bins), len(obs_locs)))
    for i, loc in enumerate(obs_locs):
        for j, bin in enumerate(bins):
            if abs == True:
                temp = results[results["obs_loc"] == loc][results[aggregate] <= bin][
                    error
                ]
                temp.tolist()
                temp = [np.abs(i) for i in temp]
                err[j, i] = np.mean(temp)
            else:
                err[j, i] = np.mean(
                    results[results["obs_loc"] == loc][results[aggregate] <= bin][error]
                )

    for i in np.arange(len(bins)):
        plt.plot(obs_locs, err[i, :], label="< " + "{0:0.3e}".format(bins[i]))
    plt.legend()
    plt.title(("Plot: " + error + ", " + "Aggregation: " + aggregate))
    plt.ylabel(error)
    plt.xlabel("location [m]")
    plt.savefig(path_to_results + "/" + error + "-" + aggregate, bbox_inches="tight")
    plt.close()


def plot_errors_vs_loc_hetero(obs, error_list, legend, ylabel, path, comment):
    """
    Plot the error vs location in the aquifer.

    Parameters
    ----------

    obs : list
        List of x value of observations points
    error_list : list of lists for different errors
        List of errors for each observation point.
    legend : list of strings for different errors
        String for ylabel
    ylabel : string
        ylabel
    path : string
        Path where to srote the images.
    comment : string
        Give a comment to be stored in the filename.

    Yields
    ------

    Saves a plot in the path directory.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os.path

    for i, error in enumerate(error_list):
        plt.plot(obs, error, label=legend[i])
    plt.hlines(0, 0, np.max(obs), colors='k', linestyles="dashed")
    plt.legend()
    plt.ylabel(ylabel)
    plt.title("Error vs location: " + os.path.basename(path))
    plt.savefig(path + "/" + comment + "Error_vs_loc", dpi=dpi, bbox_inches="tight")


def plot_errors_vs_loc_homo(results, path_to_results, comment=""):
    """
    Plot errors of input and output parameters (S, T, tc) vs the observation
    location in the aquifer (2D transect). This results in three plots per OGS
    model run.

    Parameters
    ----------

    results : pandas dataframe
        Dataframe with results generated by multi_psd.py.
    comment : string
        Give a meaningful comment for the file name. Otherwise it will override existing files.

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
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()

    for project_folder in results.name.unique():
        err_S = results[results["name"] == project_folder]["err_S"]
        err_T = results[results["name"] == project_folder]["err_T"]
        err_tc = results[results["name"] == project_folder]["err_tc"]
        obs_loc = results[results["name"] == project_folder]["obs_loc"]
        plot(
            obs_loc,
            err_S,
            path_to_results + "/" + comment + project_folder + "_err_S.png",
            label=project_folder,
            title="Relative error in storativity",
        )
        plot(
            obs_loc,
            err_T,
            path_to_results + "/" + comment + project_folder + "_err_T.png",
            label=project_folder,
            title="Relative error in transmissivity",
        )
        plot(
            obs_loc,
            err_tc,
            path_to_results + "/" + comment + project_folder + "_err_tc.png",
            label=project_folder,
            title="Relative error in characteristic time",
        )


def plot_heatmap(results, path_to_results, abs=True, comment=""):
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
    comment : string
        Give a meaningful comment for the file name. Otherwise it will override existing files.

    Yields
    ------

    see docstring description
    """

    import seaborn as sns
    import os

    # extract input values for achsis limits
    # achsisticks_x = [1,5,10,15,20,25,30,35,40,45]
    # achsislabel_x = ["%1.2e" % i for i in achsisticks_x]
    # achsisticks_y = [1,5,10,15,20,25,30,35,40,45]
    # achsislabel_y = ["%1.2e" % i for i in achsisticks_y]

    def plot(pivotted, error):
        import numpy as np
        from matplotlib.colors import LogNorm
        import math

        # set axismin and axis max based on input space (hard coded, BAD SOLUTION)
        # achsismin_y, achsismax_y = 1e-6, 1e-1
        # achsismin_x, achsismax_x = 1e-5, 1
        # achsisticks_x = [math.pow(10, i) for i in range(math.floor(math.log10(achsismin_x)), 1+math.ceil(math.log10(achsismax_x)))]
        # achsisticks_y = [math.pow(10, i) for i in range(math.floor(math.log10(achsismin_y)), 1+math.ceil(math.log10(achsismax_y)))]
        barmin, barmax = 1, 1000
        cbar_ticks = [1, 10, 100, 1000]
        log_norm = LogNorm(vmin=barmin, vmax=barmax)
        ax = sns.heatmap(
            pivotted,
            cmap="PuBu",
            cbar_kws={"ticks": cbar_ticks},
            vmax=barmax,
            vmin=barmin,
            norm=log_norm,
        )  # , yticklabels=achsislabel_y, xticklabels=achsislabel_x)
        # cmap="Spectral_r",
        # ax.set_yticks(achsisticks_y)
        # ax.set_xticks(achsisticks_x)
        ax.invert_yaxis()
        # import matplotlib.ticker as ticker
        # tick_locator = ticker.MaxNLocator(12)
        # ax.xaxis.set_major_locator(tick_locator)
        # ax.yaxis.set_major_locator(tick_locator)

        fig = ax.get_figure()
        # fig.set_size_inches(5, 5)
        if not os.path.exists(path_to_results + "/" + comment + "heatmap"):
            os.mkdir(path_to_results + "/" + comment + "heatmap")

        fig.savefig(
            path_to_results
            + "/"
            + comment
            + "heatmap"
            + "/"
            + str(obs_loc)
            + "_"
            + error,
            dpi=dpi,
            bbox_inches="tight",
        )
        fig.clf(fig)

    for obs_loc in results["obs_loc"].unique():
        # extract only rows with obs_loc==obs_loc
        df_obs_loc = results[results.obs_loc == obs_loc]
        # extract columns for plotting
        for error in ["err_S", "err_T", "err_tc"]:
            # absolute erros, NOT A GOOD SOLUTION
            if abs == True:
                results[error] = results[error].apply(
                    lambda x: x * (-1) if x < 0 else x
                )
            df_obs_loc_cut = df_obs_loc[["S_in", "T_in", error]]
            # pivot this table
            pivot_df_obs_loc_cut = df_obs_loc_cut.pivot("S_in", "T_in", error)
            # plot heatmap
            plot(pivot_df_obs_loc_cut, error)

def plot_parameter_vs_location(path_to_results, parameter, location, y_label, error=None):
    """
    Plots a list/array of parameters along location and saves it in path_to_results

    Parameters
    ----------

    path_to_results : string
        Path to results.
    parameter : 1D List, array
        Parameter to plot
    location : 1D list, array
        Locations
    y_label : string
        Give the y-achsis a name.
    error : 1D-array
        If None, no errors will be plotted.
        If 1D array, error bars will be used for plotting.

    Yields
    ------

    A plot per OGS folder in path_to_results.
    """

    import matplotlib.pyplot as plt
    import os.path

    if error == None:
        plt.plot(location, parameter, label=y_label)
    elif error != None:
        plt.errorbar(location, parameter, error, label=y_label)
    plt.xlabel("location [m]")
    plt.ylabel(y_label)
    plt.title(os.path.basename(path_to_results))
    plt.savefig(path_to_results + "/" + y_label + "_vs_location.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # plot_error_vs_tc(results, path_to_results, abs=True)
    # plot_errors_vs_loc(results, path_to_results)
    # plot_heatmap(results, path_to_results)

    # call plot_errors_vs_loc_aggregate for different parameter combinations
    # bins for category "tc_in"
    # error = ["err_S", "err_T", "err_tc"]
    # aggregate = ["S_in", "T_in", "tc_in"]
    # bins = [
    # np.power(10,np.linspace(-5,-1,10)).tolist(),
    # np.power(10,np.linspace(-6,-2,10)).tolist(),
    # [1,10,100,1000,3000,9000,15000,20000,25000,30000,100000],
    # ]
    # for err in error:
    #    for i,agg in enumerate(aggregate):
    #        #print(err,agg,bins[i])
    #        plot_errors_vs_loc_aggregate(results, path_to_results, err, agg, bins[i], abs=True)

    ## execute plot_parameter_vs_location_block
    ## ----------------------------------------

    #borders = [800,810,820,830,840,850,860,870,880,890,900]
    #borders = [700,710,720,730,740,750,760,770,780,790,800]
    #borders = [200,210,250,280,290,300,310,320,330,340,500]
    #borders = [110,120,130,140,150,160,170,180,190,200]
    borders = [50,100,200,300,400,500,600,800,950,970,990]
    recharge1 = "recharge_daily.txt"
    recharge2 = "recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"

    # configuration for 20190531_SA_hetero_block
#    results = pd.read_csv("/Users/houben/phd/results/20190531_SA_hetero_block/results_merge.csv")
#    path_to_results = "/Users/houben/phd/results/20190531_SA_hetero_block"
#    plot_parameter_vs_location_block(results, path_to_results, borders=borders, S_in = 0.003, recharge1=recharge1, recharge2=recharge2, comment="S = 0.003")

    # configuration for 20190717_SA_hetero_block_2
#    results = pd.read_csv("/Users/houben/phd/results/20190717_SA_hetero_block_2/1_results_merge.csv")
#    path_to_results = "/Users/houben/phd/results/20190717_SA_hetero_block_2"
#    plot_parameter_vs_location_block(results, path_to_results, borders=borders, S_in = 0.3, recharge1=recharge1, recharge2=recharge2, comment="S = 0.3")
