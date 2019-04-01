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

#['#f1eef6', '#d0d1e6', '#a6bddb', '#74a9cf','#3690c0','#0570b0','#034e7b']
    plt.rc('axes', prop_cycle=(cycler('color', ['#edf8b1',
    '#c7e9b4',
    '#7fcdbb',
    '#41b6c4',
    '#1d91c0',
    '#225ea8',
    '#253494'])))
    if not os.path.exists(path_to_results + "/error_vs_tc"):
        os.mkdir(path_to_results + "/error_vs_tc")

    for error in ["err_S","err_T","err_tc"]:
        tc_agg = results["tc_in"].apply(lambda x: np.around(x,2)).unique()
        tc_agg.sort()
        for loc in [100,200,400,500,600,900,990]:#results.obs_loc.unique():
            results_loc = results[error][results["obs_loc"] == loc]
            err_vs_tc_at_loc = []
            for tc in tc_agg:
                print("...currently grouping values for tc = " + str(tc), "location: " + str(loc), "Error: " + error)
                # append error to list for specific tc
                if abs == True:
                    err_vs_tc_at_loc.append(np.mean(np.abs(results_loc[results["tc_in"].apply(lambda x: np.around(x,2)) == tc])))
                if abs == False:
                    err_vs_tc_at_loc.append(np.mean(results_loc[results["tc_in"].apply(lambda x: np.around(x,2)) == tc]))
            plt.loglog(tc_agg,err_vs_tc_at_loc, label=str(loc) + " m")
        plt.ylabel(error + " [%]")
        plt.xlabel("t_c [days]")
        plt.legend()
        plt.savefig(path_to_results + "/error_vs_tc/" + error + "_vs_tc_abs_" + str(abs) + "_" + comment + ".png", dpi=dpi)
        plt.close()


def plot_errors_vs_loc_aggregate(results, path_to_results, error, aggregate, bins, abs, comment=""):
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
    #print(results)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    print("Your have provided the following bins: ", bins)
    # get list of locations from df
    obs_locs = results["obs_loc"].unique()
    obs_locs = [int(i) for i in obs_locs]
    # make arrays with errors for obs_locs
    err = np.zeros((len(bins),len(obs_locs)))
    for i, loc in enumerate(obs_locs):
        for j, bin in enumerate(bins):
            if abs == True:
                temp = results[results["obs_loc"] == loc][results[aggregate] <= bin][error]
                temp.tolist()
                temp = [np.abs(i) for i in temp]
                err[j,i] = np.mean(temp)
            else:
                err[j,i] = np.mean(results[results["obs_loc"] == loc][results[aggregate] <= bin][error])

    for i in np.arange(len(bins)):
        plt.plot(obs_locs,err[i,:],label="< " + "{0:0.3e}".format(bins[i]))
    plt.legend()
    plt.title(("Plot: " + error + ", " + "Aggregation: " + aggregate))
    plt.ylabel(error)
    plt.xlabel("location [m]")
    plt.savefig(path_to_results + "/" + error + "-" + aggregate)
    plt.close()



def plot_errors_vs_loc(results, path_to_results, comment=""):
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
    #achsisticks_x = [1,5,10,15,20,25,30,35,40,45]
    #achsislabel_x = ["%1.2e" % i for i in achsisticks_x]
    #achsisticks_y = [1,5,10,15,20,25,30,35,40,45]
    #achsislabel_y = ["%1.2e" % i for i in achsisticks_y]

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
        ax = sns.heatmap(pivotted, cmap="PuBu",cbar_kws={"ticks": cbar_ticks}, vmax=barmax, vmin=barmin, norm=log_norm)#, yticklabels=achsislabel_y, xticklabels=achsislabel_x)
        #cmap="Spectral_r",
        #ax.set_yticks(achsisticks_y)
        #ax.set_xticks(achsisticks_x)
        ax.invert_yaxis()
        #import matplotlib.ticker as ticker
        #tick_locator = ticker.MaxNLocator(12)
        #ax.xaxis.set_major_locator(tick_locator)
        #ax.yaxis.set_major_locator(tick_locator)

        fig = ax.get_figure()
        #fig.set_size_inches(5, 5)
        if not os.path.exists(path_to_results + "/" + comment + "heatmap"):
            os.mkdir(path_to_results + "/heatmap")

        fig.savefig(
            path_to_results + "/" + comment + "heatmap" + "/" + str(obs_loc) + "_" + error, dpi=dpi
        )
        fig.clf(fig)

    for obs_loc in results["obs_loc"].unique():
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
    import numpy as np

    results = pd.read_csv(
        "/Users/houben/Desktop/test/csv_merge.csv"
    )
    path_to_results = "/Users/houben/Desktop/test"

    plot_error_vs_tc(results, path_to_results, abs=True)
    #plot_errors_vs_loc(results, path_to_results)
    #plot_heatmap(results, path_to_results)

    # call plot_errors_vs_loc_aggregate for different parameter combinations
    # bins for category "tc_in"
    #error = ["err_S", "err_T", "err_tc"]
    #aggregate = ["S_in", "T_in", "tc_in"]
    #bins = [
    #np.power(10,np.linspace(-5,-1,10)).tolist(),
    #np.power(10,np.linspace(-6,-2,10)).tolist(),
    #[1,10,100,1000,3000,9000,15000,20000,25000,30000,100000],
    #]
    #for err in error:
    #    for i,agg in enumerate(aggregate):
    #        #print(err,agg,bins[i])
    #        plot_errors_vs_loc_aggregate(results, path_to_results, err, agg, bins[i], abs=True)
