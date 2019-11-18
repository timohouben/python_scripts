def plot_combined_violins_for_ensemble_runs_hetero(
    path, filename, order, len_scales, y, savename="", y_lims=None, bw=None, yachsis="lin"
):
    """
    Make violin plots for ensemble runs with log-normal distributed hydraulic
    conductivity. Each side of a violin represents a
    correlation length. Each violin represents an observation point.

    Parameters
    ----------

    path : string
        Path to results.
    filename : string
        Name of the results file.
    savename : string
        Specify a name for the file to be saved
    order : list
        Give a list with the observation points
    len_scales : list
        give a list of size 2 with correlation lenghts
    y : string
        Which column of data frame to use for violins.
    ylims : tuple (y_min,y_max), Default: None
        Tuple of limits for y-axis:
    bw : float, Default: None
        Bandwith of violin plots.
    yachsi : string, Default: "lin"
        "lin" : linear y-achsis
        "log" : log y-achsis


    Yields
    ------
    A violin plot in the provided directory.

    """

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(path + "/" + filename)
    fig = plt.figure()
    sns.set_context("paper")
    ax1 = sns.catplot(
        x="obs_loc",
        y=y,
        order=order,
        data=data[data.len_scale.isin(len_scales)],
        kind="violin",
        height=5,
        aspect=2,
        scale="count",
        hue="len_scale",
        split=True,
        legend=False,
        bw=bw,
    )
    if y == "T_out":
        ax2 = plt.hlines(
            y=data.T_in_geo.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="geomean",
            color="c",
            alpha=0.5,
        )
        ax2 = plt.hlines(
            y=data.T_in_har.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="harmean",
            color="r",
            alpha=0.005,
        )
        ax2 = plt.hlines(
            y=data.T_in_ari.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="arimean",
            color="y",
            alpha=0.005,
        )
        ax1.set_ylabels("$T\;[m^2/s]$ derived by fit")
    if y == "S_out":
        ax2 = plt.hlines(
            y=data.S_in.unique(),
            xmin=-0.5,
            xmax=4.5,
            label="S input",
            color="c",
            alpha=0.5,
        )
        ax1.set_ylabels("$S\;[-]$ derived by fit")
    plt.legend(loc="upper left")
    ax1.set_xlabels("location of observation point [m]")
    if yachsis == "log":
        plt.yscale('log')
    elif yachsis == "lin":
        pass
    else:
        print("Argument yachsis has to be 'lin' or 'log'. Selected default 'lin'.")
    plt.ylim(y_lims)
    plt.savefig(
        path
        + "/"
        + savename
        + y
        + "_len_"
        + "_".join([str(i) for i in len_scales])
        + "_obs_"
        + "_".join([str(i) for i in order])
        + "_bw_"
        + str(bw)
        + "_"
        + yachsis
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_combined_violins_for_ensemble_runs_layered(
    path, filename, order, recharges, y, S_in, savename="", y_lims=None, ylog=True, bw=None
):
    """
    Make violin plots for ensemble runs with layered aquifers.
    Each side of a violin represents a recharge process.
    Each violin represents an observation point.

    Parameters
    ----------

    path : string
        Path to results.
    filename : string
        Name of the results file.
    savename : string
        Specify a name for the file to be saved
    order : list
        Give a list with the observation points
    recharges : list
        give a list of size 2 with names of recharge-files
    y : string
        Which column of data frame to use for violins.
    S_in : float
        For which storage value shall data be plotted?
    ylims : tuple (y_min,y_max)
        Tuple of limits for y-axis:
    ylog : Bool (default: True)
        If True, log scale for y-achsis is used
    bw : float
        Bandwith of violin plots.


    Yields
    ------
    A violin plot in the provided directory.

    """

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(path + "/" + filename)
    data = data[data["S_in"] == S_in]
    fig = plt.figure()
    sns.set_context("paper")
    ax1 = sns.catplot(
        x="obs_loc",
        y=y,
        order=order,
        data=data[data.recharge.isin(recharges)],
        kind="violin",
        height=5,
        aspect=2,
        scale="count",
        hue="recharge",
        split=True,
        legend=False,
        bw=bw,
    )
    if y == "T_out":
        ax2 = plt.hlines(
            y=data.T_in_geo.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="geomean",
            color="c",
            alpha=0.5,
        )
        ax3 = plt.hlines(
            y=data.T_in_har.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="harmean",
            color="r",
            alpha=0.5,
        )
        ax4 = plt.hlines(
            y=data.T_in_ari.unique(),
            xmin=-0.5,
            xmax=len(order) - 0.5,
            label="arimean",
            color="y",
            alpha=0.5,
        )
        ax1.set_ylabels("$T\;[m^2/s]$ derived by fit")
    if y == "S_out":
        ax2 = plt.hlines(
            y=data.S_in.unique(),
            xmin=-0.5,
            xmax=4.5,
            label="S input",
            color="c",
            alpha=0.5,
        )
        ax1.set_ylabels("$S\;[-]$ derived by fit")
    plt.legend(loc="upper left")
    ax1.set_xlabels("location of observation point [m]")
    plt.ylim(y_lims)
    plt.title("Derived Transmissivity for Storativity " + str(S_in))
    if ylog == True:
        plt.yscale("log")
    plt.savefig(
        path
        + "/"
        + savename
        + y
        + "_S_"
        + str(S_in)
        + "_len_"
        + "_".join([str(i) for i in recharges])
        + "_obs_"
        + "_".join([str(i) for i in order])
        + "_ylog_"
        + str(ylog)
        + "_bw_"
        + str(bw)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_violins_for_ensemble_runs_layered_baseflow(path, filename, recharges, y, savename="", y_lims=None, ylog=True, bw=None):
    """
    Plot violin plot for each setup (e.g. whitenoise and stor == 0.3)
    next to each other. Categorial violin plot.

    Parameters
    ----------

    path : string
        Path to results.
    filename : string
        Name of the results file.
    savename : string
        Specify a name for the file to be saved
    order : list
        Give a list with the observation points
    recharges : list
        give a list of size 2 with names of recharge-files
    y : string
        Which column of data frame to use for violins.
    S_in : float
        For which storage value shall data be plotted?
    ylims : tuple (y_min,y_max)
        Tuple of limits for y-axis:
    ylog : Bool (default: True)
        If True, log scale for y-achsis is used
    bw : float
        Bandwith of violin plots.

    Yields
    ------
    A violin plot in the provided directory.

    """

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(path + "/" + filename)
    fig = plt.figure()
    sns.set_context("paper")
    ax1 = sns.catplot(
        x="S_in",
        y=y,
        order=data.S_in.unique(),
        data=data[data.recharge.isin(recharges)],
        kind="violin",
        height=5,
        aspect=2,
        scale="count",
        hue="recharge",
        split=True,
        legend=False,
        bw=bw,
    )
    if y == "T_out":
        ax2 = plt.hlines(
            y=data.T_in_geo.unique(),
            xmin=-0.5,
            xmax=2 - 0.5,
            label="geomean",
            color="c",
            alpha=0.5,
        )
        ax3 = plt.hlines(
            y=data.T_in_har.unique(),
            xmin=-0.5,
            xmax=2 - 0.5,
            label="harmean",
            color="r",
            alpha=0.5,
        )
        ax4 = plt.hlines(
            y=data.T_in_ari.unique(),
            xmin=-0.5,
            xmax=2 - 0.5,
            label="arimean",
            color="y",
            alpha=0.5,
        )
        ax1.set_ylabels("$T\;[m^2/s]$ derived by fit")
    if y == "S_out":
        ax2 = plt.hlines(
            y=data.S_in.unique(),
            xmin=-0.5,
            xmax=4.5,
            label="S input",
            color="c",
            alpha=0.5,
        )
        ax1.set_ylabels("$S\;[-]$ derived by fit")
    plt.legend(loc="upper left")
    ax1.set_xlabels("input Storativity [-]")
    plt.ylim(y_lims)
    plt.title("Derived Transmissivity from Baseflow")
    if ylog == True:
        plt.yscale("log")
    plt.savefig(
        path
        + "/"
        + savename
        + y
        + "_baseflow_"
        + "_len_"
        + "_".join([str(i) for i in recharges])
        + "_ylog_"
        + str(ylog)
        + "_bw_"
        + str(bw)
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == '__main__':
    pass
#    order = [[],[]]
#    len_scale = [[],[]]
#    y = [,]
#    for orderT in order:
#        for len_scalesT in len_scales:
#            for yT in y:
#                plot_combined_violins_for_ensemble_runs("/Users/houben/phd/results/20190513_spec_anal_hetero_ensemble_1", "merge_results.csv", orderT, len_scalesT,yT,bw=0.2)


# plot the results for 20191108_results_merge
path = "/Users/houben/phd/results/20190513_spec_anal_hetero_ensemble_1/20191114_combined_results"
filename = "20191108_results_merge.csv"
order = [100,200,500,800,940]
len_scales = [5, 15]
y = "T_out"
yachsis = "log"


plot_combined_violins_for_ensemble_runs_hetero(path, filename, order, len_scales, y, savename="", y_lims=None, bw=None, yachsis=yachsis)






'''
# plot the results for 20190917_ogs_layered_ensemble

path = "/Users/houben/phd/results/20190917_ogs_layered_ensemble"
filename = "1_results_merge.csv"
order = [0,50,250,500,750,900,990]
recharges = ["recharge_daily.txt","recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"]
y = "T_out"
savename="plot"
# y lims for log plot incl all means
#y_lims=[7e-5,1e-1]
# y lims for log plot zoomed in
#y_lims=[3e-3,8e-3]
# y lims for lin plot
y_lims=[0.00002,0.008]
S_in = 0.003
ylog=False

plot_combined_violins_for_ensemble_runs_layered(path, filename, order, recharges, y, S_in=S_in, savename="", y_lims=y_lims, ylog=ylog, bw=None)
'''
'''
# plt the baseflow results for 20190917_ogs_layered_ensemble

path = "/Users/houben/phd/results/20190917_ogs_layered_ensemble"
filename = "1_results_merge.csv"
recharges = ["recharge_daily.txt","recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"]
y = "T_out"
# lims for lin axis
#y_lims = [0.00002,0.008]
# lims for log axis
#y_lims=[7e-5,1e-1]
# lims for log axis zoomed
y_lims=[3e-3,8e-3]
plot_violins_for_ensemble_runs_layered_baseflow(path, filename, recharges, y, savename="", y_lims=y_lims, ylog=True, bw=None)
'''
