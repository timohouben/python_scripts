def plot_combined_violins_for_ensemble_runs(
    path, filename, order, len_scales, y, savename="", y_lims=None, bw=None
):
    """
    Make violin plots for ensemble runs. Each side of a violin represents a
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
    ylims : tuple (y_min,y_max)
        Tuple of limits for y-axis:
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
