"""
Script to plot the results from

20200212_application_spectral_anal_sensitivity.py
"""
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp

def scatter_L_x(results, path):
    """
    Parameters
    ----------
    results : pandas.dataframe
        Should contain the following columns:
            name_h, L, x, S_out, T_out, tc_out

    Yields
    ------
    3 scatter plots in path for each uniqie value of column name_h
    1 for each output parameter T, S and tc. X-achis: L, y-achsis: x

    """

    # remove nans
    results = results[results["cov"] != "[[nan, nan], [nan, nan]]"]
    names = results.name_h.unique()
    lims_for_tc = [(360, 800), (93, 110), (88, 110), (170, 350)]
    for name, lim_for_tc in zip(names, lims_for_tc):
        print("Plotting: " + name)
        x = results.x[results["name_h"] == name].values
        L = results.L[results["name_h"] == name].values
        tc = results.tc_out[results["name_h"] == name].values
        T = results.T_out[results["name_h"] == name].values
        S = results.S_out[results["name_h"] == name].values

        parameters = [T, S, tc]
        para_names = ["T [m2/s]", "S [-]", "tc [days]"]

        for parameter, para_name in zip(parameters, para_names):
            plt.plot(L, L, linestyle="-.", linewidth=0.1)
            if para_name == para_names[2]:
                sc = plt.scatter(L, x, c=parameter, cmap="jet", norm=colors.Normalize(vmin=int(lim_for_tc[0]), vmax=int(lim_for_tc[1])), edgecolors='none')
            if para_name == para_names[1]:
                sc = plt.scatter(L, x, c=parameter, cmap="jet", norm=colors.LogNorm(vmin=8e-3, vmax=2e-1), edgecolors='none')
            if para_name == para_names[0]:
                sc = plt.scatter(L, x, c=parameter, cmap="jet", norm=colors.LogNorm(vmin=1e-4, vmax=5e-1), edgecolors='none')
            plt.xscale("log")
            plt.yscale("log")
            plt.ylim((1e2,5e4))
            plt.xlim((1e2,5e4))
            plt.xlabel("L")
            plt.ylabel("x")
            plt.title(name + ": " + para_name)
            plt.colorbar(sc, extend='max')
            if para_name == "tc [days]":
                plt.savefig(osp.join(path_to_results, name) + "_" + para_name[:2] + ".png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(osp.join(path_to_results, name) + "_" + para_name[0] + ".png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    pass
    #
    path_to_results = "/Users/houben/phd/application_spectral_analysis/main/plots/spectral_analysis/sensitivity_analysis/1_20200212/results"
    results_name = "1_results.csv"
    results = pd.read_csv(osp.join(path_to_results, results_name))
    scatter_L_x(results, path_to_results)
