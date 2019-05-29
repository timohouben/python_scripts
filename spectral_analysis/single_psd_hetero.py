# -*- coding: utf-8 -*
"""
Script to perform the spectral analysis on a single directory of an ogs model run and stores the results in a folder called "fitting results" in the specific ogs project folder.
"""
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division
# ------------------------------------------------------------------------------

# import modules
import time
time_begin = time.time()
import sys
import numpy as np
import os
import pandas as pd
import os.path

# add search path for own modules
sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
# add search path for owdn modules on eve
sys.path.append("/home/houben/python_pkg/scripts/spectral_analysis")

# own modules
from transect_plot import extract_timeseries, plot_head_timeseries_vs_recharge, extract_rfd
from calc_tc import calc_tc
from processing import *
from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from get_obs import get_obs
from get_ogs_parameters import get_ogs_parameters
from shh_analytical import shh_analytical_fit, shh_analytical
from plot_fitting_results import plot_errors_vs_loc_hetero
from tools import get_ogs_folders

# ------------------------------------------------------------------------------
# set some arameters for the analysis manually
# ------------------------------------------------------------------------------
aquifer_length = 1000
aquifer_thickness = 30
which = "mean"
# m and n are only taken into account if shh_anlytical_man is used. shh_analytical
# also has m and n as arguments but is not using them.
m = None
n = None
comment = "1_"  # give a specific comment for the analysis e.g. "parameterset1_"
# set cut index and limit recharge and head time series to the first #cut_index values
# set it to None to take all values
cut_index = None
# plot the power spectrum normalized by recharge or not
norm = False
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------s


"""
Description:

- Time series will be loaded and all necesarry parameters just from the ogs files
    and stored in array.
- preprocessing on time series should be considered? Detrending?
- power spectrum will be calculated
- fit of power spectrum and parameters will be stored in array.

Requirements
------------
- obs_points should be formatted like the following: 'obs_00100' with x = 100
- Recharge time series mus me stored in: rfd_curve#1_y_values.txt
- Ensure that there is no other folder in the directory except for the OGS mode runs.

Yields
------
dataframe : n_observation_points x n_parameters
    name : Name of OGS model run (project_folder)
    S_in : input storativity from OGS model
    T_in_geo : geometric mean of input conductivity field * thicknes
    T_in_har : harmonic mean of input conductivity field * thicknes
    T_in_ari : arithmetic mean of input conductivity field * thicknes
    tc_in_geo : input characteristic time scale calculated from T_in_geo and S_in
    tc_in_har : input characteristic time scale calculated from T_in_har and S_in
    tc_in_ari : input characteristic time scale calculated from T_in_ari and S_in
    S_out : output storativity from shh_analytical_fit
    T_out : output transmissivity from shh_analytical_fit
    tc_out : output characteristic time scale calculatet from T_out and S_out
    cov : covariance matrix of fit
    err_T_geo : Error in T in % (T_in/T_out)*100
    err_T_har : Error in T in % (T_in/T_out)*100
    err_T_ari : Error in T in % (T_in/T_out)*100
    err_S : Error in S in % (S_in/S_out)*100
    err_tc_geo : Error in tc in % (tc_in/tc_out)*100
    err_tc_har : Error in tc in % (tc_in/tc_out)*100
    err_tc_ari : Error in tc in % (tc_in/tc_out)*100
    loc : location of the observation point. loc = 0 : water divide
    time_step_size : size of time step in seconds [s]
    time_steps : number of time steps
    model_period : Modelling period in days [d]
    which : Screening deapth of observation point. "mean", "min", "max"
    recharge : type of recharge
    aquifer_length : aquifer_length
    aquifer_thickness : aquifer_thickness
    len_scale : correlation length of conductivity field
    variance : variance of conductivity field
    geomean : geomean of conductivity field
    harmean : harmean of conductivity field
    arimean : arimean of conductivity field
    mean
    seed
    dim
"""

# specify the path to the parent directory of multiple OGS model runs
try:
    path_to_multiple_projects = sys.argv[1]
except IndexError:
    print("You forgot to give the path to multiple projects as argument...")
    path_to_multiple_projects = input("Insert path to multiple projects: ")

# path_to_multiple_projects = (
#    "/Users/houben/PhD/modelling/20190318_spectral_analysis_homogeneous/models"
# )

# get a list of all directories containing OGS model runs
project_folder_list = get_ogs_folders(path_to_multiple_projects)
print(
    "Spectral analysis started for parent folder of multiple ogs runs: "
    + path_to_multiple_projects
)

# remove folder "fitting_results" from list and sort
try:
    project_folder_list.remove("fitting_results")
except ValueError:
    pass
project_folder_list.sort()

# initiate the dataframe
pd.set_option("precision", 10)
columns = [
    "name",
    "T_in_geo",
    "T_in_har",
    "T_in_ari",
    "S_in",
    "tc_in_geo",
    "tc_in_har",
    "tc_in_ari",
    "T_out",
    "S_out",
    "tc_out",
    "cov",
    "err_T_geo",
    "err_T_har",
    "err_T_ari",
    "err_S",
    "err_tc_geo",
    "err_tc_har",
    "err_tc_ari",
    "obs_loc",
    "time_step_size",
    "time_steps",
    "model_period",
    "which",
    "recharge",
    "aquifer_length",
    "aquifer_thickness",
    "len_scale",
    "variance",
    "geomean",
    "harmean",
    "arimean",
    "mean",
    "seed",
    "dim"
]
results = pd.DataFrame(columns=columns)


# outer loop over all project_folders containing OGS model runs
for i, project_folder in enumerate(project_folder_list):
    time_1_folder_begin = time.time()
    # initialize the dataframe
    results = pd.DataFrame(columns=columns)
    print("###################################################################")
    print("Starting spectral analysis for folder " + project_folder)
    print("###################################################################")
    path_to_project = path_to_multiple_projects + "/" + project_folder
    # get list of observation points in current porject_folder
    obs_point_list = get_obs(path_to_project, without_max=True)[1]
    obs_loc_list = get_obs(path_to_project, without_max=True)[2]
    # check if time series for different observation points have already been extracted
    checker = []
    for item in obs_point_list:
        if os.path.exists(str(path_to_project) + "/" + "head_ogs_" + str(item) + "_" + str(which) + ".txt"):
            checker.append(True)
    print(checker)
    if all(checker) == True and checker != []:
        print("All time series have already been extracted. Continuing without checking if content is correct.")
    else:
        # extract the time series from the tec files
        print("Extracting time series...")
        extract_timeseries(path_to_project, which="mean", process="GROUNDWATER_FLOW")
    # extract the rfd curve
    time_time_series, recharge_time_series = extract_rfd(
        path=path_to_project, rfd=1
    )
    # plot the time series vs recharge
    plot_head_timeseries_vs_recharge(path=path_to_project)
    # write OGS input parameters in DataFrame, but don't return kf because it is ditrubuted
    Ss, time_step_size, time_steps = get_ogs_parameters(path_to_project, noKf=True)
    # load from a file with information about the generated field
    field_info = open(path_to_project + "/field_info" + ".dat", "r")
    for line in field_info:
        dim, var, len_scale, mean, seed, geomean, harmean, arimean = line.split()
    field_info.close()
    kf_geo = float(geomean)
    kf_har = float(harmean)
    kf_ari = float(arimean)
    S = Ss * aquifer_thickness
    T_in_geo = kf_geo * aquifer_thickness
    T_in_har = kf_har * aquifer_thickness
    T_in_ari = kf_ari * aquifer_thickness
    # make directory for results
    path_to_results = (
        path_to_multiple_projects + "/" + project_folder + "/" + "spectral_analysis"
    )
    if not os.path.exists(path_to_results):
        os.mkdir(path_to_results)
    # inner loop over all observation points of current OGS model run
    for j, (obs_point, obs_loc) in enumerate(zip(obs_point_list, obs_loc_list)):
        # Do not perform the fit on observation point x=L, not necessary any more
        # because get_obs has been modified with "without_max" argument
        if obs_loc == aquifer_length:
            break
        print("###################################################################")
        print("Project folder: " + project_folder)
        print("Observation point: " + obs_point)
        print("Observation point location: " + str(obs_loc))
        # load head time series
        head_time_series = np.loadtxt(
            path_to_multiple_projects
            + "/"
            + project_folder
            + "/"
            + "head_ogs_"
            + obs_point
            + "_"
            + which
            + ".txt"
        )
        # do some preprocessing on time series
        # ------------------------------------
        # DETREND THE HEAD TIME SERIES?

        # cut the time series of head and recharge at a given point
        # ony get the first cut_index values
        head_time_series = head_time_series[:cut_index]
        recharge_time_series = recharge_time_series[:cut_index]
        if cut_index != None:
            print(
                "Time series have been cut. First "
                + str(cut_index)
                + " values remained."
            )
        # calculate the power spectrum: Shh/Sww, output/input to PLOT only!
        frequency_oi, Shh_Sww = power_spectrum(
            input=recharge_time_series,
            output=head_time_series,
            time_step_size=time_step_size,
            method="scipyffthalf",
            o_i="oi",
        )

        # calculate the power spectrum: Shh, output to FIT with analy sol only!
        frequency, Shh = power_spectrum(
            input=recharge_time_series,
            output=head_time_series,
            time_step_size=time_step_size,
            method="scipyffthalf",
            o_i="o",
        )
        frequency, Sww = power_spectrum(
            input=recharge_time_series,
            output=head_time_series,
            time_step_size=time_step_size,
            method="scipyffthalf",
            o_i="i",
        )
        # fit the power spectrum with the analytical solution
        popt, pcov = shh_analytical_fit(
            Sww=Sww,
            Shh=Shh,
            f=frequency,
            x=obs_loc,
            m=m,
            n=n,
            L=aquifer_length,
            norm=False,
        )
        # absolute values for popt because T and S are squared in equation of shh_anlytical
        popt = [abs(i) for i in popt]
        # add values to dataframe
        print("Sy fit: ", "{0:.3e}".format(popt[0]))
        print("Sy input: ", "{0:.3e}".format(S))
        print("T fit: ", "{0:.3e}".format(popt[1]))
        print("T geomean: ", "{0:.3e}".format(T_in_geo))
        print("T harmean: ", "{0:.3e}".format(T_in_har))
        print("T arimean: ", "{0:.3e}".format(T_in_ari))
        print("Covariance of fit:" + str([i for i in pcov]))

        # fill temporal dataframe for one model run
        results_temp = {
            "name": project_folder,
            "S_in": S,
            "T_in_geo": T_in_geo,
            "T_in_har": T_in_har,
            "T_in_ari": T_in_ari,
            "tc_in_geo": calc_tc(aquifer_length, S, T_in_geo),
            "tc_in_har": calc_tc(aquifer_length, S, T_in_har),
            "tc_in_ari": calc_tc(aquifer_length, S, T_in_ari),
            "S_out": popt[0],
            "T_out": popt[1],
            "tc_out": calc_tc(aquifer_length, popt[0], popt[1]),
            "cov": pcov,
            "err_S": percent_difference_fraction(S, popt[0]),
            "err_T_geo": percent_difference_fraction(T_in_geo, popt[1]),
            "err_T_har": percent_difference_fraction(T_in_har, popt[1]),
            "err_T_ari": percent_difference_fraction(T_in_ari, popt[1]),
            "err_tc_geo": percent_difference_fraction(
                calc_tc(aquifer_length, S, T_in_geo),
                calc_tc(aquifer_length, popt[0], popt[1]),
            ),
            "err_tc_har": percent_difference_fraction(
                calc_tc(aquifer_length, S, T_in_har),
                calc_tc(aquifer_length, popt[0], popt[1]),
            ),
            "err_tc_ari": percent_difference_fraction(
                calc_tc(aquifer_length, S, T_in_ari),
                calc_tc(aquifer_length, popt[0], popt[1]),
            ),
            "obs_loc": obs_loc,
            "time_step_size": time_step_size,
            "time_steps": time_steps,
            "model_period": time_step_size * time_steps / 86400,
            "which": which,
            "recharge": get_filename_from_rfd_top_com(path_to_project),
            "aquifer_length": aquifer_length,
            "aquifer_thickness": aquifer_thickness,
            "len_scale": len_scale,
            "variance": var,
            "geomean": geomean,
            "harmean": harmean,
            "arimean": arimean,
            "mean": mean,
            "seed": seed,
            "dim": dim
        }
        results = results.append(other=results_temp, ignore_index=True, sort=False)

        # calculate the power spectra: Shh from ogs runs, Shh theoretical (geo, har, ari), Shh fitted
        Shh_geo = shh_analytical(
            (frequency, Sww), S, T_in_geo, obs_loc, aquifer_length, m=m, n=n, norm=norm
        )
        Shh_har = shh_analytical(
            (frequency, Sww), S, T_in_har, obs_loc, aquifer_length, m=m, n=n, norm=norm
        )
        Shh_ari = shh_analytical(
            (frequency, Sww), S, T_in_ari, obs_loc, aquifer_length, m=m, n=n, norm=norm
        )
        Shh_fitted = shh_analytical(
            (frequency, Sww),
            popt[0],
            popt[1],
            obs_loc,
            aquifer_length,
            m=n,
            n=m,
            norm=norm,
        )

        if norm == True:
            data = np.vstack((Shh_Sww, Shh_fitted, Shh_geo, Shh_har, Shh_ari))
        elif norm == False:
            data = np.vstack((Shh, Shh_fitted, Shh_geo, Shh_har, Shh_ari))

        labels = [
            "Shh numerical",
            "Shh fitted",
            "Shh geometric mean",
            "Shh harmonic mean",
            "Shh arithmetic mean",
        ]
        linestyle = ["-", "-", "--", "--", "--"]
        # lims = [(1e-9,6e-6),(1e-6,1e5)]
        marker = ["", "d", "*", "+", "^"]
        figtxt = "OGS Input Parameter: S = %1.3e, T_geo = %1.3e, T_har = %1.3e, T_ari = %1.3e" % (
            S,
            T_in_geo,
            T_in_har,
            T_in_ari,
        ) + "\nDerived Parameter:    S = %1.3e, T = %1.3e" % (
            popt[0],
            popt[1],
        )

        plot_spectrum(
            data,
            frequency,
            labels=labels,
            path=path_to_results,
            #   lims=lims,
            linestyle=linestyle,
            marker=marker,
            heading="Folder: " + project_folder + "\nLocation: " + str(obs_loc),
            name="PSD_"
            + project_folder
            + "_"
            + str(obs_loc).zfill(len(str(aquifer_length))),
            figtxt=figtxt,
            comment=comment,
        )


    time_1_folder_end = time.time() - time_1_folder_begin
    print(str(time_1_folder_end) + " s elapsed for " + project_folder + "...")
    # set path to results incl file name of results
    path_to_results_df = path_to_results + "/" + comment + "results.csv"
    # if os.path.isfile(path_to_results_df): # override = true, not necesarry
    results.to_csv(path_to_results_df)

    plot_errors_vs_loc_hetero(
        obs=obs_loc_list,
        error_list=[results["err_T_geo"], results["err_T_har"], results["err_T_ari"]],
        legend=["Error T geo.mean", "Error T har.mean", "Error T ari.mean"],
        ylabel="Error [%] " + r"$\frac{T_{in} - T_{out}}{T_{in}} * 100$",
        comment=comment,
        path=path_to_results,
    )
print(results)
print("Ready!")

time_end = time.time() - time_begin
print("%1.1d min elapsed." % (time_end / 60))
