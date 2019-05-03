#! /Library/Frameworks/Python.framework/Versions/3.6/bin/python3
# -*- coding: utf-8 -*
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

# add search path for own modules
sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")

# add search path for owdn modules on eve


# own modules
from calc_tc import calc_tc
from processing import *
from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from get_obs import get_obs
from get_ogs_parameters import get_ogs_parameters
from shh_analytical import shh_analytical_fit, shh_analytical
from plot_fitting_results import plot_errors_vs_loc


# ------------------------------------------------------------------------------
# set some parameters for the analysis manually
# ------------------------------------------------------------------------------
aquifer_length = 1000
aquifer_thickness = 30
which = "mean"
m = 5
n = 5
comment = "1_"  # give a specific comment for the analysis e.g. "parameterset1_"
# set cut index and limit recharge and head time series to the first #cut_index values
# set it to None to take all values
cut_index = None
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------s


"""
Description:

- specify a parent folder (path_to_multiple_projects) containing different
    model runs (project_folder) of OGS
- The project_folders have to contain .txt files from different observation
    points as time series and recharge time series as rfd_curve#1.txt.
- Time series will be loaded and all necesarry parameters just from model runs
    and stored in array.
- preprocessing on time series
- power spectrum will be calculated
- fit of power spectrum and parameters will be stored in array.

ToBeDone
--------
- Find an appropriate measure to compare input and output parameters.
- imporove labeling for legend in plot_errors_vs_loc

- get the geometric and arithmetic mean from the field

Requirements
------------
- Head time series must be stored in advace for each obs point in file called:
    "head_ogs_" + obs_point + "_" + which + ".txt"
- obs_point should be formatted like the following: 'obs_00100' with x = 100
- Recharge time series mus me stored in: rfd_curve#1.txt
- Ensure that there no other folder in the directory except for the OGS mode runs.

Yields
------
dataframe : len(project_folder_list) x 15
    name : Name of OGS model run (project_folder)
    S_in : input storativity from OGS model
    T_in : input transmissivity from OGS model
    tc_in : input characteristic time scale calculated from T_in and S_in
    S_out : output storativity from shh_analytical_fit
    T_out : output transmissivity from shh_analytical_fit
    tc_out : output characteristic time scale calculatet from T_out and S_out
    cov : covariance matrix of fit
    err_T : Error in T in % (T_in/T_out)*100
    err_S : Error in S in % (S_in/S_out)*100
    err_tc : Error in tc in % (tc_in/tc_out)*100
    loc : location of the observation point. loc = 0 : water divide
    time_step_size : size of time step in seconds [s]
    time_steps : number of time steps
    model_period : Modelling period in days [d]
    which : Screening deapth of observation point. "mean", "min", "max"
    length : aquifer_length
    thickness : aquifer_thickness
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
project_folder_list = [
    f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(".")
]

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
    "T_in",
    "S_in",
    "tc_in",
    "T_out",
    "S_out",
    "tc_out",
    "cov",
    "err_T",
    "err_S",
    "err_tc",
    "obs_loc",
    "time_step_size",
    "time_steps",
    "model_period",
    "which",
    "recharge"
]
results = pd.DataFrame(columns=columns)
print(results)

# outer loop over all project_folders containing OGS model runs
for i, project_folder in enumerate(project_folder_list):
    path_to_project = path_to_multiple_projects + "/" + project_folder
    # read the OGS model run and its parameters
    recharge_time_series = np.loadtxt(path_to_project + "/" + "rfd_curve#1.txt")
    # write OGS input parameters in DataFrame and multiply Ss and kf by thickness
    Ss, time_step_size, time_steps = get_ogs_parameters(path_to_project, noKf=True)
    # set kf to the geometric or arithmetic mean of the generated conductivity field
    # load from a file with information about the generated field
    field_info = open(path_to_project + '/field_info'+'.dat', 'r')
    for line in field_info:
        dim, var, len_scale, mean, seed, geomean = line.split()
    field_info.close()
    kf = float(geomean)
    S = Ss * aquifer_thickness
    T = kf * aquifer_thickness
    # get list of observation points in current porject_folder
    obs_point_list = get_obs(path_to_project)[1]
    obs_loc_list = get_obs(path_to_project)[2]
    # inner loop over all observation points of current OGS model run
    for j, (obs_point, obs_loc) in enumerate(zip(obs_point_list, obs_loc_list)):
        # Do not perform the fit on observation point x=L
        if obs_loc == aquifer_length:
            break
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

        # cut the time series of head and recharge at a given point
        # ony get the first cut_index values
        head_time_series = head_time_series[:cut_index]
        recharge_time_series = recharge_time_series[:cut_index]

        #
        #
        #
        #
        # calculate the power spectrum
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
        print("T input: ", "{0:.3e}".format(T))
        print(popt, pcov)

        # fill temporal dataframe for one model run
        results_temp = {
            "name": project_folder,
            "S_in": S,
            "T_in": T,
            "tc_in": calc_tc(aquifer_length, S, T),
            "S_out": popt[0],
            "T_out": popt[1],
            "tc_out": calc_tc(aquifer_length, popt[0], popt[1]),
            "cov": pcov,
            "err_S": percent_difference_fraction(S, popt[0]),
            "err_T": percent_difference_fraction(T, popt[1]),
            "err_tc": percent_difference_fraction(
                calc_tc(aquifer_length, S, T), calc_tc(aquifer_length, popt[0], popt[1])
            ),
            "obs_loc": obs_loc,
            "time_step_size": time_step_size,
            "time_steps": time_steps,
            "model_period": time_step_size * time_steps / 86400,
            "which": which,
            "recharge": get_filename_from_rfd_top_com(path_to_project)
        }
        results = results.append(other=results_temp, ignore_index=True, sort=False)

        # plot the power spectra: Shh from ogs runs, Shh theoretical, Shh fitted
        Shh_numerical = Shh
        Shh_theoretical = shh_analytical(
            (frequency, Sww), S, T, obs_loc, aquifer_length, m=5, n=5, norm=False
        )
        Shh_fitted = shh_analytical(
            (frequency, Sww),
            popt[0],
            popt[1],
            obs_loc,
            aquifer_length,
            m=5,
            n=5,
            norm=False,
        )
        data = np.vstack((Shh_numerical, Shh_fitted, Shh_theoretical))
        labels = ["Shh numerical", "Shh fitted", "Shh theoretical"]
        linestyle = ["-", "", ""]
        #lims = [(1e-9,6e-6),(1e-6,1e5)]
        marker = ["", "*", "."]
        figtxt = "OGS Input Parameter: S = %1.3e, T = %1.3e" % (
            S,
            T,
        ) + "\nDerived Parameter:    S = %1.3e, T = %1.3e" % (popt[0], popt[1])
        plot_spectrum(
            data,
            frequency,
            labels=labels,
            path=path_to_project,
         #   lims=lims,
            linestyle=linestyle,
            marker=marker,
            heading="Folder: " + project_folder + "\nLocation: " + str(obs_loc),
            name=comment
            + "PSD_"
            + project_folder
            + "_"
            + str(obs_loc).zfill(len(str(aquifer_length))),
            figtxt=figtxt,
            comment=comment
        )
    time_1_model = time.time() - time_begin
    print(str(time_1_model) + " s elapsed for " + project_folder + "...")
print(results)

# make directory for results
path_to_results = path_to_multiple_projects + "/" + "fitting_results"
if not os.path.exists(path_to_results):
    os.mkdir(path_to_results)
# set path to results incl file name of results
path_to_results_df = (
    path_to_multiple_projects + "/" + "fitting_results" + "/" + comment + "results.csv"
)
# if os.path.isfile(path_to_results_df): # override = true, not necesarry
results.to_csv(path_to_results_df)
plot_errors_vs_loc(results, path_to_results,comment=comment)
time_end = time.time() - time_begin
print("%1.1d min elapsed." % (time_end / 60))
