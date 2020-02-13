# -*- coding: utf-8 -*
"""
For 'homogeneous' aquifer.

e.g. 20190322_spectral_analysis_homogeneous_neu

Distance to the river will be taken from the numbering of the obeservation
points. Results will be saved in a results.csv. This has to be conbined
afterwards for postprocessing.
For the polyline at location == aquifer length the diffusivity is derived
with the spectral analysis of the baseflow.

In this script you can aggregate the time series and derive the parameters with
different time stepping in the frequency domain.
YOU CAN ONLY perform the SA on 1 obersavtion point at a time.
USE THE parameter: single_location !!
Use the paramter agg_days to state to how many days this should be aggregated.

!!! THIS IS ALL BASED ON A DAILY TIME STEPPING IN THE OGS RFD FILE!!!!

Use:
mpirun -n NUMBERofSLOTS python3 name_of_this_script.py the/working/directory NUMBERofSLOTS



Parameters
----------

aquifer_length
aquifer_thickness
which
comment
# the number of the curve (1st == 1)
recharge_rfd

path_to_multiple_projects as first argument
number of cores as second argument

NEW PARAMETERS

single_location
agg_days

"""
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division
# ------------------------------------------------------------------------------

# import modules
import time
import sys
import numpy as np
import os
import pandas as pd
import os.path
from mpi4py import MPI


# add search path for own modules
sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
# add search path for owdn modules on eve
sys.path.append("/home/houben/python/scripts/spectral_analysis")

# own modules
from transect_plot import extract_timeseries, plot_head_timeseries_vs_recharge, extract_rfd
from calc_tc import calc_tc
from processing import *
from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from get_obs import get_obs
from get_ogs_parameters import get_ogs_parameters
from shh_analytical import shh_analytical_fit, shh_analytical
from plot_fitting_results import plot_errors_vs_loc_hetero, plot_parameter_vs_location
from tools import get_ogs_folders
from calculate_flow import plot_recharge_vs_baseflow, get_baseflow_from_polyline
from tools import get_ogs_task_id
from transfer_functions import discharge_ftf_fit, discharge_ftf
# ------------------------------------------------------------------------------
# set some arameters for the analysis manually
# ------------------------------------------------------------------------------
aquifer_length = 1000
aquifer_thickness = 30
which = "mean"
# convergence criterion: Series of Shh analytical will be truncated when next
# iteration adds less than this realtive value
convergence = 0.1
# the number of the curve (1st == 1)
recharge_rfd = 1
# m an,d n are only taken into account if shh_anlytical_man is used. shh_analytical
# also ,has m and n as arguments but is not using them.
m = None
n = None
comment = "1_day_"  # give a specific comment for the analysis e.g. "parameterset1_"
# set cut index and limit recharge and head time series to the first #cut_index values
# set it to None to take all values
cut_index = None
# plot the power spectrum normalized by recharge or not
norm = False
# set the location you want to analyse
single_location = 500
# set the number of days you want to aggregate to
agg_days = 1


# ------------------------------------------------------------------------------
# some parameters for the mpi run
# ------------------------------------------------------------------------------
# get the number of slots from a system argument
slots = int(sys.argv[2])
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Description:

- Time series will be loaded and all necessary parameters just from the ogs files
    and stored in array.
- preprocessing on time series should be considered? Detrending?
- power spectrum will be calculated
- fit of power spectrum and parameters will be stored in results.csv.

Requirements
------------
- obs_points should be formatted like the following: 'obs_00100' with x = 100
- Recharge time series mus me stored in: rfd_curve#1_y_values.txt !!!!
- Ensure that there is no other folder in the directory except for the OGS mode runs.

Yields
------
dataframe : n_observation_points x n_parameters
    name : Name of OGS model run (project_folder)
    S_in : input storativity
    T_in : input transmissivity
    D_in : input Diffusivity
    S_out : output storativity from shh_analytical_fit
    T_out : output transmissivity from shh_analytical_fit
    tc_out : output characteristic time scale calculatet from T_out and S_out
    cov : covariance matrix of fit
    loc : location of the observation point. loc = 0 : water divide
    time_step_size : size of time step in seconds [s]
    time_steps : number of time steps
    model_period : Modelling period in days [d]
    which : Screening deapth of observation point. "mean", "min", "max"
    recharge : type of recharge
    aquifer_length : aquifer_length
    aquifer_thickness : aquifer_thickness
    D : Diffusivity
    D_cov : Covariance of fit or Diffusivity
"""

# specify the path to the parent directory of multiple OGS model runs
try:
    path_to_multiple_projects = sys.argv[1]
except IndexError:
    print("You forgot to give the path to multiple projects as argument...")
    path_to_multiple_projects = input("Insert path to multiple projects: ")

# get a list of all directories containing OGS model runs
project_folder_list = get_ogs_folders(path_to_multiple_projects)

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
    "S_in",
    "T_in",
    "D_in",
    "T_out",
    "S_out",
    "tc_out",
    "cov",
    "obs_loc",
    "time_step_size",
    "time_steps",
    "model_period",
    "which",
    "recharge",
    "aquifer_length",
    "aquifer_thickness",
    "D",
    "D_cov",
]

# outer loop over all project_folders containing OGS model runs
for i, project_folder in enumerate(project_folder_list):
    if i%slots == rank:
        time_1_folder_begin = time.time()
        # initialize the dataframe
        results = pd.DataFrame(columns=columns)
        print("###################################################################")
        print("Starting spectral analysis for folder " + project_folder + " on rank " + str(rank))
        print("###################################################################")
        path_to_project = path_to_multiple_projects + "/" + project_folder
        # get list of observation points in current porject_folder
        obs_point_list = get_obs(path_to_project, without_max=False)[1]
        obs_loc_list = get_obs(path_to_project, without_max=False)[2]
        obs_point_list = [i for i in obs_point_list if str(single_location) in i or str(aquifer_length) in i]
        obs_loc_list = [i for i in obs_loc_list if single_location == i or aquifer_length == i]
        print(obs_point_list)
        print(obs_loc_list)
        # check if time series for different observation points have already been extracted
        checker = []
        for item in obs_point_list:
            if os.path.exists(str(path_to_project) + "/" + "head_ogs_" + str(item) + "_" + str(which) + ".txt"):
                checker.append(True)
            else:
                checker.append(False)
        if all(checker) == True and checker != []:
            print("All time series have already been extracted. Continuing without checking if content is correct.")
        else:
            # extract the time series from the tec files
            print("Extracting time series...")
            extract_timeseries(path_to_project, which="mean", process="GROUNDWATER_FLOW")
        # extract the rfd curve
        time_time_series, recharge_time_series = extract_rfd(path=path_to_project, rfd=recharge_rfd)
        # aggregate the head_time_series
        recharge_time_series = aggregate_series(recharge_time_series, agg_days)
        # plot the time series vs recharge
        #plot_head_timeseries_vs_recharge(path=path_to_project)
        # write OGS input parameters in DataFrame, but don't return kf because it is ditrubuted
        Ss_in, kf_in, time_step_size, time_steps = get_ogs_parameters(path_to_project, noKf=False)
        time_step_size = time_step_size * agg_days
        S_in = Ss_in * aquifer_thickness
        T_in = kf_in * aquifer_thickness
        # make directory for results
        path_to_results = (
            path_to_multiple_projects + "/" + project_folder + "/" + "spectral_analysis"
        )
        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)
        np.savetxt(path_to_results + "/" + comment + "recharge.txt", recharge_time_series)
        # inner loop over all observation points of current OGS model run
        # change the order of the lists
        #myorder = np.arange(-len(obs_point_list)+1,1)*-1
        #obs_point_list = [obs_point_list[i] for i in myorder]
        #obs_loc_list = [obs_loc_list[i] for i in myorder]
        for j, (obs_point, obs_loc) in enumerate(zip(obs_point_list, obs_loc_list)):
            print("###################################################################")
            print("Project folder: " + project_folder)
            print("Observation point: " + obs_point)
            print("Observation point location: " + str(obs_loc))
            if obs_loc == aquifer_length:
                print("Spectral analysis for the baseflow (not every functionality is considered (e.g. cut_index, norm))")
                # If the current observation point is equal to the aquifer
                # length, it is assumed that this polyline-file contains the
                # velocities to calculate the baseflow. First, the baseflow
                # is calculated and afterwards, the diffusivity is derived
                # with the spectral analysis.
                task_id = get_ogs_task_id(path_to_project)
                # baseflow = get_baseflow_from_polyline(task_id, path_to_project, path_to_project + "/" + task_id + "_ply_obs_01000_t" + str(len(obs_point_list)) + "_GROUNDWATER_FLOW.tec")
                baseflow_q = np.loadtxt(path_to_project + "/transect_ply_obs_01000_t20_GROUNDWATER_FLOW_flow_timeseries.txt")
                baseflow_q = aggregate_series(baseflow_q, agg_days)
                np.savetxt(path_to_results + "/" + comment + "baseflow.txt", baseflow_q)
                # multiply the recharge time series with the aquifer length to get the total inflow
                recharge_q = recharge_time_series * aquifer_length
                try:
                    D, D_cov, frequency, Sqq = discharge_ftf_fit(recharge_q, baseflow_q, time_step_size, aquifer_length)
                except RuntimeError:
                    print("Optimal parameters not found...")
                    D[0], D_cov[0] = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
                    print("popt and pcov have been set to np.nan")
                except ValueError:
                    print("either ydata or xdata contain NaNs, or if incompatible options are used")
                    D[0], D_cov[0] = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
                except OptimizeWarning:
                    print("Covariance of the parameters could not be estimated.")
                    #popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]

                # add values to dataframe
                print("D: ", "{0:.3e}".format(D[0]))
                print("Covariance of fit:" + str(D_cov[0]))

                # fill temporal dataframe for one model run
                results_temp = {
                    "name": project_folder,
                    "S_in": S_in,
                    "T_in": T_in,
                    "D_in": T_in / S_in,
                    "T_out": np.nan,
                    "S_out": np.nan,
                    "tc_out": np.nan,
                    "cov": np.nan,
                    "obs_loc": obs_loc,
                    "time_step_size": time_step_size,
                    "time_steps": time_steps,
                    "model_period": time_step_size * time_steps / 86400,
                    "which": which,
                    "recharge": get_filename_from_rfd_top_com(path_to_project),
                    "aquifer_length": aquifer_length,
                    "aquifer_thickness": aquifer_thickness,
                    "D": D[0],
                    "D_cov": D_cov[0],
                }

                results = results.append(other=results_temp, ignore_index=True, sort=False)

                # calculate the fitted power spectra
                Sqq_fitted = discharge_ftf(frequency, D, aquifer_length)

                Sqq_fitted = np.reshape(Sqq_fitted,(len(Sqq_fitted),))
                Sqq = np.reshape(Sqq,(len(Sqq),))

                data = np.vstack((Sqq, Sqq_fitted))

                labels = [
                    "Sqq numerical",
                    "Sqq fitted"
                ]

                linestyle = ["-", "-"]
                # lims = [(1e-9,6e-6),(1e-6,1e5)]
                marker = ["", ""]
                figtxt = "OGS Input Parameter: S = %1.3e, T = %1.3e, D = %1.3e" % (
                    S_in,
                    T_in,
                    T_in/S_in
                ) + "\nDerived Parameter:    D = %1.3e, D_cov = %1.1e" % (
                    D[0],
                    D_cov[0],
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
                    name="SA_"
                    + project_folder
                    + "_"
                    + str(obs_loc).zfill(len(str(aquifer_length)))
                    + "_baseflow",
                    figtxt=figtxt,
                    comment=comment,
                )
                # break this itearation and continue with next obs point
                continue
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

            # aggregate the head_time_series
            head_time_series = aggregate_series(head_time_series, agg_days)
            np.savetxt(path_to_results + "/" + comment + str(obs_loc) + "_head.txt", head_time_series)
            # calculate the power spectrum: Shh/Sww, output/input to PLOT only!
            frequency_oi, Shh_Sww = power_spectrum(
                input=recharge_time_series,
                output=head_time_series,
                time_step_size=time_step_size,
                method="scipyffthalf",
                o_i="oi",
            )

            # calculate the power spectrum: Shh, output to FIT with analy solution only!
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
            try:
                popt, pcov = shh_analytical_fit(
                    Sww=Sww,
                    Shh=Shh,
                    f=frequency,
                    x=obs_loc,
                    m=m,
                    n=n,
                    L=aquifer_length,
                    norm=False,
                    convergence=convergence,
                )
            except RuntimeError:
                print("Optimal parameters not found...")
                popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
                print("popt and pcov have been set to np.nan")
            except ValueError:
                print("either ydata or xdata contain NaNs, or if incompatible options are used")
                popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
            except OptimizeWarning:
                print("Covariance of the parameters could not be estimated.")
                #popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]


            # absolute values for popt because T and S are squared in equation
            # of shh_anlytical and negative values are possible
            popt = [abs(i) for i in popt]
            # add values to dataframe
            print("S fit: ", "{0:.3e}".format(popt[0]))
            print("S input: ", "{0:.3e}".format(S_in))
            print("T fit: ", "{0:.3e}".format(popt[1]))
            print("T input: ", "{0:.3e}".format(T_in))
            print("Covariance of fit:" + str([i for i in pcov]))

            # fill temporal dataframe for one model run
            results_temp = {
                "name": project_folder,
                "S_in": S_in,
                "T_in": T_in,
                "D_in": T_in/S_in,
                "T_out": popt[1],
                "S_out": popt[0],
                "tc_out": calc_tc(aquifer_length, popt[0], popt[1]),
                "cov": pcov,
                "obs_loc": obs_loc,
                "time_step_size": time_step_size,
                "time_steps": time_steps,
                "model_period": time_step_size * time_steps / 86400,
                "which": which,
                "recharge": get_filename_from_rfd_top_com(path_to_project),
                "aquifer_length": aquifer_length,
                "aquifer_thickness": aquifer_thickness,
                "D": np.nan,
                "D_cov": np.nan,
            }

            results = results.append(other=results_temp, ignore_index=True, sort=False)

            # calculate the fitted power spectra
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
                data = np.vstack((Shh_Sww, Shh_fitted))
            elif norm == False:
                data = np.vstack((Shh, Shh_fitted))

            labels = [
                "Shh numerical",
                "Shh fitted"
            ]
            linestyle = ["-", "-"]
            # lims = [(1e-9,6e-6),(1e-6,1e5)]
            marker = ["", "d"]
            figtxt = "OGS Input Parameter: S = %1.3e, T = %1.3e" % (
                S_in,
                T_in,
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
                name="SA_"
                + project_folder
                + "_"
                + str(obs_loc).zfill(len(str(aquifer_length))),
                figtxt=figtxt,
                comment=comment,
            )


        time_1_folder_end = time.time() - time_1_folder_begin
        print("Ready! " + str(time_1_folder_end) + " s elapsed for " + project_folder + "...")
        # set path to results incl file name of results
        path_to_results_df = path_to_results + "/" + comment + "results.csv"
        # if os.path.isfile(path_to_results_df): # override = true, not necesarry
        results.to_csv(path_to_results_df)

        # a few lines to plot the error, variance of the fit also as an error plot, not working yet!!!
        #results["cov_numbers"] = results["cov"].apply(identify_numbers_from_string)
        #results["sigma_T"] = results["cov_numbers"].apply(lambda x: x[3] if x != [] else np.nan)
        #results["sigma_S"] = results["cov_numbers"].apply(lambda x: x[0] if x != [] else np.nan)

        plot_parameter_vs_location(path_to_results, results["T_out"], obs_loc_list, y_label="T_out")
