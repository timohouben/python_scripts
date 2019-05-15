#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
########################################
#   the following must be considered:
# - initial conditions in OGS must be switched off (saturation will be = 1 in whole domain)
# - script not yet functioning for transient analysis
# -

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
import numpy as np
import re
import glob
import time

then = time.time()
cwd = os.getcwd()

# =============================================================================
# extract head over x for 1 time step from OGS TECPLOT from vertical observation lines
# =============================================================================
#### grab all files ending with .tec
tec_files = glob.glob("*.tec")
#### sort the list with file names
tec_files.sort()

#### Defne some useful globar variables
number_obs = sum(1 for line in open("H.OUT", "r")) - 2
obs = np.linspace(0, 1000, 101)
timesteps = 0
z_fit = []
error = 0
head_ogs_obs_mean = []
rmse_anal_mean = 0
rmse_anal_lower = 0
rmse_anal_upper = 0

# number_rows = 0
file_name_ogs = tec_files[0]

#### extracts the number of time steps used (inkl. initial condition)
data_H_ogs = open(file_name_ogs, "r")
for i, line in enumerate(data_H_ogs):
    if line[0].isdigit() != True:
        timesteps = timesteps + 1
timesteps = timesteps / 3

#### generate array for head at obs
head_ogs = np.zeros([number_obs, timesteps])
head_ogs_lower = np.zeros([number_obs, timesteps])
head_ogs_upper = np.zeros([number_obs, timesteps])
head_gw_model = []

#### generate x-z-data for plots
for k, file_name_ogs in enumerate(tec_files):
    head_z_data_ogs = []
    data_H_ogs = open(file_name_ogs, "r")
    #### extracts the number of time steps used (inkl. initial condition)
    for i, line in enumerate(data_H_ogs):
        if line[0].isdigit() == True:
            line_numbers = re.findall(
                "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line
            )
            if float(line_numbers[1]) != 0:
                head_z_data_ogs.append(line_numbers)
    # number_rows = (i + 1 - timesteps * 2) / 2                                       # number of rows each time step

    #### find mean head
    head_ogs_obs_mean_tmp = []
    # convert string to float
    for i, line in enumerate(head_z_data_ogs):
        head_ogs_obs_mean_tmp.append(float(line[1]))
    # calculate mean
    head_ogs_obs_mean.append(np.mean(head_ogs_obs_mean_tmp))

    #### find position of first value (lowest concerning z value of node)
    head_ogs_lower[k] = head_z_data_ogs[0][1]
    #### find position of last value (highest concerning z value of node)
    head_ogs_upper[k] = head_z_data_ogs[len(head_z_data_ogs) - 1][1]

    #### find position of head (old configuration)
    """
    for j, x in enumerate(head_z_data_ogs):
        try:
            if float(head_z_data_ogs[j][1]) > 0:                                     
                head_ogs[k, 0] = head_z_data_ogs[j][1]        
                break
        except: IndexError
     """

data_H_ogs.close()

# =============================================================================
# polynomial fit of ogs data
# =============================================================================
"""
def poly_fit(x):
    fit_coeff = np.polyfit(obs, head_ogs[:,0], 4)
    z = fit_coeff[0]* x**2 + fit_coeff[1]* x**2 + fit_coeff[2] * x**2 + fit_coeff[3] * x + fit_coeff[4]
    return z

for i, line in enumerate(obs):
    line = poly_fit(obs[i])
    z_fit.append(line)
"""

# =============================================================================
# extract head over x for 1 time step from gw_model deRooij 2012
# =============================================================================

data_H_gw_model = open("H.out", "r")


def getlist(data_H_gw_model):
    head_x_data_gw_model = []
    for line in data_H_gw_model:
        line_numbers = re.findall(
            "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line
        )
        head_x_data_gw_model.append(line_numbers)
    del head_x_data_gw_model[0:2]
    data_H_gw_model.close()
    return head_x_data_gw_model


def split(data):
    for i in range(0, len(data)):
        head_gw_model.append(float(data[i][2]))


# =============================================================================
# plot data
# =============================================================================


def plot():
    # fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        "Head Analytical vs Head Numerical\nsteady state - confined\n" + str(cwd)
    )

    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylim(30, 36)
    ax.set_xlim(0, 1010)
    plt.xticks(np.arange(0, 1100, 100))
    plt.yticks(np.arange(30, 36.2, 0.2))
    # configuration for variable ylim
    # plt.yticks(np.arange(30,max(head_ogs_obs_mean)+1,0.2))
    ax.text(
        obs[10],
        (
            (max(head_ogs_lower[:, 0]) - min(head_ogs_lower[:, 0])) * 0.1
            + min(head_ogs_lower[:, 0])
        ),
        "RMSE Anal - Mean: " + str(format(rmse_anal_mean, "10.2E")) + " m",
    )
    ax.text(
        obs[10],
        (
            (max(head_ogs_lower[:, 0]) - min(head_ogs_lower[:, 0])) * 0.15
            + min(head_ogs_lower[:, 0])
        ),
        "RMSE Anal - Lower: " + str(format(rmse_anal_lower, "10.2E")) + " m",
    )
    ax.text(
        obs[10],
        (
            (max(head_ogs_lower[:, 0]) - min(head_ogs_lower[:, 0])) * 0.2
            + min(head_ogs_lower[:, 0])
        ),
        "RMSE Anal - Upper: " + str(format(rmse_anal_upper, "10.2E")) + " m",
    )
    ax.text(
        obs[10],
        (
            (max(head_ogs_lower[:, 0]) - min(head_ogs_lower[:, 0])) * 0.25
            + min(head_ogs_lower[:, 0])
        ),
        "Mean groundwater head: " + str(round(mean_head, 2)) + " m",
    )
    # ax.plot(obs, head_gw_model, label="head analytical [m]")
    ax.plot(obs, head_ogs_obs_mean, label="head ogs mean [m]")
    ax.plot(obs, head_ogs_lower[:, 0], label="head ogs lower [m]")
    ax.plot(obs, head_ogs_upper[:, 0], label="head ogs upper [m]")
    ax.set_title("head vs location")
    ax.set_xlabel("location [m]")
    ax.set_ylabel("head [m]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(
        min(min(head_gw_model, head_ogs_obs_mean)),
        max(max(head_gw_model, head_ogs_obs_mean)),
    )
    ax.set_ylim(
        min(min(head_gw_model, head_ogs_obs_mean)),
        max(max(head_gw_model, head_ogs_obs_mean)),
    )
    # ax5 = ax.scatter(head_gw_model, head_ogs_lower[:, 0])
    # ax6 = ax.scatter(head_gw_model, head_ogs_upper[:, 0])
    ax.scatter(head_gw_model, head_ogs_obs_mean)
    ax.plot(head_gw_model, head_gw_model)
    ax.set_title("scatter analytical vs numerical (ogs mean)")
    ax.set_xlabel("head analytical [m]")
    ax.set_ylabel("head numerical [m]")
    plt.gca().invert_xaxis()

    # plt.show()
    fig.savefig(str(os.path.basename(cwd)) + ".png")
    return fig


# =============================================================================
# execute script
# =============================================================================
mean_head = np.mean(head_ogs_obs_mean[:])
print("Mean groundwater head: " + str(mean_head) + " m")
split(getlist(data_H_gw_model))

# =============================================================================
# calculate RMSE
# =============================================================================
rmse_anal_mean = sum(
    ([(a - b) ** 2 for a, b in zip(head_ogs_obs_mean, head_gw_model)])
) / len(head_ogs_obs_mean)
rmse_anal_lower = sum(
    ([(a - b) ** 2 for a, b in zip(head_ogs_upper[:][0], head_gw_model)])
) / len(head_ogs_obs_mean)
rmse_anal_upper = sum(
    ([(a - b) ** 2 for a, b in zip(head_ogs_lower[:][0], head_gw_model)])
) / len(head_ogs_obs_mean)

# =============================================================================
# plot results
# =============================================================================
now = time.time()
print("ogs-data script runtime: " + str(now - then))
fig = plot()
