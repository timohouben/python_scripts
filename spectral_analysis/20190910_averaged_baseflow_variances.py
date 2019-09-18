
# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------

"""
Sums up the baseflow from multiple ogs runs and calculate the variance. Plot
the stuff for every time step.
"""

# some variables
filename = "transect_ply_obs_01000_t8_GROUNDWATER_FLOW_flow_timeseries.txt"
path_to_multiple_projects = "/Users/houben/Desktop/eve_work/20190808_generate_ogs_homogeneous_baseflow_sa/setup"

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TKAgg",warn=False, force=True)

from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from transfer_functions import discharge_ftf_fit, discharge_ftf
from transect_plot import extract_rfd


listdir = sorted(os.listdir(path_to_multiple_projects))

clean_listdir = []
for dir in listdir:
    try:
        int(dir[:4])
        clean_listdir.append(dir)
    except ValueError:
        pass

# stor = 0.01, white noise
listdir_1001_1100 = []
for dir in clean_listdir:
    if int(dir[:4]) > 1000 and int(dir[:4]) <= 1100:
        listdir_1001_1100.append(dir)

# stor = 0.01, mHM
listdir_1101_1200 = []
for dir in clean_listdir:
    if int(dir[:4]) > 1100 and int(dir[:4]) <= 1200:
        listdir_1101_1200.append(dir)

# stor = 0.0001, white noise
listdir_1201_1300 = []
for dir in clean_listdir:
    if int(dir[:4]) > 1200 and int(dir[:4]) <= 1300:
        listdir_1201_1300.append(dir)

# stor = 0.0001, mHM
listdir_1301_1400 = []
for dir in clean_listdir:
    if int(dir[:4]) > 1300 and int(dir[:4]) <= 1400:
        listdir_1301_1400.append(dir)

# get the baseflow from every folder and sum it up
baseflow = np.loadtxt(path_to_multiple_projects + "/" + listdir_1001_1100[0] + "/" + filename)
for dir in listdir_1001_1100[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow = np.column_stack((baseflow_tmp, baseflow))
baseflow_sum_1001_1100 = np.sum(baseflow,axis=1)

# get the baseflow from every folder and sum it up
baseflow = np.loadtxt(path_to_multiple_projects + "/" + listdir_1101_1200[0] + "/" + filename)
for dir in listdir_1101_1200[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow = np.column_stack((baseflow_tmp, baseflow))
baseflow_sum_1101_1200 = np.sum(baseflow,axis=1)

# get the baseflow from every folder and sum it up
baseflow = np.loadtxt(path_to_multiple_projects + "/" + listdir_1201_1300[0] + "/" + filename)
for dir in listdir_1201_1300[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow = np.column_stack((baseflow_tmp, baseflow))
baseflow_sum_1201_1300 = np.sum(baseflow,axis=1)

# get the baseflow from every folder and sum it up
baseflow = np.loadtxt(path_to_multiple_projects + "/" + listdir_1301_1400[0] + "/" + filename)
for dir in listdir_1301_1400[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow = np.column_stack((baseflow_tmp, baseflow))
baseflow_sum_1301_1400 = np.sum(baseflow,axis=1)

def moving_variance(timeseries):
    variance=[]
    for timestep in np.arange(1,len(timeseries)+1):
        variance.append(np.var(timeseries[0:timestep]))
    varnorm = [i/np.nanmax(variance) for i in variance]
    return variance, varnorm

####
# execute from here
####


plt.figure(figsize=(16,10))
for baseflow, name, color in zip([baseflow_sum_1001_1100, baseflow_sum_1101_1200, baseflow_sum_1201_1300, baseflow_sum_1301_1400],["0.01, white noise, ensemble", "0.01, mHM, ensemble", "0.0001, white noise, ensemble", "0.0001, mHM, ensemble"],["red", "blue", "green", "black"]):
    plt.plot(moving_variance(baseflow)[1], label=name, linestyle="-", color=color)


## add the other models with single setups
folders = os.listdir("/Users/houben/Desktop/eve_work/20190910_ogs_homogeneous_baseflow_sa_means" + "/setup")
try:
    folders.remove("20190910_variance.py")
    folders.remove("variances.png")
    folders.remove("._variances.png")
    folders.remove("var_results")
    folders.sort()
except ValueError:
    pass


baseflows = []
variances = []

for folder, style, color in zip(sorted(folders),["--",":","-.","--",":","-.","--",":","-.","--",":","-."],["red", "red", "red", "blue","blue","blue", "green","green","green", "black", "black", "black"]):
    basetemp = np.loadtxt("/Users/houben/Desktop/eve_work/20190910_ogs_homogeneous_baseflow_sa_means" + "/setup/" + folder + "/" + "transect_ply_obs_01000_t8_GROUNDWATER_FLOW_flow_timeseries.txt")
    vartemp = moving_variance(basetemp)[1]
    plt.plot(vartemp, label=folder[8:], linestyle=style, color=color)
    print("Finished " + folder + ". The variance approaches in the last time step: " + str(vartemp[-1]))


plt.legend()
#plt.ylim(0.001,1)
plt.title("Evolution of Baseflow Variance over Time")
plt.ylabel("normalized variance")
plt.xlabel("time step [days]")
plt.savefig("/Users/houben/Desktop/eve_work/20190808_generate_ogs_homogeneous_baseflow_sa/variance_analysis/variance.png", dpi=300)
