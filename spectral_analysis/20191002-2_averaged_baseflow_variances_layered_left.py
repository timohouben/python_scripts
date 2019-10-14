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
path_to_multiple_projects = "/Users/houben/Desktop/eve_work/20191002-2_generate_ogs_layered_ensemble_2_left/setup"
path_to_multiple_projects_single = "/Users/houben/Desktop/eve_work/20191013_ogs_layered_means_left/setup"


import sys
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
baseflow_array_1001_1100 = np.loadtxt(path_to_multiple_projects + "/" + listdir_1001_1100[0] + "/" + filename)
for dir in listdir_1001_1100[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow_array_1001_1100 = np.column_stack((baseflow_tmp, baseflow_array_1001_1100))
baseflow_sum_1001_1100 = np.sum(baseflow_array_1001_1100,axis=1)
print("Loading of 1001-1100 finished.")

# get the baseflow from every folder and sum it up
baseflow_array_1101_1200 = np.loadtxt(path_to_multiple_projects + "/" + listdir_1101_1200[0] + "/" + filename)
for dir in listdir_1101_1200[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow_array_1101_1200 = np.column_stack((baseflow_tmp, baseflow_array_1101_1200))
baseflow_sum_1101_1200 = np.sum(baseflow_array_1101_1200,axis=1)
print("Loading of 1101-1200 finished.")

# get the baseflow from every folder and sum it up
baseflow_array_1201_1300 = np.loadtxt(path_to_multiple_projects + "/" + listdir_1201_1300[0] + "/" + filename)
for dir in listdir_1201_1300[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow_array_1201_1300 = np.column_stack((baseflow_tmp, baseflow_array_1201_1300))
baseflow_sum_1201_1300 = np.sum(baseflow_array_1201_1300,axis=1)
print("Loading of 1201-1300 finished.")

# get the baseflow from every folder and sum it up
baseflow_array_1301_1400 = np.loadtxt(path_to_multiple_projects + "/" + listdir_1301_1400[0] + "/" + filename)
for dir in listdir_1301_1400[1:]:
    baseflow_tmp = np.loadtxt(path_to_multiple_projects + "/" + dir + "/" + filename)
    baseflow_array_1301_1400 = np.column_stack((baseflow_tmp, baseflow_array_1301_1400))
baseflow_sum_1301_1400 = np.sum(baseflow_array_1301_1400,axis=1)
print("Loading of 1301-1400 finished.")

def moving_variance(timeseries):
    variance=[]
    for timestep in np.arange(1,len(timeseries)+1):
        variance.append(np.var(timeseries[0:timestep]))
    varnorm = [i/np.nanmax(variance) for i in variance]
    return variance, varnorm

print("Data loading finished!")

## add the other models with single setups
folders = os.listdir(path_to_multiple_projects_single)
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

plt.figure(figsize=(16,10))

####
# Plot The variance of the baseflows as ensemble variance
####
#x, y = extract_rfd(path_to_multiple_projects + "/" + listdir_1101_1200[0],1,export=False)
#plt.plot(np.var(baseflow_array_1101_1200, axis=1), label="ensemble variance")
#plt.plot([i/100000000000 for i in y], label="recharge / 100000000000")
#plt.legend()
#plt.xlim(4000,6000)
#plt.savefig(path_to_multiple_projects[:-6] + "/variance_analysis/ensemble_variance_vs_recharge.png", dpi=300)
#plt.show()
#sys.exit()

####
# Plot The temporal evolution of the baseflows
####
from scipy.stats import gmean, hmean
plt.plot(np.mean(baseflow_array_1001_1100,axis=1), label="ensemble arimean")
plt.plot(gmean(baseflow_array_1001_1100,axis=1), label="ensemble geomean")
#plt.plot(hmean([i*-1 if i < 0 else i for i in baseflow_array_1301_1400],axis=1), label="ensemble harmean")
for folder, style, color in zip(sorted(folders),["--",":","-.","--",":","-.","--",":","-.","--",":","-."],["red", "red", "red", "blue","blue","blue", "green","green","green", "black", "black", "black"]):
    if folder[:4] != "1100":
        continue
    basetemp = np.loadtxt(path_to_multiple_projects_single + "/" + folder + "/" + "transect_ply_obs_01000_t8_GROUNDWATER_FLOW_flow_timeseries.txt")
    plt.plot(basetemp, label=folder[8:], linestyle=style, color=color)
    print("Finished " + folder + ".")
plt.legend()
plt.title("Baseflow over Time")
plt.ylabel("baseflow")
plt.xlabel("time step [days]")
plt.xlim(0,6000)
plt.ylim(1.4e-7,2e-7)
plt.savefig(path_to_multiple_projects[:-6] + "/variance_analysis/1001_1100_baseflow.png", dpi=300)
plt.show()
plt.close()
sys.exit()


####
# Plot The temporal evolution of the variances
####
plt.figure(figsize=(16,10))
for baseflow, name, color in zip([baseflow_sum_1001_1100, baseflow_sum_1101_1200, baseflow_sum_1201_1300, baseflow_sum_1301_1400],["0.01, white noise, ensemble", "0.01, mHM, ensemble", "0.0001, white noise, ensemble", "0.0001, mHM, ensemble"],["red", "blue", "green", "black"]):
    plt.semilogy(moving_variance(baseflow)[1], label=name, linestyle="-", color=color)


for folder, style, color in zip(sorted(folders),["--",":","-.","--",":","-.","--",":","-.","--",":","-."],["red", "red", "red", "blue","blue","blue", "green","green","green", "black", "black", "black"]):
    basetemp = np.loadtxt(path_to_multiple_projects_single + "/" + folder + "/" + "transect_ply_obs_01000_t8_GROUNDWATER_FLOW_flow_timeseries.txt")
    vartemp = moving_variance(basetemp)[1]
    plt.semilogy(vartemp, label=folder[8:], linestyle=style, color=color)
    print("Finished " + folder + ". The variance approaches in the last time step: " + str(vartemp[-1]))


plt.legend()
#plt.ylim(0.001,1)
plt.title("Evolution of Baseflow Variance over Time")
plt.ylabel("normalized variance")
plt.xlabel("time step [days]")
plt.savefig(path_to_multiple_projects[:-6] + "/variance_analysis/moving_variance_norm_log.png", dpi=300)
plt.close()
