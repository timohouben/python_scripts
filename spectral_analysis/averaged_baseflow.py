
# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------

"""
Sums up the baseflow and the recharge from multiple ogs runs
and performs a spectra analysis. The derived diffusivity will be
compared to the geomean, harmean and arimean of the input
diffusivity values.
"""

# some variables
aquifer_length = 1000
filename = "transect_ply_obs_01000_t8_GROUNDWATER_FLOW_flow_timeseries.txt"
path_to_multiple_projects = "/Users/houben/Desktop/eve_work/20190808_generate_ogs_homogeneous_baseflow_sa/setup"
path_to_kf_values = "/Users/houben/Desktop/eve_work/20190808_generate_ogs_homogeneous_baseflow_sa/kf_values/kf_list_file.txt"
Ss1 = 0.01
Ss2 = 0.0001

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

# get the recharge and multiply it by the amount of model: White Noise
rfd = extract_rfd(path_to_multiple_projects + "/" + listdir_1001_1100[0], export=False)
# multiplied with the number of models and multiplied with the aquifer length to get the flow
recharge_1001_1100 = rfd[1] * len(listdir_1001_1100) * aquifer_length

# get the recharge and multiply it by the amount of model: mHM
rfd = extract_rfd(path_to_multiple_projects + "/" + listdir_1101_1200[0], export=False)
# multiplied with the number of models and multiplied with the aquifer length to get the flow
recharge_1101_1200 = rfd[1] * len(listdir_1101_1200) * aquifer_length

# get the recharge and multiply it by the amount of model: White Noise
rfd = extract_rfd(path_to_multiple_projects + "/" + listdir_1201_1300[0], export=False)
# multiplied with the number of models and multiplied with the aquifer length to get the flow
recharge_1201_1300 = rfd[1] * len(listdir_1201_1300) * aquifer_length

# get the recharge and multiply it by the amount of model: mHM
rfd = extract_rfd(path_to_multiple_projects + "/" + listdir_1301_1400[0], export=False)
# multiplied with the number of models and multiplied with the aquifer length to get the flow
recharge_1301_1400 = rfd[1] * len(listdir_1301_1400) * aquifer_length

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

# load the ogs input parameter from a file
kf_list_file = open(path_to_kf_values, "r")
for i, line in enumerate(kf_list_file):
    if i == 1:
        splitted_line = line.strip().split(", ")
        kf_geomean = float(splitted_line[0])
        kf_harmean = float(splitted_line[1])
        kf_arimean = float(splitted_line[2])

# calculate the power spectrum for 1001_1100: White Noise, 0.01
D_1001_1100, D_cov, frequency_1001_1100, power_spectrum_1001_1100 = discharge_ftf_fit(recharge_1001_1100, baseflow_sum_1001_1100, 86400, 1000)
power_spectrum_1001_1100_anal = discharge_ftf(frequency_1001_1100, D_1001_1100, aquifer_length)
power_spectrum_1001_1100_anal = np.reshape(power_spectrum_1001_1100_anal,(len(power_spectrum_1001_1100_anal),))
power_spectrum_1001_1100 = np.reshape(power_spectrum_1001_1100,(len(power_spectrum_1001_1100),))
figtxt = "OGS Input Parameter: Ss = %1.3e, D_geo = %1.3e, D_har = %1.3e, D_ari = %1.3e" % (
    Ss1,
    kf_geomean / Ss1,
    kf_harmean / Ss1,
    kf_arimean / Ss1,
) + "\nDerived Parameter:    D = %1.3e, D_cov = %1.1e" % (
    D_1001_1100[0],
    D_cov[0],
)
plot_spectrum(np.vstack((power_spectrum_1001_1100, power_spectrum_1001_1100_anal)), frequency_1001_1100, name="white_noise_0.01", labels=["Sqq","Sqq_fitted"], heading="Recharge: white noise, Stor = 0.01", marker=["",""], linestyle=["-","-"], path="/Users/houben/Desktop/baseflow_sa", figtxt=figtxt)

# calculate the power spectrum for 1101_1200
D_1101_1100, D_cov, frequency_1101_1200, power_spectrum_1101_1200 = discharge_ftf_fit(recharge_1101_1200, baseflow_sum_1101_1200, 86400, 1000)
power_spectrum_1101_1200_anal = discharge_ftf(frequency_1101_1200, D_1101_1100, aquifer_length)
power_spectrum_1101_1200_anal = np.reshape(power_spectrum_1101_1200_anal,(len(power_spectrum_1101_1200_anal),))
power_spectrum_1101_1200 = np.reshape(power_spectrum_1101_1200,(len(power_spectrum_1101_1200),))
figtxt = "OGS Input Parameter: Ss = %1.3e, D_geo = %1.3e, D_har = %1.3e, D_ari = %1.3e" % (
    Ss1,
    kf_geomean / Ss1,
    kf_harmean / Ss1,
    kf_arimean / Ss1,
) + "\nDerived Parameter:    D = %1.3e, D_cov = %1.1e" % (
    D_1101_1100[0],
    D_cov[0],
)
plot_spectrum(np.vstack((power_spectrum_1101_1200, power_spectrum_1101_1200_anal)), frequency_1101_1200, name="mHM_0.01", labels=["Sqq","Sqq_fitted"], heading="Recharge: mHM, Stor = 0.01", marker=["",""], linestyle=["-","-"], path="/Users/houben/Desktop/baseflow_sa", figtxt=figtxt)

# calculate the power spectrum for 1201_1300
D_1201_1300, D_cov, frequency_1201_1300, power_spectrum_1201_1300 = discharge_ftf_fit(recharge_1201_1300, baseflow_sum_1201_1300, 86400, 1000)
power_spectrum_1201_1300_anal = discharge_ftf(frequency_1201_1300, D_1201_1300, aquifer_length)
power_spectrum_1201_1300_anal = np.reshape(power_spectrum_1201_1300_anal,(len(power_spectrum_1201_1300_anal),))
power_spectrum_1201_1300 = np.reshape(power_spectrum_1201_1300,(len(power_spectrum_1201_1300),))
figtxt = "OGS Input Parameter: Ss = %1.3e, D_geo = %1.3e, D_har = %1.3e, D_ari = %1.3e" % (
    Ss2,
    kf_geomean / Ss2,
    kf_harmean / Ss2,
    kf_arimean / Ss2,
) + "\nDerived Parameter:    D = %1.3e, D_cov = %1.1e" % (
    D_1201_1300[0],
    D_cov[0],
)
plot_spectrum(np.vstack((power_spectrum_1201_1300, power_spectrum_1201_1300_anal)), frequency_1201_1300, name="whitenoise_0.0001", labels=["Sqq","Sqq_fitted"], heading="Recharge: white noise, Stor = 0.0001", marker=["",""], linestyle=["-","-"], path="/Users/houben/Desktop/baseflow_sa", figtxt=figtxt)

# calculate the power spectrum for 1301_1400
D_1301_1400, D_cov, frequency_1301_1400, power_spectrum_1301_1400 = discharge_ftf_fit(recharge_1301_1400, baseflow_sum_1301_1400, 86400, 1000)
power_spectrum_1301_1400_anal = discharge_ftf(frequency_1301_1400, D_1301_1400, aquifer_length)
power_spectrum_1301_1400_anal = np.reshape(power_spectrum_1301_1400_anal,(len(power_spectrum_1301_1400_anal),))
power_spectrum_1301_1400 = np.reshape(power_spectrum_1301_1400,(len(power_spectrum_1301_1400),))
figtxt = "OGS Input Parameter: Ss = %1.3e, D_geo = %1.3e, D_har = %1.3e, D_ari = %1.3e" % (
    Ss2,
    kf_geomean / Ss2,
    kf_harmean / Ss2,
    kf_arimean / Ss2,
) + "\nDerived Parameter:    D = %1.3e, D_cov = %1.1e" % (
    D_1301_1400[0],
    D_cov[0],
)
plot_spectrum(np.vstack((power_spectrum_1301_1400, power_spectrum_1301_1400_anal)), frequency_1301_1400, name="mHM_0.0001", labels=["Sqq","Sqq_fitted"], heading="Recharge: mHM, Stor = 0.0001", marker=["",""], linestyle=["-","-"], path="/Users/houben/Desktop/baseflow_sa", figtxt=figtxt)




#########
# plot time series
plt.figure(figsize=(16,10))
plt.bar(np.linspace(1,len(recharge_1001_1100),len(recharge_1001_1100)),recharge_1001_1100, label="Recharge white noise", color="#1f77b4")
plt.bar(np.linspace(1,len(recharge_1101_1200),len(recharge_1101_1200)),recharge_1101_1200, label="Recharge mHM", color="#ff7f0e")
#plt.bar(recharge_1101_1200, label="Recharge mHM")
#plt.plot(recharge_1201_1300)
#plt.plot(recharge_1301_1400)
plt.plot(baseflow_sum_1201_1300, label="Baseflow white noise, Stor = 0.0001", linestyle="-", color="#1f77b4")
plt.plot(baseflow_sum_1001_1100, label="Baseflow white noise, Stor = 0.01", linestyle="-", color="#2ca02c")
plt.plot(baseflow_sum_1101_1200, label="Baseflow mHM, Stor = 0.01", linestyle="-", color="#ff7f0e")
plt.plot(baseflow_sum_1301_1400, label="Baseflow mHM, Stor = 0.0001", linestyle="-", color="#d62728")
plt.ylabel("Flow [L/T]")
plt.xlabel("Time [day]")
plt.legend(loc="best")
plt.savefig("/Users/houben/Desktop/baseflow_sa/time_series.png", dpi=300)


#########
# calculate and plot the means from single spectral analysis
import pandas as pd
results = pd.read_csv("/Users/houben/Desktop/eve_work/20190808_generate_ogs_homogeneous_baseflow_sa/combined_results/baseflow_results_merge.csv")
plt.figure(figsize=(16,10))
results_sorted = results.sort_values("D_in")
x_values = np.linspace(1,len(results),100)
D_in_001 = results_sorted["D_in"][results_sorted["recharge"] == "recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"][results_sorted["S_in"] == 0.3]
D_in_00001 = results_sorted["D_in"][results_sorted["recharge"] == "recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"][results_sorted["S_in"] == 0.003]
D_out_mhm_001 = results_sorted["D"][results_sorted["recharge"] == "recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"][results_sorted["S_in"] == 0.3]
D_out_mhm_00001 = results_sorted["D"][results_sorted["recharge"] == "recharge_daily_30years_seconds_mm_mHM_estanis_danube.txt"][results_sorted["S_in"] == 0.003]
D_out_wn_001 = results_sorted["D"][results_sorted["recharge"] == "recharge_daily.txt"][results_sorted["S_in"] == 0.3]
D_out_wn_00001 = results_sorted["D"][results_sorted["recharge"] == "recharge_daily.txt"][results_sorted["S_in"] == 0.003]
plt.semilogy(x_values, D_in_001, color="black", linestyle="-", marker="", markersize="9", label="D in")
plt.semilogy(x_values, D_in_00001, color="black", linestyle="-", marker="", markersize="9", label="")
#plt.semilogy(x_values, results_sorted["D_in"][results_sorted["recharge"] == "recharge_daily.txt"][results_sorted["S_in"] == 0.3], color="orange", linestyle="-", marker="", markersize="10", label="D in : white noise, Ss = 0.01")
#plt.semilogy(x_values, results_sorted["D_in"][results_sorted["recharge"] == "recharge_daily.txt"][results_sorted["S_in"] == 0.003], color="orange", linestyle="-", marker="", markersize="10", label="D in : white noise, Ss = 0.0001")
plt.semilogy(x_values, D_out_mhm_001, color="blue", linestyle="", marker="o", markersize="10", label="D out : mHM, Ss = 0.01")
plt.semilogy(x_values, D_out_wn_001, color="orange", linestyle="", marker="o", markersize="10", label="D out : white noise, Ss = 0.01")
plt.semilogy(x_values, D_out_mhm_00001, color="blue", linestyle="", marker="*", markersize="10", label="D out : mHM, Ss = 0.0001")
plt.semilogy(x_values, D_out_wn_00001, color="orange", linestyle="", marker="*", markersize="10", label="D out : white noise, Ss = 0.0001")
plt.legend()
plt.ylabel("Diffusivity [m^2/s]")
plt.xlabel("model")
plt.savefig("/Users/houben/Desktop/baseflow_sa/in_vs_out.png", dpi=300)


# calculate the geomean, harmean, arimean from the derived values:
from scipy.stats.mstats import gmean, hmean
geomean_D_in_001 = gmean(D_in_001)
harmean_D_in_001 = hmean(D_in_001)
arimean_D_in_001 = np.mean(D_in_001)
geomean_D_in_00001 = gmean(D_in_00001)
harmean_D_in_00001 = hmean(D_in_00001)
arimean_D_in_00001 = np.mean(D_in_00001)
geomean_D_out_mhm_001 = gmean(D_out_mhm_001)
harmean_D_out_mhm_001 = hmean(D_out_mhm_001)
arimean_D_out_mhm_001 = np.mean(D_out_mhm_001)
geomean_D_out_wn_001 = gmean(D_out_wn_001)
harmean_D_out_wn_001 = hmean(D_out_wn_001)
arimean_D_out_wn_001 = np.mean(D_out_wn_001)
geomean_D_out_mhm_00001 = gmean(D_out_mhm_00001)
harmean_D_out_mhm_00001 = hmean(D_out_mhm_00001)
arimean_D_out_mhm_00001 = np.mean(D_out_mhm_00001)
geomean_D_out_wn_00001 = gmean(D_out_wn_00001)
harmean_D_out_wn_00001 = hmean(D_out_wn_00001)
arimean_D_out_wn_00001 = np.mean(D_out_wn_00001)

########
# plot scatter plot for mean values
#geomeans = {'mhm_001': geomean_D_out_mhm_001, 'whitenoise_001': geomean_D_out_wn_001, 'mhm_00001': geomean_D_out_mhm_00001, 'whitenoise_00001': geomean_D_out_wn_00001}
#harmeans = {'mhm_001': harmean_D_out_mhm_001, 'whitenoise_001': harmean_D_out_wn_001, 'mhm_00001': harmean_D_out_mhm_00001, 'whitenoise_00001': harmean_D_out_wn_00001}
#arimeans = {'mhm_001': arimean_D_out_mhm_001, 'whitenoise_001': arimean_D_out_wn_001, 'mhm_00001': arimean_D_out_mhm_00001,  'whitenoise_00001': arimean_D_out_wn_00001}
#names_geo = list(geomeans.keys())
#values_geo = list(geomeans.values())
#names_har = list(harmeans.keys())
#values_har = list(harmeans.values())
#names_ari = list(arimeans.keys())
#values_ari = list(arimeans.values())
#plt.figure(figsize=(9,9))
#plt.scatter(names_geo, values_geo, label = "geomean", color="blue")
#plt.scatter(names_har, values_har, label = "harmean", color="orange")
#plt.scatter(names_ari, values_ari, label = "arimean", color="red")
#plt.yscale('log')
#plt.hlines(y=geomean_D_in_001, xmin=0, xmax=1, color="blue")
#plt.hlines(y=harmean_D_in_001, xmin=0, xmax=1, color="orange")
#plt.hlines(y=arimean_D_in_001, xmin=0, xmax=1, color="red")
#plt.hlines(y=geomean_D_in_00001, xmin=2, xmax=3, color="blue")
#plt.hlines(y=harmean_D_in_00001, xmin=2, xmax=3, color="orange")
#plt.hlines(y=arimean_D_in_00001, xmin=2, xmax=3, color="red")
#plt.ylabel("Diffusivity [m^2/s]")
#plt.legend()
#plt.savefig("/Users/houben/Desktop/baseflow_sa/in_vs_out_means.png", dpi=300)

########
# plot for integrated recharge
geomeans = {'mhm_001': geomean_D_out_mhm_001, 'whitenoise_001': geomean_D_out_wn_001, 'mhm_00001': geomean_D_out_mhm_00001, 'whitenoise_00001': geomean_D_out_wn_00001}
harmeans = {'mhm_001': harmean_D_out_mhm_001, 'whitenoise_001': harmean_D_out_wn_001, 'mhm_00001': harmean_D_out_mhm_00001, 'whitenoise_00001': harmean_D_out_wn_00001}
arimeans = {'mhm_001': arimean_D_out_mhm_001, 'whitenoise_001': arimean_D_out_wn_001, 'mhm_00001': arimean_D_out_mhm_00001, 'whitenoise_00001': arimean_D_out_wn_00001}
data = {'whitenoise_001': D_1001_1100, 'mhm_001': D_1101_1100, 'whitenoise_00001': D_1201_1300,  'mhm_00001': D_1301_1400}
print(data)
print(geomeans)
names_geo = list(geomeans.keys())
values_geo = list(geomeans.values())
names_har = list(harmeans.keys())
values_har = list(harmeans.values())
names_ari = list(arimeans.keys())
values_ari = list(arimeans.values())
names_data = list(data.keys())
values_data = list(data.values())
plt.figure(figsize=(9,9))
plt.scatter(names_geo, values_geo, label = "SA single: geomean", color="blue", s=250)
plt.scatter(names_har, values_har, label = "SA single: harmean", color="orange", s=250)
plt.scatter(names_ari, values_ari, label = "SA single: arimean", color="red", s=250)
plt.scatter(names_data, values_data, label = "SA sum of baseflow", color="pink", marker="*", s=250)
plt.yscale('log')
plt.hlines(y=geomean_D_in_001, xmin=0, xmax=1, color="blue", label="geomean in")
plt.hlines(y=harmean_D_in_001, xmin=0, xmax=1, color="orange", label="harmean in")
plt.hlines(y=arimean_D_in_001, xmin=0, xmax=1, color="red", label="arimean in")
plt.hlines(y=geomean_D_in_00001, xmin=2, xmax=3, color="blue")#, label="geomean in, Ss = 0.0001")
plt.hlines(y=harmean_D_in_00001, xmin=2, xmax=3, color="orange")#, label="harmean in Ss = 0.0001")
plt.hlines(y=arimean_D_in_00001, xmin=2, xmax=3, color="red")#, label="arimean in, Ss = 0.0001")
plt.ylabel("Diffusivity [m^2/s]")
plt.tick_params(rotation=20)
plt.legend()
plt.savefig("/Users/houben/Desktop/baseflow_sa/in_vs_out_means_with_data.png", dpi=300)
